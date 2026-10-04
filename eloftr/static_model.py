"""Fixed-shape EfficientLoFTR for ONNX/TensorRT export.

Upstream inference produces a data-dependent number of matches (torch.where on the mutual-nearest
mask) and then indexes fine features by those matches. Here every coarse cell of image0 gets a
candidate match, so all shapes are static, and a `valid` mask marks the cells that upstream would
return. The backbone, coarse transformer and fine FPN are the upstream modules and weights; only
the coarse/fine matching logic is re-implemented, following upstream step by step.
"""
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval

# backbone: RepVGG CNN; coarse: coarse transformer; fine: fine FPN; fine_matching: matmuls of the fine windows
GROUPS = ('backbone', 'coarse', 'fine', 'fine_matching')
# mixed: fp16 for the convolutions and the transformer. The fine-window matmuls stay fp32: their fp16 outputs
# create ties in the fine argmax that measurably lower the keypoint accuracy (upstream runs them in fp16).
PRECISIONS = {'fp32': (), 'mixed': ('backbone', 'coarse', 'fine')}


class Fp32LayerNorm(nn.LayerNorm):
    """LayerNorm that always normalizes in fp32 and returns the input dtype."""

    def forward(self, x):
        return F.layer_norm(x.float(), self.normalized_shape, self.weight, self.bias, self.eps).to(x.dtype)


def precision_groups(precision, fp16=None):
    """Module groups that run in fp16, from a preset name or an explicit comma-separated list."""
    if fp16 is not None:
        groups = tuple(g for g in fp16.split(',') if g)
        unknown = set(groups) - set(GROUPS)
        if unknown:
            raise ValueError(f'Unknown fp16 groups {sorted(unknown)}, choose from {GROUPS}')
        return tuple(g for g in GROUPS if g in groups)
    if precision not in PRECISIONS:
        raise ValueError(f'Unknown precision {precision!r}, choose from {list(PRECISIONS)}')
    return PRECISIONS[precision]


def precision_label(fp16_groups):
    for name, groups in PRECISIONS.items():
        if tuple(fp16_groups) == groups:
            return name
    return 'fp16-' + '-'.join(fp16_groups)


def _logsumexp(x, dim):
    """Numerically stable logsumexp (the ONNX ReduceLogSumExp import in TensorRT does not subtract the max)."""
    m = x.max(dim=dim, keepdim=True)[0]
    return m + torch.log(torch.exp(x - m).sum(dim=dim, keepdim=True))


def _cast_module(module, dtype):
    """Cast parameters and floating buffers to dtype, keeping LayerNorms in fp32."""
    for m in module.modules():
        if isinstance(m, nn.LayerNorm):
            m.__class__ = Fp32LayerNorm
            continue
        for name, p in m.named_parameters(recurse=False):
            p.data = p.data.to(dtype)
        for name, b in m.named_buffers(recurse=False):
            if b.is_floating_point():
                m._buffers[name] = b.to(dtype)


def _fuse_fine_conv_bn(fine_preprocess):
    """Fold the BatchNorms of the fine FPN into their convolutions (exact in fp32, one rounding in fp16)."""
    for seq in (fine_preprocess.layer2_outconv2, fine_preprocess.layer1_outconv2):
        if isinstance(seq[0], nn.Conv2d) and isinstance(seq[1], nn.BatchNorm2d):
            seq[0] = fuse_conv_bn_eval(seq[0], seq[1])
            seq[1] = nn.Identity()


class StaticELoFTR(nn.Module):
    def __init__(self, upstream, height, width, fp16_groups=()):
        """
        Args:
            upstream: upstream LoFTR after `reparameter`, in eval mode (it is copied, not modified)
            height, width: fixed input size, multiples of 32
            fp16_groups: subset of GROUPS that runs in fp16; coarse similarity, LayerNorms and softmaxes stay fp32
        """
        super().__init__()
        if height % 32 or width % 32:
            raise ValueError(f'Input size must be a multiple of 32, got {height}x{width}')
        config = upstream.config
        if config['resolution'] != (8, 1) or config['fine_window_size'] != 8:
            raise ValueError('Only the released EfficientLoFTR configuration (resolution (8, 1), window 8) is supported')

        net = deepcopy(upstream)
        self.backbone = net.backbone
        self.loftr_coarse = net.loftr_coarse
        self.fine_preprocess = net.fine_preprocess
        _fuse_fine_conv_bn(self.fine_preprocess)

        self.height, self.width = height, width
        self.fp16_groups = tuple(fp16_groups)
        self.dtypes = {g: torch.float16 if g in self.fp16_groups else torch.float32 for g in GROUPS}
        _cast_module(self.backbone, self.dtypes['backbone'])
        _cast_module(self.loftr_coarse, self.dtypes['coarse'])
        _cast_module(self.fine_preprocess, self.dtypes['fine'])

        match_coarse = config['match_coarse']
        self.thr = match_coarse['thr']
        self.temperature = match_coarse['dsmax_temperature']
        self.skip_softmax = match_coarse['skip_softmax']
        self.slicedim = config['match_fine']['local_regress_slicedim']
        self.regress_temperature = config['match_fine']['local_regress_temperature']

        stride, win = 8, 8  # coarse stride (1/8) and fine window size
        self.stride, self.win = stride, win
        self.hc, self.wc = height // stride, width // stride
        n = self.hc * self.wc
        rows = torch.arange(self.hc).repeat_interleave(self.wc)
        cols = torch.arange(self.wc).repeat(self.hc)

        # upstream mask_border: drop cells within border_rm of the coarse grid border in both images
        b = match_coarse['border_rm']
        border = (rows >= b) & (rows < self.hc - b) & (cols >= b) & (cols < self.wc - b) if b > 0 else torch.ones(n, dtype=torch.bool)
        self.register_buffer('border', border, persistent=False)
        self.register_buffer('cell_ids', torch.arange(n), persistent=False)
        # coarse keypoints: [i % w, i // w] * scale
        self.register_buffer('coords_c', torch.stack([cols, rows], 1).float() * stride, persistent=False)

        # flat indices of the (win+2)^2 windows (unfold with padding 1, stride 8) in the padded image1 map
        padded_w = width + 2
        k = torch.arange(win + 2)
        self.register_buffer('window_base', rows * stride * padded_w + cols * stride, persistent=False)
        self.register_buffer('window_offsets', (k[:, None] * padded_w + k[None, :]).reshape(-1), persistent=False)

        # per argmax index over the cropped win^2 x win^2 matrix: idx_l = idx // ww, idx_r = idx % ww
        ww = win * win
        idx = torch.arange(ww * ww)
        idx_l, idx_r = idx // ww, idx % ww
        grid = torch.stack([torch.arange(ww) % win, torch.arange(ww) // win], 1).float() - win // 2 + 0.5
        self.register_buffer('delta_l', grid[idx_l], persistent=False)
        self.register_buffer('delta_r', grid[idx_r], persistent=False)
        # second stage: 3x3 neighbourhood around (idx_r // win, idx_r % win) in the (win+2)^2 conf_matrix_ff.
        # Upstream adds the -1..1 offsets directly to the cropped index, so -1 wraps around to win+1 as in
        # torch negative indexing; precomputing the indices keeps negative indices out of the graph.
        d = torch.tensor([-1, 0, 1])
        dr, dc = d.repeat_interleave(3), d.repeat(3)
        r = (idx_r // win)[:, None] + dr[None]
        c = (idx_r % win)[:, None] + dc[None]
        r, c = r % (win + 2), c % (win + 2)
        self.register_buffer('ff_index', idx_l[:, None] * (win + 2) ** 2 + r * (win + 2) + c, persistent=False)
        # kornia dsnt.spatial_expectation2d on a 3x3 heatmap with normalized coordinates
        self.register_buffer('expect_xy', torch.stack([dc, dr], 1).float(), persistent=False)

    def forward(self, image0, image1):
        """
        Args:
            image0, image1: [1, 1, H, W] grayscale images in [0, 1]
        Returns:
            keypoints0, keypoints1: [L, 2] (x, y) in input pixels, L = H/8 * W/8 (one candidate per coarse cell of image0)
            confidence: [L] coarse confidence (dual-softmax score, or raw similarity for the opt model)
            valid: [L] bool, True for the matches upstream would return
        """
        # 1. local feature CNN
        x = torch.cat([image0, image1], 0).to(self.dtypes['backbone'])
        feats = self.backbone(x)

        # 2. coarse-level transformer
        feat_c = feats['feats_c'].to(self.dtypes['coarse'])
        feat_c0, feat_c1 = self.loftr_coarse(feat_c[:1], feat_c[1:])
        channels = feat_c0.shape[1]

        # 3. coarse matching in fp32: dual softmax, threshold, border and mutual nearest neighbour.
        # sim and its transpose come from two matmuls so that both softmaxes and all reductions run along
        # the last axis (faster in TensorRT than reducing a 4800 x 4800 matrix along its first axis).
        f0 = feat_c0.flatten(2).transpose(1, 2).float()[0] / channels ** .5
        f1 = feat_c1.flatten(2).transpose(1, 2).float()[0] / channels ** .5
        sim = torch.matmul(f0, f1.t()) / self.temperature
        sim_t = torch.matmul(f1, f0.t()) / self.temperature
        if self.skip_softmax:
            conf, conf_t = sim, sim_t
        else:
            softmax_rows, softmax_cols = F.softmax(sim, 1), F.softmax(sim_t, 1)
            conf, conf_t = softmax_rows * softmax_cols.t(), softmax_cols * softmax_rows.t()
        confidence, j_ids = conf.max(dim=1)
        i_back = conf_t.argmax(dim=1)
        valid = (confidence > self.thr) & self.border & self.border[j_ids] & (i_back[j_ids] == self.cell_ids)

        # 4. fine features: upstream scales coarse features by 1/sqrt(C) before the fine FPN
        dtype = self.dtypes['fine']
        feat_c = torch.cat([feat_c0, feat_c1], 0) / channels ** .5
        feat_f = self.fine_preprocess.inter_fpn(feat_c.to(dtype), feats['feats_x2'].to(dtype),
                                                feats['feats_x1'].to(dtype), self.stride)
        cf = feat_f.shape[1]
        win, n = self.win, self.hc * self.wc
        # image0: non-overlapping win x win windows, ordered like F.unfold (row-major inside the window)
        win0 = feat_f[0].reshape(cf, self.hc, win, self.wc, win).permute(1, 3, 2, 4, 0).reshape(n, win * win, cf)
        # image1: (win+2)^2 windows around the matched coarse cells j
        padded = F.pad(feat_f[1], (1, 1, 1, 1)).flatten(1).transpose(0, 1)
        index = (self.window_base[j_ids][:, None] + self.window_offsets[None]).reshape(-1)
        win1 = padded.index_select(0, index).reshape(n, (win + 2) ** 2, cf)

        # 5. fine matching (upstream FineMatching). The argmax of the dual softmax is taken in log space,
        # argmax(softmax_1 * softmax_2) = argmax(2 x - logsumexp_1 - logsumexp_2), saving the two products.
        win0, win1 = win0.to(self.dtypes['fine_matching']), win1.to(self.dtypes['fine_matching'])
        s = self.slicedim
        conf_f = torch.matmul(win0[..., :-s] / cf ** .5, (win1[..., :-s] / cf ** .5).transpose(1, 2)).float()
        conf_ff = torch.matmul(win0[..., -s:], (win1[..., -s:] / s ** .5).transpose(1, 2))
        score_f = 2 * conf_f - _logsumexp(conf_f, 1) - _logsumexp(conf_f, 2)
        score_f = score_f.reshape(n, win * win, win + 2, win + 2)[..., 1:-1, 1:-1].reshape(n, -1)
        idx = score_f.argmax(dim=1)

        heatmap = torch.gather(conf_ff.reshape(n, -1), 1, self.ff_index[idx]).float()
        heatmap = F.softmax(heatmap / self.regress_temperature, -1)
        coords = torch.matmul(heatmap, self.expect_xy)

        keypoints0 = self.coords_c + self.delta_l[idx]
        keypoints1 = self.coords_c[j_ids] + self.delta_r[idx] + coords
        return keypoints0, keypoints1, confidence, valid
