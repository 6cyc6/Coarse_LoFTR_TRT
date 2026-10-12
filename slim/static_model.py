"""Fixed-shape SLiM for ONNX/TensorRT export.

Upstream test inference (src/slim.py, _forward_test) selects coarse candidates by feature norm, keeps a data-dependent
number of coarse matches (flat top-k of the dual softmax over the candidates) and fine matches (per-window top-k with
k depending on the number of coarse matches), and refines the fine matches in 4800 padded slots. Here every step has a
static shape: candidates are masks, the top-k keep their maximum size with a `valid` mask, and the valid fine matches
are compacted in upstream order into the 4800 refinement slots. The backbone and the refinement unit are the
upstream modules and weights; the Triton cross-scan becomes plain tensor ops, the selective scan the `SelectiveScan`
op (TensorRT plugin) and, for export, Mamba2 the plain torch form of slim.mamba2.
"""
import types
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from slim.mamba2 import mamba2_forward
from slim.ops import selective_scan
from slim.upstream import import_upstream

# backbone: ConvVMamba (convolutions, SS2D and Mamba2 projections, FPN); refine: recurrent refinement unit
GROUPS = ('backbone', 'refine')
# mixed: fp16 for the convolutions and projections. Scans, similarities, softmaxes, LayerNorms, the SSM parameters
# and the sampling of the refinement windows stay fp32.
PRECISIONS = {'fp32': (), 'mixed': GROUPS}
CANDIDATE_RATIO = 0.65  # upstream _coarse_match_norm_clue
# SSM parameters that upstream keeps in fp32 under autocast
FP32_PARAMETERS = ('A_logs', 'Ds', 'dt_projs_bias', 'A_log', 'D', 'dt_bias')


def _fp32_layer_norm(module):
    """Run a LayerNorm (or one of SLiM's LayerNorm2d subclasses) in fp32 and return the input dtype."""
    forward = type(module).forward
    module.forward = types.MethodType(lambda self, x: forward(self, x.float()).to(x.dtype), module)


def cast_module(module, dtype):
    """Cast parameters and floating buffers to dtype, keeping LayerNorms and SSM parameters in fp32."""
    for m in module.modules():
        if isinstance(m, nn.LayerNorm):
            _fp32_layer_norm(m)
            continue
        for name, p in m.named_parameters(recurse=False):
            if name not in FP32_PARAMETERS:
                p.data = p.data.to(dtype)
        for name, b in m.named_buffers(recurse=False):
            if b.is_floating_point():
                m._buffers[name] = b.to(dtype)


def cross_scan(x):
    """[B, D, H, W] -> [B, 4, D, H*W]: row-major, column-major and both reversed (csm_triton.cross_scan_fwd)."""
    x_hw, x_wh = x.flatten(2), x.transpose(2, 3).flatten(2)
    return torch.stack([x_hw, x_wh, x_hw.flip(-1), x_wh.flip(-1)], 1)


def cross_merge(y, H, W):
    """[B, 4, D, H*W] -> [B, D, H*W], the sum of the 4 scans in row-major order (csm_triton.cross_merge_fwd)."""
    B, _, D, L = y.shape
    y = y[:, 0:2] + y[:, 2:4].flip(-1)
    return y[:, 0] + y[:, 1].reshape(B, D, W, H).transpose(2, 3).reshape(B, D, L)


def ss2d_forward(self, x):
    """Easy_SS2D.forwardSS2D with the torch cross-scan and the SelectiveScan op."""
    B, D, H, W = x.shape
    N, K, R, L = self.d_state, self.k_group, self.dt_rank, H * W
    xs = cross_scan(x).reshape(B, K * D, L)
    x_dbl = F.conv1d(xs, self.x_proj_weight.reshape(-1, D, 1), groups=K)
    dts, Bs, Cs = torch.split(x_dbl.reshape(B, K, -1, L), [R, N, N], dim=2)
    dts = F.conv1d(dts.reshape(B, -1, L), self.dt_projs_weight.reshape(K * D, -1, 1), groups=K)
    ys = selective_scan(xs, dts, -self.A_logs.float().exp(), Bs.reshape(B, K, N, L), Cs.reshape(B, K, N, L),
                        self.Ds.float(), self.dt_projs_bias.reshape(-1).float())
    return cross_merge(ys.reshape(B, K, D, L), H, W).reshape(B, D, H, W).to(x.dtype)


def _upsample_sum(not_upsample_feature, upsample_feature):
    """ConvVMamba.upsample_sum without TorchScript."""
    H, W = not_upsample_feature.shape[2:]
    return not_upsample_feature + F.interpolate(upsample_feature, size=(H, W), mode='bilinear', align_corners=False)


def side_by_side(x):
    """[2, C, H, W] (image0, image1) -> [1, C, H, 2W], the images side by side as upstream's stages see them."""
    _, C, H, W = x.shape
    return x.permute(1, 2, 0, 3).reshape(1, C, H, 2 * W)


def batched(x):
    """[1, C, H, 2W] -> [2, C, H, W], the inverse of side_by_side."""
    _, C, H, W2 = x.shape
    return x.reshape(C, H, 2, W2 // 2).permute(2, 0, 1, 3)


def backbone_forward(bb, image0, image1):
    """ConvVMamba.forward, returning the batched (image0, image1) FPN levels [f1 (1/2), f2 (1/4), coarse (1/8)].

    Upstream moves between the batched and the side-by-side layout with chunk and concat. TensorRT 10.9 miscomputes
    a slice-and-concatenate of a stage output that the next stage also consumes (the copy reads overwritten memory),
    so the layouts are converted with reshapes and transposes here (same values).
    """
    first = bb.first_stage(torch.cat([image0, image1], 0))
    x = side_by_side(bb.pe(first))
    levels = [first]
    for stage in bb.stages:
        x = stage(x)
        levels.append(batched(x))
    f3 = bb.output_conv_3(levels[2])
    f2 = bb.output_conv_2(_upsample_sum(bb.lateral_conv_2(levels[1]), f3))
    f1 = bb.output_conv_1(_upsample_sum(bb.lateral_conv_1(levels[0]), f2))
    return f1, f2, levels[2]


class StaticPE(nn.Module):
    """SizeInvariantPE at a fixed size on the batched (image0, image1) features: the interpolated position encodings
    are precomputed, and the last channel encodes the image index (0 for image0, 1 for image1)."""

    def __init__(self, pe, height, width):
        super().__init__()
        enc = F.interpolate(pe.pos_emb[:, :pe.d_model], size=(height, width), mode='bilinear', align_corners=False)
        enc[:, -1] = 0
        index = torch.zeros(2, pe.d_model, 1, 1)
        index[1, -1] = 1
        self.register_buffer('enc', enc, persistent=False)
        self.register_buffer('index', index, persistent=False)
        self.gamma = pe.gamma

    def forward(self, x):
        return x + (self.enc * self.gamma).to(x.dtype) + (self.index * self.gamma).to(x.dtype)


class StaticSLiM(nn.Module):
    def __init__(self, upstream, height=480, width=640, fp16_groups=(), exportable=True):
        """
        Args:
            upstream: upstream SLiM (slim.upstream.load_upstream), in eval mode (it is copied, not modified)
            height, width: fixed input size, multiples of 32
            fp16_groups: subset of GROUPS that runs in fp16
            exportable: compute Mamba2 in plain torch ops (slim.mamba2); False keeps mamba_ssm's fused kernels,
                which use TF32 internally and reproduce upstream exactly (self-check of the matching)
        """
        super().__init__()
        if height % 32 or width % 32:
            raise ValueError(f'Input size must be a multiple of 32, got {height}x{width}')
        if upstream.coarse_scale_idx != 2 or upstream.fine_scale_idx != 0:
            raise ValueError('Only the released SLiM configuration (coarse 1/8, fine 1/2 upsampled to 1/1) is supported')
        convmamba = import_upstream()['convmamba']
        from mamba_ssm import Mamba2

        net = deepcopy(upstream)
        self.backbone = net.feature_backbone
        self.rru = net.recurrent_refine_unit
        self.coarse_temperature = net.coarse_temperature
        self.fine_temperature = net.fine_temperature
        self.backbone.pe = StaticPE(self.backbone.pe, height // 2, width // 2)
        for m in self.backbone.modules():
            if isinstance(m, convmamba.Easy_SS2D):
                m.forwardSS2D = types.MethodType(ss2d_forward, m)
            elif isinstance(m, Mamba2) and exportable:
                m.forward = types.MethodType(mamba2_forward, m)

        self.height, self.width = height, width
        self.fp16_groups = tuple(fp16_groups)
        self.dtypes = {g: torch.float16 if g in self.fp16_groups else torch.float32 for g in GROUPS}
        cast_module(self.backbone, self.dtypes['backbone'])
        cast_module(self.rru, self.dtypes['refine'])

        self.coarse_thr, self.fine_thr = upstream.coarse_match_thres, upstream.fine_match_thres
        self.max_coarse, self.max_fine = upstream.max_coarse_matches, upstream.max_fine_matches
        self.refine_iters, self.radius = upstream.refine_iters, upstream.refine_lookup_radius
        self.lookup = upstream.lookup_window_size  # 6

        stride, win = 8, 8  # coarse scale and fine window (coarse / fine scale, fine features at full resolution)
        self.stride, self.win = stride, win
        self.hc, self.wc = height // stride, width // stride
        rows = torch.arange(win).repeat_interleave(win)
        cols = torch.arange(win).repeat(win)
        # fine window offsets in the flattened full-resolution map and their (x, y) within the window, row-major
        self.register_buffer('window_offsets', rows * width + cols, persistent=False)
        self.register_buffer('window_xy', torch.stack([cols, rows], 1).float(), persistent=False)
        # bilinear refinement windows: lookup x lookup samples centred on the coordinates, (x, y) in input pixels
        offsets = torch.arange(self.lookup).float() - (self.lookup - 1) / 2
        grid = torch.stack([offsets[None, :].expand(self.lookup, -1), offsets[:, None].expand(-1, self.lookup)], -1)
        self.register_buffer('lookup_grid', grid, persistent=False)  # [lookup (h), lookup (w), 2]
        self.register_buffer('coarse_rank', torch.arange(self.max_coarse), persistent=False)
        self.register_buffer('fine_rank', torch.arange(win * win), persistent=False)

    def _coarse_matches(self, feat0, feat1):
        """_coarse_match_norm_clue: (i, j) cells [K], confidence [K] and valid [K], K = max coarse matches."""
        C = feat0.shape[1]
        f0 = feat0.flatten(2)[0].t().float() / C ** .5  # [Lc, C]
        f1 = feat1.flatten(2)[0].t().float() / C ** .5
        norm0, norm1 = f0.norm(dim=-1), f1.norm(dim=-1)
        cand0, cand1 = norm0 > norm0.max() * CANDIDATE_RATIO, norm1 > norm1.max() * CANDIDATE_RATIO
        # upstream falls back to the norm above the mean of image0 (for both images) without candidates
        empty = torch.minimum(cand0.sum(), cand1.sum()) == 0
        cand0 = torch.where(empty, norm0 > norm0.mean(), cand0)
        cand1 = torch.where(empty, norm1 > norm0.mean(), cand1)
        min_len = torch.minimum(cand0.sum(), cand1.sum())

        # dual softmax restricted to the candidate rows and columns
        mask = cand0[:, None] & cand1[None, :]
        sim = (f0 @ f1.t() / (self.coarse_temperature.float() + 1e-8)).masked_fill(~mask, -1e9)
        conf = F.softmax(sim, 0) * F.softmax(sim, 1) * mask

        # flat top-k of the [Lc, Lc] confidence, exactly but without a top-k over all Lc^2 entries (slow in
        # TensorRT): the k largest entries lie in the k rows with the largest maxima, since those rows alone hold
        # k entries at least as large as the maximum of any other row
        k = self.max_coarse
        _, rows = torch.topk(conf.max(dim=1).values, k)
        row_values, cols = torch.topk(conf.index_select(0, rows), k, dim=1)
        values, index = torch.topk(row_values.flatten(), k)
        valid = (self.coarse_rank < min_len) & (values > values[0] * self.coarse_thr)
        return rows[index // k], cols.flatten()[index], valid

    def _fine_matches(self, feat0, feat1, i, j, valid_c):
        """_fine_correlation + _get_fine_coord_test: intermediate keypoints [K, w*w, 2], confidence, valid."""
        W = self.width
        cf = feat0.shape[1]
        coords0 = torch.stack([i % self.wc, i // self.wc], 1) * self.stride  # [K, 2] top-left of the coarse cells
        coords1 = torch.stack([j % self.wc, j // self.wc], 1) * self.stride
        flat0, flat1 = (f[0].flatten(1).t() for f in (feat0, feat1))  # [H*W, C]
        base0, base1 = coords0[:, 1] * W + coords0[:, 0], coords1[:, 1] * W + coords1[:, 0]
        win0 = flat0.index_select(0, (base0[:, None] + self.window_offsets).reshape(-1)).reshape(-1, self.win ** 2, cf)
        win1 = flat1.index_select(0, (base1[:, None] + self.window_offsets).reshape(-1)).reshape(-1, self.win ** 2, cf)
        sim = torch.matmul(win0 / cf ** .5, (win1 / cf ** .5).transpose(1, 2)) / (self.fine_temperature.float() + 1e-8)
        conf = F.softmax(sim, 1) * F.softmax(sim, 2)  # [K, w*w, w*w]

        n_coarse = valid_c.sum()
        k = torch.clamp(torch.floor(self.max_fine / torch.clamp(n_coarse, min=1).float()), max=self.win ** 2)
        values, index = torch.topk(conf.reshape(conf.shape[0], -1), self.win ** 2, dim=1)  # [K, w*w]
        global_max = torch.where(valid_c, values[:, 0], torch.zeros_like(values[:, 0])).max()
        valid = valid_c[:, None] & (self.fine_rank[None] < k) & (values > global_max * self.fine_thr)
        n = self.win ** 2
        kpts0 = coords0[:, None].float() + 0.5 + self.window_xy[index // n]  # [K, w*w, 2]
        kpts1 = coords1[:, None].float() + 0.5 + self.window_xy[index % n]
        return kpts0, kpts1, values, valid

    def _compact(self, valid, *tensors):
        """Valid entries of [K, k, ...] tensors in upstream (row-major) order into max_fine slots, and their mask.

        There are at most max_fine valid entries (K valid coarse matches keep max_fine / K each). The scatter buffer
        is as large as its index tensor, which TensorRT requires, and invalid entries go to its last slot.
        """
        valid = valid.reshape(-1)
        n = valid.shape[0]
        slots = torch.where(valid, torch.cumsum(valid.int(), 0) - 1, n)
        out = []
        for t in tensors + (valid.float(),):
            t = t.reshape(n, -1)
            buffer = torch.zeros(n + 1, t.shape[1], dtype=t.dtype, device=t.device)
            out.append(buffer.scatter(0, slots[:, None].expand(-1, t.shape[1]), t)[:self.max_fine])
        *out, mask = out
        return out, mask[:, 0] > 0

    def _windows(self, feat, coords):
        """_extract_feat_window_bilinear: lookup x lookup bilinear windows [M, C, lookup, lookup] around coords."""
        M = coords.shape[0]
        grid = coords[:, None, None, :] + self.lookup_grid  # [M, h, w, 2] (x, y)
        scale = torch.tensor([feat.shape[3], feat.shape[2]], dtype=grid.dtype, device=grid.device)
        grid = (grid / scale * 2 - 1).reshape(1, M * self.lookup, self.lookup, 2)
        windows = F.grid_sample(feat, grid, mode='bilinear', align_corners=False, padding_mode='zeros')
        return windows.reshape(feat.shape[1], M, self.lookup, self.lookup).permute(1, 0, 2, 3)

    def forward(self, image0, image1):
        """
        Args:
            image0, image1: [1, 1, H, W] grayscale images in [0, 1]
        Returns:
            keypoints0, keypoints1: [N, 2] (x, y) in input pixels, N = max fine matches (4800)
            confidence: [N] fine dual-softmax confidence
            valid: [N] bool, True for the matches upstream returns
        """
        dtype = self.dtypes['backbone']
        fine, _, coarse = backbone_forward(self.backbone, image0.to(dtype), image1.to(dtype))
        i, j, valid_c = self._coarse_matches(coarse[:1], coarse[1:])

        # fine features: the 1/2 FPN level upsampled to the input resolution
        fine = F.interpolate(fine.float(), scale_factor=2.0, mode='bilinear', align_corners=False)
        fine0, fine1 = fine[:1], fine[1:]
        kpts0, kpts1, conf, valid_f = self._fine_matches(fine0, fine1, i, j, valid_c)
        (kpts0, kpts1, conf), valid = self._compact(valid_f, kpts0, kpts1, conf)

        # recurrent refinement of the keypoints in image1
        dtype = self.dtypes['refine']
        window0 = self._windows(fine0, kpts0).to(dtype)
        hidden = torch.zeros(self.max_fine, self.rru.hidden_dim, self.lookup, self.lookup, dtype=dtype,
                             device=image0.device)
        for _ in range(self.refine_iters):
            offset, hidden = self.rru(window0, self._windows(fine1, kpts1).to(dtype), hidden)
            kpts1 = kpts1 + offset.float() * self.radius
        return kpts0, kpts1, conf[:, 0], valid
