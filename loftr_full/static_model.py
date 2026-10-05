"""Fixed-shape original LoFTR (kornia.feature.LoFTR) for ONNX/TensorRT export.

Same approach as eloftr.static_model.StaticELoFTR: upstream inference produces a data-dependent number of
coarse matches and then crops fine windows around them. Here every coarse cell of image0 gets a candidate
match, so all shapes are static, and a `valid` mask marks the cells that upstream would return. The backbone,
both transformers and the fine preprocessing layers are the upstream modules and weights; the linear attention
is re-expressed without einsum, and the coarse/fine matching is re-implemented following upstream step by step.
"""
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from eloftr.static_model import cast_module

# backbone: ResNet-FPN; coarse: coarse transformer; fine: fine window projections and fine transformer
GROUPS = ('backbone', 'coarse', 'fine')
# mixed: fp16 for the convolutions and the linear layers of both transformers. The linear attention itself,
# the coarse similarity, LayerNorms, softmaxes and the fine expectation always run in fp32.
PRECISIONS = {'fp32': (), 'mixed': GROUPS}


class MatmulLinearAttention(nn.Module):
    """Upstream LinearAttention as batched matmuls (TensorRT parses no 3-operand einsum), computed in fp32.

    The normalizer sums the keys over the whole sequence (4800 tokens at 640x480), which can exceed the fp16
    range; the attention is cheap next to the projections around it.
    """

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """queries [N, L, H, D], keys and values [N, S, H, D] -> [N, L, H, D]"""
        if q_mask is not None or kv_mask is not None:
            raise ValueError('Masks (padded inputs) are not supported')
        q = F.elu(queries.float()).add(1).transpose(1, 2)  # [N, H, L, D]
        k = F.elu(keys.float()).add(1).transpose(1, 2)  # [N, H, S, D]
        v = values.float().transpose(1, 2)
        length = v.shape[2]
        kv = torch.matmul(k.transpose(2, 3), v / length)  # [N, H, D, D]
        z = 1 / (torch.matmul(q, k.sum(dim=2, keepdim=True).transpose(2, 3)) + self.eps)  # [N, H, L, 1]
        out = torch.matmul(q, kv) * z * length
        return out.transpose(1, 2).to(queries.dtype).contiguous()


class StaticLoFTR(nn.Module):
    def __init__(self, upstream, height, width, fp16_groups=()):
        """
        Args:
            upstream: kornia LoFTR with loaded weights, in eval mode (it is copied, not modified)
            height, width: fixed input size, multiples of 8
            fp16_groups: subset of GROUPS that runs in fp16
        """
        super().__init__()
        if height % 8 or width % 8:
            raise ValueError(f'Input size must be a multiple of 8, got {height}x{width}')
        config = upstream.config
        if (tuple(config['resolution']) != (8, 2) or config['fine_window_size'] != 5
                or not config['fine_concat_coarse_feat'] or config['match_coarse']['match_type'] != 'dual_softmax'
                or config['coarse']['attention'] != 'linear' or config['fine']['attention'] != 'linear'):
            raise ValueError('Only the released LoFTR configuration (resolution (8, 2), window 5, dual softmax, '
                             'linear attention) is supported')

        net = deepcopy(upstream)
        self.backbone = net.backbone
        self.loftr_coarse = net.loftr_coarse
        self.fine_preprocess = net.fine_preprocess
        self.loftr_fine = net.loftr_fine
        for transformer in (self.loftr_coarse, self.loftr_fine):
            for layer in transformer.layers:
                layer.attention = MatmulLinearAttention(layer.attention.eps)

        self.height, self.width = height, width
        self.fp16_groups = tuple(fp16_groups)
        self.dtypes = {g: torch.float16 if g in self.fp16_groups else torch.float32 for g in GROUPS}
        cast_module(self.backbone, self.dtypes['backbone'])
        cast_module(self.loftr_coarse, self.dtypes['coarse'])
        cast_module(self.fine_preprocess, self.dtypes['fine'])
        cast_module(self.loftr_fine, self.dtypes['fine'])

        match_coarse = config['match_coarse']
        self.thr = match_coarse['thr']
        self.temperature = match_coarse['dsmax_temperature']

        stride, win = 8, 5  # coarse stride (1/8) and fine window size on the 1/2 feature map
        fine_stride = stride // 2  # coarse cell size on the 1/2 feature map
        self.stride, self.win = stride, win
        self.hc, self.wc = height // stride, width // stride
        n = self.hc * self.wc
        rows = torch.arange(self.hc).repeat_interleave(self.wc)
        cols = torch.arange(self.wc).repeat(self.hc)

        # sine positional encoding of the coarse grid; upstream grows `pe` beyond 256 cells the same way
        pos_encoding = net.pos_encoding
        if self.hc > pos_encoding.pe.shape[2] or self.wc > pos_encoding.pe.shape[3]:
            pos_encoding.update_position_encoding_size((max(self.hc, pos_encoding.pe.shape[2]),
                                                        max(self.wc, pos_encoding.pe.shape[3])))
        self.register_buffer('pe', pos_encoding.pe[:, :, :self.hc, :self.wc].clone(), persistent=False)

        # upstream mask_border: drop cells within border_rm of the coarse grid border in both images
        b = match_coarse['border_rm']
        border = (rows >= b) & (rows < self.hc - b) & (cols >= b) & (cols < self.wc - b) if b > 0 else torch.ones(n, dtype=torch.bool)
        self.register_buffer('border', border, persistent=False)
        self.register_buffer('cell_ids', torch.arange(n), persistent=False)
        # coarse keypoints: [i % w, i // w] * scale
        self.register_buffer('coords_c', torch.stack([cols, rows], 1).float() * stride, persistent=False)

        # flat indices of the win x win windows (upstream: unfold with padding win // 2, stride 4) in the padded
        # 1/2 feature map, row-major inside the window; image0 uses all cells, image1 the matched cells j
        padded_w = width // 2 + 2 * (win // 2)
        k = torch.arange(win)
        window_base = rows * fine_stride * padded_w + cols * fine_stride
        window_offsets = (k[:, None] * padded_w + k[None, :]).reshape(-1)
        self.register_buffer('window_base', window_base, persistent=False)
        self.register_buffer('window_offsets', window_offsets, persistent=False)
        self.register_buffer('window_index0', (window_base[:, None] + window_offsets[None]).reshape(-1), persistent=False)
        # kornia dsnt.spatial_expectation2d on the win x win heatmap with normalized (x, y) coordinates
        grid = torch.linspace(-1, 1, win)
        self.register_buffer('expect_xy', torch.stack([grid.repeat(win), grid.repeat_interleave(win)], 1), persistent=False)
        # normalized window coordinates to input pixels: win // 2 cells of the 1/2 map, 2 pixels each
        self.fine_scale = (win // 2) * (stride // fine_stride)

    def forward(self, image0, image1):
        """
        Args:
            image0, image1: [1, 1, H, W] grayscale images in [0, 1]
        Returns:
            keypoints0, keypoints1: [L, 2] (x, y) in input pixels, L = H/8 * W/8 (one candidate per coarse cell of image0)
            confidence: [L] dual-softmax confidence
            valid: [L] bool, True for the matches upstream would return
        """
        # 1. local feature CNN: 1/8 coarse and 1/2 fine feature maps
        x = torch.cat([image0, image1], 0).to(self.dtypes['backbone'])
        feat_c, feat_f = self.backbone(x)

        # 2. coarse-level transformer on the positionally encoded [1, L, C] sequences
        feat_c = (feat_c.float() + self.pe).to(self.dtypes['coarse']).flatten(2).transpose(1, 2)
        feat_c0, feat_c1 = self.loftr_coarse(feat_c[:1], feat_c[1:])
        channels = feat_c0.shape[2]

        # 3. coarse matching in fp32: dual softmax, threshold, border and mutual nearest neighbour.
        # sim and its transpose come from two matmuls so that both softmaxes and all reductions run along
        # the last axis (faster in TensorRT than reducing a 4800 x 4800 matrix along its first axis).
        f0 = feat_c0[0].float() / channels ** .5
        f1 = feat_c1[0].float() / channels ** .5
        sim = torch.matmul(f0, f1.t()) / self.temperature
        sim_t = torch.matmul(f1, f0.t()) / self.temperature
        softmax_rows, softmax_cols = F.softmax(sim, 1), F.softmax(sim_t, 1)
        conf, conf_t = softmax_rows * softmax_cols.t(), softmax_cols * softmax_rows.t()
        confidence, j_ids = conf.max(dim=1)
        i_back = conf_t.argmax(dim=1)
        valid = (confidence > self.thr) & self.border & self.border[j_ids] & (i_back[j_ids] == self.cell_ids)

        # 4. fine windows (upstream FinePreprocess): crops of the 1/2 maps, merged with the coarse features of
        # their cells (the transformer outputs, without the 1/sqrt(C) of the coarse matching)
        dtype = self.dtypes['fine']
        n, ww = self.hc * self.wc, self.win ** 2
        pad = self.win // 2
        padded = F.pad(feat_f.to(dtype), (pad, pad, pad, pad)).flatten(2).transpose(1, 2)  # [2, P, Cf]
        cf = padded.shape[2]
        win0 = padded[0].index_select(0, self.window_index0).reshape(n, ww, cf)
        index1 = (self.window_base[j_ids][:, None] + self.window_offsets[None]).reshape(-1)
        win1 = padded[1].index_select(0, index1).reshape(n, ww, cf)
        feat_c_win = self.fine_preprocess.down_proj(torch.cat([feat_c0[0], feat_c1[0].index_select(0, j_ids)], 0).to(dtype))
        feat_cf = self.fine_preprocess.merge_feat(
            torch.cat([torch.cat([win0, win1], 0), feat_c_win[:, None].expand(-1, ww, -1)], -1))

        # 5. fine-level transformer inside each pair of windows
        win0, win1 = self.loftr_fine(feat_cf[:n], feat_cf[n:])

        # 6. fine matching in fp32 (upstream FineMatching): softmax over image1's window of the similarity to the
        # centre feature of image0's window, and its expected position
        center = win0[:, ww // 2].float()
        sim_f = torch.matmul(win1.float(), center[:, :, None])[..., 0] / cf ** .5
        coords = torch.matmul(F.softmax(sim_f, 1), self.expect_xy)

        keypoints0 = self.coords_c
        keypoints1 = self.coords_c[j_ids] + coords * self.fine_scale
        return keypoints0, keypoints1, confidence, valid
