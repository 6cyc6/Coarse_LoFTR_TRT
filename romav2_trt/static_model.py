"""Fixed-shape RoMa v2 for ONNX/TensorRT export.

Upstream RoMa v2 already has static shapes for a given input size: it predicts, for every pixel of image0, its position
in image1 (a dense warp) and a confidence (overlap logit and the parameters of a 2x2 precision matrix), and samples
matches from them afterwards. This model is its single-resolution forward (`RoMaV2.forward(img_A, img_B)` without the
high-resolution pass), with the upstream modules and weights. What upstream computes from the input size on every call
(RoPE tables, positional embedding grid, coordinate grids) is precomputed, DINOv3 stops after its last used block, and
the transformer blocks, DPT fusion and refiner local correlation are re-expressed with explicit shapes and positive dims
for the ONNX export. Both directions of a bidirectional model run as a batch of two.
"""
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval

from loma_trt.static_model import PRECISIONS, autocast_module, cast, fold_batchnorms, half_groups, interpolate
from romav2_trt.upstream import import_upstream

# dinov3: DINOv3 ViT-L/16 features (upstream casts it to bf16); matcher: multi-view ViT; head: DPT head; refiner:
# VGG19-BN features and the refiner convolutions. These are the parts upstream runs in bf16 autocast. The global
# matching, the refiner projections, sampling, local correlation and heads, and all warp and confidence arithmetic stay
# fp32.
GROUPS = ('dinov3', 'matcher', 'head', 'refiner')
# DINOv3's residual stream holds activations of ~1.6e5, beyond the fp16 range: in fp16 models it stays bf16 (upstream
# runs everything in bf16).
FP16_FALLBACK = {'dinov3': torch.bfloat16}
PATCH = 16


def group_dtypes(precision, groups=None):
    """dtype of each module group for a precision and an optional comma-separated subset of half-precision groups."""
    half = PRECISIONS[precision]
    selected = half_groups(precision, groups, GROUPS)
    return {g: (FP16_FALLBACK.get(g, half) if half == torch.float16 else half) if g in selected else torch.float32
            for g in GROUPS}


class StaticRoMaV2(nn.Module):
    def __init__(self, upstream, height, width, precision='fp32', groups=None, bidirectional=False, batch_images=True):
        """
        Args:
            upstream: upstream RoMaV2 (romav2_trt.upstream.load_upstream, preferably fp32=True); copied, not modified
            height, width: fixed input size, multiples of 16
            precision: fp32, bf16 or fp16 (the half dtype of the groups that upstream autocasts)
            groups: comma-separated module groups in half precision, default all of GROUPS
            bidirectional: also predict the warp of image1 into image0
            batch_images: run DINOv3 on both images as one batch (faster). Upstream runs it per image; the matcher
                applies RoPE in bf16 even in fp32, which turns the ~1e-6 rounding differences of batching into ~1e-3
                ones, so the exact comparison with upstream (export self-check) runs it per image.
        """
        super().__init__()
        if height % PATCH or width % PATCH:
            raise ValueError(f'Input size must be a multiple of {PATCH}, got {height}x{width}')
        self.height, self.width = height, width
        self.h, self.w = height // PATCH, width // PATCH
        self.bidirectional, self.batch_images = bidirectional, batch_images
        self.directions = 2 if bidirectional else 1
        self.dtypes = group_dtypes(precision, groups)

        import_upstream()
        from romav2.geometry import get_normalized_grid
        dino = upstream.f
        layers = [i if i >= 0 else len(dino.blocks) + i for i in upstream.cfg.descriptor.layer_idx]
        self.dino_layers = sorted(layers)
        with torch.no_grad():  # RoPE tables for this size: DINOv3's in fp32, the multi-view ViT's in bf16 (upstream)
            dino_rope = dino.rope_embed(H=self.h, W=self.w)
            mv_rope = upstream.matcher.mv_vit.rope_embed(H=self.h, W=self.w)
            # random Fourier positional embedding of image1's patch grid (Matcher.forward)
            grid = get_normalized_grid(1, self.h, self.w).reshape(self.h * self.w, 2)
            x_emb = F.linear(grid, upstream.matcher.scale * upstream.matcher.omega)
            pos_emb_grid = torch.cat((x_emb.sin(), x_emb.cos()), 1)
        self.register_buffer('dino_sin', dino_rope[0].cpu(), persistent=False)
        self.register_buffer('dino_cos', dino_rope[1].cpu(), persistent=False)
        self.register_buffer('mv_sin', mv_rope[0].cpu(), persistent=False)
        self.register_buffer('mv_cos', mv_rope[1].cpu(), persistent=False)
        self.register_buffer('pos_emb_grid', pos_emb_grid.cpu(), persistent=False)
        self.register_buffer('inv_temp', (1 / upstream.matcher.temp).float().cpu(), persistent=False)

        net = deepcopy(upstream)
        dino = net.f
        self.patch_proj, self.dino_norm = dino.patch_embed.proj, dino.norm
        if not isinstance(dino.patch_embed.norm, nn.Identity):
            raise ValueError('Only DINOv3 without a patch-embedding norm is supported')
        self.dino_blocks = nn.ModuleList(dino.blocks[:self.dino_layers[-1] + 1])  # later blocks are unused
        self.dino_prefix = 1 + dino.n_storage_tokens
        with torch.no_grad():
            cls_token = dino.cls_token + 0 * dino.mask_token  # prepare_tokens_with_masks without masks
            storage = dino.storage_tokens if dino.n_storage_tokens else cls_token[:, :0]
            prefix = torch.cat([cls_token, storage], 1)
        self.register_buffer('dino_prefix_tokens', prefix.to(self.dtypes['dinov3']).cpu(), persistent=False)
        for block in self.dino_blocks:  # LinearKMaskedBias: fold the (0/1) key-bias mask into the bias
            qkv = block.attn.qkv
            if hasattr(qkv, 'bias_mask'):
                qkv.bias.data = qkv.bias.data * qkv.bias_mask.to(qkv.bias.dtype)
                del qkv.bias_mask
                qkv.__class__ = nn.Linear
        mv = net.matcher.mv_vit
        self.mv_projector, self.mv_blocks = mv.projector, mv.blocks
        self.mv_norm, self.mv_output_projector = mv.norm, mv.output_projector
        self.mv_dim, self.mv_out = mv.embed_dim, mv.output_projector.out_features
        self.dino_dim = dino.embed_dim
        self.head = net.matcher.head
        self.vgg = net.refiner_features
        self.refiners = net.refiners
        self.anchor = (upstream.anchor_width, upstream.anchor_height)

        fold_batchnorms(self.vgg)
        for refiner in self.refiners.values():
            for block in [refiner.block1, *refiner.hidden_blocks]:
                block.conv_depthwise = fuse_conv_bn_eval(block.conv_depthwise, block.norm)
                block.norm = nn.Identity()
                block.enable_amp = False  # the static model sets the dtypes itself

        autocast_module(self.patch_proj, self.dtypes['dinov3'], cast_all=True)
        autocast_module(self.dino_blocks, self.dtypes['dinov3'], cast_all=True)
        autocast_module(self.dino_norm, self.dtypes['dinov3'])
        for module in (self.mv_projector, self.mv_blocks, self.mv_norm, self.mv_output_projector):
            autocast_module(module, self.dtypes['matcher'])
        scratch = self.head.scratch
        for module in (self.head.norm, self.head.projects, self.head.resize_layers, scratch.layer1_rn,
                       scratch.layer2_rn, scratch.layer3_rn, scratch.layer4_rn, scratch.refinenet1, scratch.refinenet2,
                       scratch.refinenet3, scratch.refinenet4, scratch.output_conv1):
            autocast_module(module, self.dtypes['head'])  # output_conv2 runs in fp32 after the autocast region
        if self.dtypes['head'] == torch.bfloat16:  # TensorRT 10.9 has no bf16 ConvTranspose; the next conv casts back
            for module in self.head.resize_layers:
                if isinstance(module, nn.ConvTranspose2d):
                    module.float()
        autocast_module(self.vgg, self.dtypes['refiner'], cast_all=True)
        for refiner in self.refiners.values():
            autocast_module(refiner.block1, self.dtypes['refiner'])
            autocast_module(refiner.hidden_blocks, self.dtypes['refiner'])

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)  # romav2.normalizers.imagenet
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer('mean', mean, persistent=False)
        self.register_buffer('std', std, persistent=False)
        # per refiner scale: coordinate grid, the warp update scale and the local correlation window (in pixels of
        # that scale, an integer offset grid in normalized coordinates)
        for p, refiner in self.refiners.items():
            h, w = height // int(p), width // int(p)
            self.register_buffer(f'grid_{p}', get_normalized_grid(1, h, w).cpu(), persistent=False)
            self.register_buffer(f'warp_scale_{p}', refiner.cfg.refine_init * torch.tensor((w, h)), persistent=False)
            r = refiner.cfg.local_corr_radius
            if r is not None:
                ys, xs = torch.meshgrid(torch.linspace(-2 * r / h, 2 * r / h, 2 * r + 1, device=grid.device),
                                        torch.linspace(-2 * r / w, 2 * r / w, 2 * r + 1, device=grid.device),
                                        indexing='ij')
                self.register_buffer(f'window_{p}', torch.stack((xs, ys), 2).reshape(-1, 2).cpu(), persistent=False)
        self.register_buffer('scale_factor', torch.tensor((width / self.anchor[0], height / self.anchor[1])),
                             persistent=False)

    @staticmethod
    def _rope(t, sin, cos, prefix, d):
        """SelfAttention.apply_rope for q or k [B, heads, N, d]: RoPE (rotate-half form) on the tokens after the prefix,
        computed in the dtype of the table."""
        dtype = t.dtype
        t = cast(t, sin.dtype)
        rest = t[:, :, prefix:]
        rotated = torch.cat([-rest[:, :, :, d // 2:], rest[:, :, :, :d // 2]], 3)
        rest = rest * cos + rotated * sin
        t = torch.cat([t[:, :, :prefix], rest], 2) if prefix else rest
        return cast(t, dtype)

    def _block(self, block, x, b, n, rope=None, prefix=0):
        """SelfAttentionBlock (DINOv3 and RoMa v2's ViT, eval branch) on x [b, n, C]."""
        attn = block.attn
        c, heads = attn.qkv.in_features, attn.num_heads
        qkv = attn.qkv(block.norm1(x)).reshape(b, n, 3, heads, c // heads)
        q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))
        if rope is not None:
            q, k = self._rope(q, *rope, prefix, c // heads), self._rope(k, *rope, prefix, c // heads)
        y = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, c)
        x = x + block.ls1(attn.proj(y))
        return x + block.ls2(block.mlp(block.norm2(x)))

    def dinov3(self, images):
        """Descriptor: DINOv3 get_intermediate_layers (normed) of both images -> list of [2, h*w, 1024] fp32."""
        if not self.batch_images:
            per_image = [self._dinov3(images[:1], 1), self._dinov3(images[1:], 1)]
            return [torch.cat(layer, 0) for layer in zip(*per_image)]
        return self._dinov3(images, 2)

    def _dinov3(self, images, b):
        hw = self.h * self.w
        x = self.patch_proj((images - self.mean) / self.std).flatten(2).transpose(1, 2)
        prefix = self.dino_prefix_tokens
        x = torch.cat([prefix.expand(b, self.dino_prefix, self.dino_dim), x], 1)
        outputs = []
        for i, block in enumerate(self.dino_blocks):
            x = self._block(block, x, b, self.dino_prefix + hw, (self.dino_sin, self.dino_cos), self.dino_prefix)
            if i in self.dino_layers:
                outputs.append(self.dino_norm(x)[:, self.dino_prefix:])
        return outputs

    def multiview(self, features):
        """Matcher.mv_vit on both images' concatenated DINOv3 layers [2, h*w, 2048] -> [2, h*w, 1024] fp32. Even blocks
        attend across both images without RoPE, odd blocks within each image with RoPE."""
        hw = self.h * self.w
        x = self.mv_projector(features.reshape(1, 2 * hw, 2 * self.dino_dim))
        for i, block in enumerate(self.mv_blocks):
            if i % 2:
                x = self._block(block, x.reshape(2, hw, self.mv_dim), 2, hw, (self.mv_sin, self.mv_cos))
                x = x.reshape(1, 2 * hw, self.mv_dim)
            else:
                x = self._block(block, x, 1, 2 * hw)
        return cast(self.mv_output_projector(self.mv_norm(x)), torch.float32).reshape(2, hw, self.mv_out)

    def _fusion(self, block, x, skip, size):
        """FeatureFusionBlock with the interpolation in fp32 (TensorRT has no bf16 Resize)."""
        if block.has_residual:
            x = block.skip_add.add(x, block.resConfUnit1(skip))
        x = block.resConfUnit2(x)
        return block.out_conv(interpolate(x, size, align_corners=block.align_corners))

    def dpt(self, tokens):
        """DPTHead on two token maps [D, h*w, 1024] (the shallower layer, the deeper one plus the matcher features) ->
        warp and overlap logit [D, H/4, W/4, 3]."""
        head, scratch = self.head, self.head.scratch
        h, w = self.h, self.w
        layers = []
        for i, x in enumerate([tokens[0], tokens[0], tokens[1], tokens[1]]):
            x = head.norm(x).permute(0, 2, 1).reshape(self.directions, self.dino_dim, h, w)
            layers.append(head.resize_layers[i](head.projects[i](x)))
        l1, l2, l3, l4 = (rn(x) for rn, x in zip(
            (scratch.layer1_rn, scratch.layer2_rn, scratch.layer3_rn, scratch.layer4_rn), layers))
        out = self._fusion(scratch.refinenet4, l4, None, (h, w))
        out = self._fusion(scratch.refinenet3, out, l3, (2 * h, 2 * w))
        out = self._fusion(scratch.refinenet2, out, l2, (4 * h, 4 * w))
        out = self._fusion(scratch.refinenet1, out, l1, (8 * h, 8 * w))
        out = scratch.output_conv1(out)
        out = interpolate(out, (PATCH * h // head.down_ratio, PATCH * w // head.down_ratio),
                          align_corners=head.align_corners)
        return scratch.output_conv2(cast(out, torch.float32)).permute(0, 2, 3, 1)

    def vgg_features(self, images):
        """FineFeatures (VGG19-BN): the maps before each max-pool, {1: [2, 64, H, W], 2: ..., 4: ...} (NCHW)."""
        x = cast((images - self.mean) / self.std, self.dtypes['refiner'])
        feats, scale = {}, 1
        for layer in self.vgg.layers:
            if isinstance(layer, nn.MaxPool2d):
                feats[scale] = x
                scale *= 2
            x = layer(x)
        return feats

    def _local_correlation(self, f_a, f_b, warp, p):
        """native_torch_local_corr: correlation of f_a [D, c, h, w] with f_b sampled at the warp plus the window offsets
        -> [D, K, h, w]."""
        window = getattr(self, f'window_{p}')
        d, c, h, w = self.directions, self.refiners[p].cfg.proj_dim, self.height // int(p), self.width // int(p)
        k = (2 * self.refiners[p].cfg.local_corr_radius + 1) ** 2
        coords = (warp[:, :, :, None] + window[None, None, None]).reshape(d, h, w * k, 2)
        sampled = F.grid_sample(f_b, coords, mode='bilinear', padding_mode='zeros', align_corners=False)
        corr = (f_a[:, :, :, :, None] / c ** .5 * sampled.reshape(d, c, h, w, k)).sum(1)
        return corr.permute(0, 3, 1, 2)

    def _refine(self, p, f_a, f_b, warp, confidence):
        """ConvRefiner.forward at scale p: f_a, f_b [D, C, h, w], warp [D, h, w, 2], confidence [D, h, w, 1 or 4]."""
        refiner = self.refiners[p]
        d, c, h, w = self.directions, refiner.cfg.feat_dim, self.height // int(p), self.width // int(p)
        proj = refiner.cfg.proj_dim
        f_a = refiner.proj(cast(f_a, torch.float32).permute(0, 2, 3, 1).reshape(d, h * w, c)).reshape(d, h, w, proj)
        f_b = refiner.proj(cast(f_b, torch.float32).permute(0, 2, 3, 1).reshape(d, h * w, c)).reshape(d, h, w, proj)
        f_a, f_b = f_a.permute(0, 3, 1, 2), f_b.permute(0, 3, 1, 2)
        f_ba = F.grid_sample(f_b, warp, mode=refiner.cfg.grid_sample_mode, align_corners=False)
        displacement = (warp - getattr(self, f'grid_{p}')).permute(0, 3, 1, 2)
        x = [f_a, f_ba, refiner.disp_emb(self.scale_factor[None, :, None, None] * displacement)]
        if refiner.cfg.local_corr_radius is not None:
            x.append(self._local_correlation(f_a, f_b, warp, p))
        z = cast(refiner.hidden_blocks(refiner.block1(torch.cat(x, 1))), torch.float32)
        warp = warp + refiner.warp_head(z).permute(0, 2, 3, 1) / getattr(self, f'warp_scale_{p}')
        delta = refiner.confidence_head(z).permute(0, 2, 3, 1)
        # Cholesky parametrization of the 2x2 precision matrix (in pixels)
        l00 = F.softplus(delta[:, :, :, 1]) + 1e-6
        l10 = delta[:, :, :, 2]
        l11 = F.softplus(delta[:, :, :, 3]) + 1e-6
        delta = torch.stack([delta[:, :, :, 0], l00 * l00, l00 * l10, l10 * l10 + l11 * l11], 3)
        if confidence.shape[3] != delta.shape[3]:
            confidence = torch.cat([confidence, torch.zeros_like(delta[:, :, :, 1:])], 3)
        return warp, confidence + delta

    def forward(self, image0, image1):
        """
        Args:
            image0, image1: [1, 3, H, W] RGB images in [0, 1]
        Returns:
            warp_AB: [1, H, W, 2] normalized [-1, 1] coordinates in image1 of every pixel of image0
            confidence_AB: [1, H, W, 4] overlap logit and the precision matrix parameters (p00, p10, p11)
            warp_BA, confidence_BA: the same for image1 into image0, if bidirectional
        """
        images = torch.cat([image0, image1], 0)
        hw = self.h * self.w
        layers = self.dinov3(images)  # [2, h*w, 1024] each
        f_mv = self.multiview(torch.cat(layers, 2))

        # global matching in fp32: softmax over image1's patches of the cosine similarity, applied to the positional
        # embedding of image1's grid (and the other way around)
        f = f_mv / f_mv.norm(dim=2, keepdim=True)
        sim = self.inv_temp * torch.matmul(f[0], f[1].t())
        sims = torch.stack([sim, sim.t()]) if self.bidirectional else sim[None]
        match_emb = torch.matmul(F.softmax(sims, 2), self.pos_emb_grid)

        # DPT head: direction AB reads image0's features, BA image1's
        d = self.directions
        tokens = [layers[0][:d], layers[1][:d] + f_mv[:d] + match_emb]
        out = self.dpt(tokens)
        warp, confidence = out[:, :, :, :2], out[:, :, :, 2:]

        # refiners at 1/4, 1/2 and 1/1 of the input
        feats = self.vgg_features(images)
        for p in self.refiners:
            scale = int(p)
            size = (self.height // scale, self.width // scale)
            warp = interpolate(warp.permute(0, 3, 1, 2), size).permute(0, 2, 3, 1)
            confidence = interpolate(confidence.permute(0, 3, 1, 2), size).permute(0, 2, 3, 1)
            f = feats[scale]
            f_a, f_b = (f, torch.cat([f[1:], f[:1]])) if self.bidirectional else (f[:1], f[1:])
            warp, confidence = self._refine(p, f_a, f_b, warp, confidence)
        if self.bidirectional:
            return warp[:1], confidence[:1], warp[1:], confidence[1:]
        return warp, confidence
