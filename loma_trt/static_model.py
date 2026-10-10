"""Fixed-shape LoMa for ONNX/TensorRT export.

Upstream detects a fixed number of keypoints per image (DaD: dense scoremap, NMS, top-k, sub-pixel refinement),
describes them (DeDoDe), matches them with a LightGlue-style transformer and keeps the mutual nearest neighbours of the
dual-softmax scores above a threshold. Only that last step has a data-dependent size; here every keypoint of image0
keeps its candidate match and a `valid` mask marks the matches upstream returns. The CNNs, DINOv2 and the transformer
layers are the upstream modules and weights; the glue between them is re-implemented following upstream step by step.
"""
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval

from loma_trt.upstream import import_upstream

# detector: DaD (VGG11 + conv decoder); descriptor: DeDoDe (VGG19 + conv decoder, and DINOv2 for LoMa-B); matcher:
# input projection, positional encoding, transformer and match assignment. These are the parts upstream autocasts.
GROUPS = ('detector', 'descriptor', 'matcher')
# Upstream mixed precision uses bf16 from Ampere on (fp16 before). fp16 is the default here: TensorRT runs it faster,
# and with 3 more mantissa bits it keeps ~96% of the fp32 keypoints where bf16 keeps ~80% (all activations, DINOv2's
# included, stay far below the fp16 range). Scoremap logits, softmaxes, NMS, top-k, sampling and matching stay fp32.
PRECISIONS = {'fp32': None, 'bf16': torch.bfloat16, 'fp16': torch.float16}
MAX_KEYPOINTS = 3840  # TensorRT TopK limit


def cast(x, dtype):
    """x in dtype. Only casts when needed: the ONNX exporter mistypes the values of a no-op cast whose input is used
    again (wrong ranks in later reshapes and concats)."""
    return x if x.dtype == dtype else x.to(dtype)


class CastLinear(nn.Linear):
    """Linear that casts its input to its weight dtype, as autocast does."""

    def forward(self, x):
        return F.linear(cast(x, self.weight.dtype), self.weight, self.bias)


class CastConv2d(nn.Conv2d):
    """Conv2d that casts its input to its weight dtype, as autocast does."""

    def forward(self, x):
        return super().forward(cast(x, self.weight.dtype))


class CastConvTranspose2d(nn.ConvTranspose2d):
    """ConvTranspose2d that casts its input to its weight dtype, as autocast does."""

    def forward(self, x, output_size=None):
        return super().forward(cast(x, self.weight.dtype), output_size)


class Fp32OutLayerNorm(nn.LayerNorm):
    """LayerNorm computed and returned in fp32, as autocast runs it."""

    def forward(self, x):
        return F.layer_norm(cast(x, torch.float32), self.normalized_shape, self.weight, self.bias, self.eps)


class HalfLayerNorm(nn.LayerNorm):
    """LayerNorm that normalizes in fp32 and returns the input dtype, as a half-precision LayerNorm kernel does."""

    def forward(self, x):
        y = F.layer_norm(cast(x, torch.float32), self.normalized_shape, self.weight, self.bias, self.eps)
        return cast(y, x.dtype)


def half_module(module, dtype):
    """Cast parameters and floating buffers to dtype; LayerNorms keep fp32 weights and normalize in fp32."""
    for m in module.modules():
        if isinstance(m, nn.LayerNorm):
            m.__class__ = HalfLayerNorm
            continue
        for name, param in m.named_parameters(recurse=False):
            param.data = param.data.to(dtype)
        for name, buffer in m.named_buffers(recurse=False):
            if buffer.is_floating_point():
                m._buffers[name] = buffer.to(dtype)


def interpolate(x, size, mode='bilinear', align_corners=False):
    """F.interpolate computed in fp32 and returned in the input dtype: TensorRT 10.9 has no bf16 Resize, and PyTorch's
    half-precision kernels also interpolate in fp32 and round the result once."""
    return cast(F.interpolate(cast(x, torch.float32), size=size, mode=mode, align_corners=align_corners), x.dtype)


def autocast_module(module, dtype, cast_all=False):
    """Emulate autocast: Linears and convolutions take their input in dtype, LayerNorms run in fp32 and return fp32.
    cast_all also casts every other parameter and buffer (upstream casts its frozen DINO backbones to the autocast
    dtype); LayerNorm weights stay fp32 either way."""
    casts = {nn.Linear: CastLinear, nn.Conv2d: CastConv2d, nn.ConvTranspose2d: CastConvTranspose2d}
    for m in module.modules():
        if isinstance(m, nn.LayerNorm):
            m.__class__ = Fp32OutLayerNorm
            continue
        if type(m) in casts:
            m.__class__ = casts[type(m)]
        if cast_all or isinstance(m, tuple(casts.values())):
            for param in m.parameters(recurse=False):
                param.data = param.data.to(dtype)
            for name, buffer in m.named_buffers(recurse=False):
                if buffer.is_floating_point():
                    m._buffers[name] = buffer.to(dtype)


def fold_batchnorms(module):
    """Fold every BatchNorm2d that directly follows a Conv2d in a Sequential or ModuleList into the conv, in fp32 (one
    rounding when the conv is cast to half precision afterwards)."""
    containers = [m for m in module.modules() if isinstance(m, (nn.Sequential, nn.ModuleList))]
    for seq in containers:
        for i in range(len(seq) - 1):
            if isinstance(seq[i], nn.Conv2d) and isinstance(seq[i + 1], nn.BatchNorm2d):
                seq[i] = fuse_conv_bn_eval(seq[i], seq[i + 1])
                seq[i + 1] = nn.Identity()


def half_groups(precision, groups=None, all_groups=GROUPS):
    """Module groups that run in the half dtype of `precision`: all of them, or an explicit comma-separated subset."""
    if precision not in PRECISIONS:
        raise ValueError(f'Unknown precision {precision!r}, choose from {list(PRECISIONS)}')
    if PRECISIONS[precision] is None:
        return ()
    if groups is None:
        return all_groups
    selected = tuple(g for g in groups.split(',') if g)
    unknown = set(selected) - set(all_groups)
    if unknown:
        raise ValueError(f'Unknown groups {sorted(unknown)}, choose from {all_groups}')
    return tuple(g for g in all_groups if g in selected)


def precision_label(precision, groups, all_groups=GROUPS):
    if not groups or tuple(groups) == all_groups:
        return precision if groups else 'fp32'
    return f'{precision}-' + '-'.join(groups)


class StaticLoMa(nn.Module):
    def __init__(self, upstream, height, width, num_keypoints=None, precision='fp32', groups=None):
        """
        Args:
            upstream: upstream LoMa (loma_trt.upstream.load_upstream, preferably fp32=True); it is copied, not modified
            height, width: fixed input size, multiples of 8 (of 56 with the DINOv2 descriptor of LoMa-B)
            num_keypoints: keypoints per image, default upstream's 2048
            precision: fp32, bf16 or fp16 (the half dtype of the groups that upstream autocasts)
            groups: module groups in half precision, default all of GROUPS
        """
        super().__init__()
        cfg = upstream.cfg
        self.dinov2 = cfg.descriptor == 'dedode_g'
        multiple = 56 if self.dinov2 else 8
        if height % multiple or width % multiple:
            raise ValueError(f'Input size must be a multiple of {multiple} for {cfg.name}, got {height}x{width}')
        self.num_keypoints = num_keypoints or cfg.num_keypoints
        if not 0 < self.num_keypoints <= min(MAX_KEYPOINTS, height * width):
            raise ValueError(f'num_keypoints must be in 1..{MAX_KEYPOINTS}, got {self.num_keypoints}')
        detector_cfg = upstream._detector.cfg
        if (detector_cfg.remove_borders or detector_cfg.increase_coverage or detector_cfg.coverage_from_sparse
                or not detector_cfg.subpixel or detector_cfg.nms_size != 3):
            raise ValueError('Only the released DaD settings (3x3 NMS, top-k, sub-pixel, no coverage) are supported')

        self.height, self.width = height, width
        self.threshold = cfg.filter_threshold
        self.subpixel_temp = detector_cfg.subpixel_temp
        self.half_groups = half_groups(precision, groups)
        half = PRECISIONS[precision]
        self.dtypes = {g: half if g in self.half_groups else torch.float32 for g in GROUPS}

        net = deepcopy(upstream).float()
        for module in net.modules():
            if isinstance(getattr(module, 'amp', None), bool):
                module.amp = False  # the static model sets the dtypes itself
        self.detector = net._detector
        self.descriptor = net._descriptor
        self.input_proj, self.posenc, self.transformers = net.input_proj, net.posenc, net.transformers
        self.assignment = net.log_assignment[cfg.n_layers - 1]  # upstream scores with the last layer's assignment
        fold_batchnorms(self.detector)
        fold_batchnorms(self.descriptor)

        if self.dinov2:
            dino = self.descriptor.encoder.frozen_dinov2.dinov2_vitl14
            n = (height // 14) * (width // 14)
            # upstream interpolates the positional embedding (bicubic, fp32) on every call; it only depends on the size
            with torch.no_grad():
                pos = dino.interpolate_pos_encoding(dino.pos_embed.new_zeros(1, n + 1, dino.embed_dim), height, width)
            self.register_buffer('dino_pos_embed', pos.to(self.dtypes['descriptor']), persistent=False)
        half_module(self.detector, self.dtypes['detector'])
        half_module(self.descriptor, self.dtypes['descriptor'])
        for module in (self.input_proj, self.posenc, self.transformers, self.assignment):
            autocast_module(module, self.dtypes['matcher'])

        mean, std = self.detector.normalizer.mean, self.detector.normalizer.std
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1), persistent=False)
        # normalized pixel-centre grid and the 3x3 sub-pixel offsets, made by upstream's own helper
        import_upstream()
        from loma.geometry import get_normalized_grid
        grid = get_normalized_grid(1, height, width).reshape(height * width, 2)
        offsets = get_normalized_grid(1, 3, 3).reshape(9, 2) * torch.tensor([3 / width, 3 / height], device=grid.device)
        self.register_buffer('grid', grid.cpu(), persistent=False)
        self.register_buffer('subpixel_offsets', offsets.cpu(), persistent=False)
        self.register_buffer('pixel_scale', torch.tensor([width / 2, height / 2]), persistent=False)
        self.register_buffer('keypoint_ids', torch.arange(self.num_keypoints), persistent=False)

    def _scales(self, n):
        """Sizes of the 1, 1/2, ..., 1/2^(n-1) feature maps (the VGG maps before each max-pool)."""
        return [(self.height >> i, self.width >> i) for i in range(n)]

    def detect(self, images):
        """DaD.forward + sample_keypoints: [2, 3, H, W] -> [2, K, 2] normalized keypoints, highest probability first."""
        dtype = self.dtypes['detector']
        encoder, decoder = self.detector.encoder, self.detector.decoder
        feats, _ = encoder(cast((images - self.mean) / self.std, dtype))
        sizes = self._scales(len(feats))
        logits = context = None
        for idx, (feature_map, scale) in enumerate(zip(reversed(feats), ('8', '4', '2', '1'))):
            x = feature_map if context is None else torch.cat((feature_map, cast(context, dtype)), 1)
            out = decoder.layers[scale](x)
            delta = cast(out[:, :decoder.num_prototypes], torch.float32)
            context = out[:, decoder.num_prototypes:]
            logits = delta if logits is None else logits + delta
            if idx < len(sizes) - 1:
                size = sizes[-(idx + 2)]
                logits = F.interpolate(logits, size=size, mode='bicubic', align_corners=False)
                context = F.interpolate(cast(context, torch.float32), size=size, mode='bilinear', align_corners=False)

        # dense probabilities, 3x3 non-maximum suppression and the top-k (sample_keypoints with sample_topk)
        h, w = self.height, self.width
        probs = F.softmax(logits.reshape(2, h * w), dim=1).reshape(2, 1, h, w)
        probs = probs * (probs == F.max_pool2d(probs, 3, stride=1, padding=1))
        inds = torch.topk(probs.reshape(2, h * w), k=self.num_keypoints).indices
        keypoints = self.grid[inds]
        # sub-pixel refinement: softmax of the zero-padded 3x3 logit patch (nn.Unfold order) weights the 3x3 offsets
        padded = F.pad(logits[:, 0], (1, 1, 1, 1))
        shifted = torch.stack([padded[:, dy:dy + h, dx:dx + w] for dy in range(3) for dx in range(3)], 1)
        patches = torch.gather(shifted.reshape(2, 9, h * w), 2, inds[:, None].expand(2, 9, self.num_keypoints))
        weights = F.softmax(patches / self.subpixel_temp, dim=1)
        return keypoints + torch.matmul(weights.transpose(1, 2), self.subpixel_offsets)

    def _dinov2(self, images):
        """FrozenDINOv2.forward with the positional embedding of this input size: [2, 1024, H/14, W/14]."""
        dino = self.descriptor.encoder.frozen_dinov2.dinov2_vitl14
        x = dino.patch_embed(cast(images, self.dtypes['descriptor']))
        x = torch.cat((dino.cls_token.expand(2, 1, dino.embed_dim), x), 1) + self.dino_pos_embed
        for block in dino.blocks:
            x = block(x)
        x = dino.norm(x)[:, 1:]
        return x.transpose(1, 2).reshape(2, dino.embed_dim, self.height // 14, self.width // 14)

    def describe(self, images, keypoints):
        """DeDoDeDescriptor.describe_keypoints: dense descriptions sampled at the keypoints, [2, K, D] in fp32."""
        dtype = self.dtypes['descriptor']
        encoder, decoder = self.descriptor.encoder, self.descriptor.decoder
        if self.dinov2:
            feats, _ = encoder.vgg(cast(images, dtype))
            feats = feats + [self._dinov2(images)]
            sizes = self._scales(len(feats) - 1) + [(self.height // 14, self.width // 14)]
        else:
            feats, _ = encoder(cast(images, dtype))
            sizes = self._scales(len(feats))
        descriptions = context = None
        for idx, (feature_map, scale) in enumerate(zip(reversed(feats), decoder.scales)):
            x = feature_map if context is None else torch.cat((feature_map, context), 1)
            out = decoder.layers[scale](x)
            delta, context = out[:, :decoder.descriptor_dim], out[:, decoder.descriptor_dim:]
            descriptions = delta if descriptions is None else descriptions + delta
            if idx < len(sizes) - 1:
                size = sizes[-(idx + 2)]
                descriptions, context = interpolate(descriptions, size), interpolate(context, size)
        sampled = F.grid_sample(cast(descriptions, torch.float32), keypoints[:, None], mode='bilinear',
                                align_corners=False)
        return sampled[:, :, 0].transpose(1, 2)

    # Shapes are Python ints (traced sizes in arithmetic export as extra ops, and a size in a float power even as a
    # cast to complex), and dims are positive: TorchScript pools equal constants, and the ONNX export rewrites a pooled
    # -1 of a stack in place, which corrupts every other op using it (a cat(..., -1) became axis 4).
    def _rotary(self, t, encoding, heads, d):
        """apply_cached_rotary_emb: t [2, heads, K, d], encoding [2 (cos, sin), 2, 1, K, d]."""
        pairs = t.reshape(2, heads, self.num_keypoints, d // 2, 2)
        rotated = torch.stack((-pairs[..., 1], pairs[..., 0]), 4).reshape(2, heads, self.num_keypoints, d)
        return t * encoding[0] + rotated * encoding[1]

    def _self_block(self, block, x, encoding):
        """SelfBlock of both images at once, x [2, K, E]. Upstream reshapes with unflatten(-1, ...), which the ONNX
        exporter turns into a wrong reshape; the explicit reshapes here are the same computation."""
        n, e, heads = self.num_keypoints, block.embed_dim, block.num_heads
        qkv = block.Wqkv(x).reshape(2, n, heads, e // heads, 3).transpose(1, 2)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        q, k = (self._rotary(t, encoding, heads, e // heads) for t in (q, k))
        context = F.scaled_dot_product_attention(q, k, v)
        message = block.out_proj(context.transpose(1, 2).reshape(2, n, e))
        return x + block.ffn(torch.cat([x, message], 2))

    def _cross_block(self, block, x):
        """CrossBlock of both images at once, x [2, K, E]: each image attends to the other one."""
        n, e, heads = self.num_keypoints, block.to_qk.out_features, block.heads
        qk = block.to_qk(x).reshape(2, n, heads, e // heads).transpose(1, 2)
        v = block.to_v(x).reshape(2, n, heads, e // heads).transpose(1, 2)
        message = F.scaled_dot_product_attention(qk, torch.cat([qk[1:], qk[:1]]), torch.cat([v[1:], v[:1]]))
        message = block.to_out(message.transpose(1, 2).reshape(2, n, e))
        return x + block.ffn(torch.cat([x, message], 2))

    def forward(self, image0, image1):
        """
        Args:
            image0, image1: [1, 3, H, W] RGB images in [0, 1]
        Returns:
            keypoints0, keypoints1: [K, 2] (x, y) in OpenCV pixels of the input (pixel centres at integers); keypoints0
                are the keypoints of image0, keypoints1 the keypoint of image1 with the highest score for each of them
            confidence: [K] dual-softmax score of mutual nearest neighbours, 0 otherwise (upstream mscores0)
            valid: [K] bool, True for the matches upstream returns (mutual nearest neighbour, score above threshold)
        """
        images = torch.cat([image0, image1], 0)
        keypoints = self.detect(images)
        descriptions = self.describe(images, keypoints)

        # LoMa.forward: transformer layers on both keypoint sets, under (emulated) autocast
        desc = self.input_proj(descriptions)
        encoding = self.posenc(keypoints)  # [2 (cos, sin), 2 (images), 1, K, head_dim]
        for layer in self.transformers:
            desc = self._self_block(layer.self_attn, desc, encoding)
            desc = self._cross_block(layer.cross_attn, desc)

        # MatchAssignment (inference branch): dual softmax of the scaled similarity. The softmax over image0 is taken
        # on the transposed similarity so that both reductions run along the last axis; scores_t is exactly scores.T.
        mdesc = self.assignment.final_proj(desc)
        mdesc = mdesc / self.assignment.dim ** .25
        sim = cast(torch.matmul(mdesc[0], mdesc[1].t()), torch.float32)
        softmax1, softmax0 = F.softmax(sim, 1), F.softmax(sim.t(), 1)
        scores, scores_t = softmax1 * softmax0.t(), softmax0 * softmax1.t()

        # filter_matches: mutual nearest neighbours above the threshold
        max0, m0 = scores.max(dim=1)
        m1 = scores_t.argmax(dim=1)
        mutual0 = m1[m0] == self.keypoint_ids
        mscores0 = torch.where(mutual0, max0, torch.zeros_like(max0))
        valid = mutual0 & (mscores0 > self.threshold)
        pixels = (keypoints + 1) * self.pixel_scale - 0.5
        return pixels[0], pixels[1][m0], mscores0, valid
