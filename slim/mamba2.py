"""mamba_ssm.Mamba2 forward without its Triton/CUDA kernels, for ONNX/TensorRT export.

SLiM runs Mamba2 only on the 4x-downsampled 1/8 feature maps (4 x 300 tokens at 640x480), so the state space
duality (SSD) form is computed over the whole sequence at once, without chunking: per head, y = (L o C B^T) x with
the decay matrix L[t, s] = exp(sum_{s<k<=t} dt_k A) for s <= t, exactly what mamba_split_conv1d_scan_combined
computes. The projections run in the dtype of the module weights; the scan, gating and RMSNorm in fp32.
"""
import torch
import torch.nn.functional as F


def segsum_decay(x):
    """exp of the segment sums: out[..., t, s] = exp(sum_{s<k<=t} x[..., k]) for s <= t, else 0.

    Computed from one cumulative sum as exp(cs[t] - cs[s]) instead of upstream's quadratic masked cumsum
    (ssd_minimal.segsum); x = dt * A <= 0, and over SLiM's 1200 tokens the fp32 differences stay accurate to ~1e-5.
    """
    cs = torch.cumsum(x, dim=-1)
    T = x.shape[-1]
    k = torch.arange(T, device=x.device)
    causal = k[:, None] >= k[None, :]
    return torch.exp((cs[..., :, None] - cs[..., None, :]).masked_fill(~causal, float('-inf')))


def mamba2_forward(m, u):
    """Mamba2 module `m` (ngroups 1, no D per channel, rmsnorm, norm after gate) on u [B, L, d_model]."""
    if m.ngroups != 1 or m.D_has_hdim or not m.rmsnorm or m.norm_before_gate or m.d_ssm != m.d_inner:
        raise ValueError('Only the Mamba2 configuration of SLiM is supported')
    if m.dt_limit != (0.0, float('inf')):
        raise ValueError('dt_limit is not supported')
    batch, length, _ = u.shape
    H, P, N = m.nheads, m.headdim, m.d_state
    zxbcdt = m.in_proj(u.to(m.in_proj.weight.dtype))
    z, xBC, dt = torch.split(zxbcdt, [m.d_inner, m.d_inner + 2 * N, H], dim=-1)

    # causal depthwise conv1d with SiLU (causal_conv1d_fn)
    xBC = m.conv1d(xBC.transpose(1, 2))[..., :length].transpose(1, 2)
    xBC = F.silu(xBC.float())
    x, B, C = torch.split(xBC, [m.d_inner, N, N], dim=-1)

    dt = F.softplus(dt.float() + m.dt_bias.float())  # [b, l, h]
    A = -torch.exp(m.A_log.float())  # [h]
    decay = segsum_decay((dt * A).transpose(1, 2))  # [b, h, l, s]
    scores = torch.matmul(C, B.transpose(1, 2))  # [b, l, s], shared by the heads (one group)
    x = x.view(batch, length, H, P)
    xdt = (x * dt[..., None]).permute(0, 2, 1, 3)  # [b, h, s, p]
    y = torch.matmul(decay * scores[:, None], xdt).permute(0, 2, 1, 3)  # [b, l, h, p]
    y = (y + x * m.D.float()[:, None]).reshape(batch, length, m.d_inner)

    # RMSNormGated with norm_before_gate=False: rmsnorm(y * silu(z))
    y = y * F.silu(z.float())
    y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + m.norm.eps) * m.norm.weight.float()
    return m.out_proj(y.to(m.out_proj.weight.dtype))
