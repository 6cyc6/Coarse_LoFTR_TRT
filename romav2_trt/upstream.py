"""Loader for the pinned upstream RoMa v2 source (third_party/RoMaV2, kept unmodified).

Upstream fetches the DINOv3 code with torch.hub at a pinned commit on first use (network access) and downloads its
weights (romav2.0.1.pt, 1.1 GB, which include DINOv3) to the torch hub cache.
"""
import contextlib
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_DIR = ROOT / 'third_party' / 'RoMaV2'
WEIGHTS_DIR = ROOT / 'weights' / 'romav2'
CHECKPOINT_NAME = 'romav2.0.1.pt'


def import_upstream():
    """Import upstream `romav2.romav2`; its package lives in third_party/RoMaV2/src, only put on sys.path here."""
    src = UPSTREAM_DIR / 'src'
    if not (src / 'romav2').is_dir():
        raise FileNotFoundError(f'{UPSTREAM_DIR} is empty, run: pixi run init-submodules')
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    import romav2.romav2
    return romav2.romav2


def upstream_commit():
    try:
        return subprocess.check_output(['git', '-C', str(UPSTREAM_DIR), 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def checkpoint_path():
    return Path(torch.hub.get_dir()) / 'checkpoints' / CHECKPOINT_NAME


@contextlib.contextmanager
def autocast_disabled():
    """Turn every torch.autocast region off; upstream's DPT head and VGG features hardcode bf16 autocast."""
    original = torch.autocast

    class Disabled(original):
        def __init__(self, device_type, dtype=None, enabled=True, cache_enabled=None):
            super().__init__(device_type, dtype=dtype, enabled=False, cache_enabled=cache_enabled)

    torch.autocast = Disabled
    try:
        yield
    finally:
        torch.autocast = original


def load_upstream(fp32=False, bidirectional=False):
    """Upstream RoMa v2 with its released weights, on upstream's device (CUDA).

    Upstream runs DINOv3 (cast to bf16), the matcher, the DPT head and the refiner convolutions under bf16 autocast;
    fp32=True builds it without that (DINOv3 keeps the values of its bf16 weights) for an all-fp32 reference, which
    run_upstream then computes with autocast forced off. Only the single-resolution `forward(img_A, img_B)` is used.
    """
    romav2 = import_upstream()
    from romav2.features import Descriptor
    from romav2.matcher import Matcher

    if fp32:
        cfg = romav2.RoMaV2.Cfg(descriptor=Descriptor.Cfg(enable_amp=False), matcher=Matcher.Cfg(enable_amp=False))
    else:
        cfg = romav2.RoMaV2.Cfg()
    model = romav2.RoMaV2(cfg)
    if fp32:
        for module in model.refiners.modules():
            if isinstance(getattr(module, 'enable_amp', None), bool):
                module.enable_amp = False
    model.bidirectional = bidirectional
    model.fp32_reference = fp32
    return model.eval()


@torch.inference_mode()
def run_upstream(model, image0, image1):
    """Upstream forward on [1, 3, H, W] RGB images in [0, 1] (H, W multiples of 16): dict with warp_AB [1, H, W, 2]
    (normalized coordinates in image1 of every pixel of image0) and confidence_AB [1, H, W, 4] (overlap logit and
    precision parameters), and warp_BA, confidence_BA if the model is bidirectional (else None)."""
    with autocast_disabled() if model.fp32_reference else contextlib.nullcontext():
        preds = model(image0, image1)
    return {k: preds[k] for k in ('warp_AB', 'confidence_AB', 'warp_BA', 'confidence_BA')}
