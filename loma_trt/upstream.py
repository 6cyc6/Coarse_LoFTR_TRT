"""Loader for the pinned upstream LoMa source (third_party/LoMa, kept unmodified)."""
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_DIR = ROOT / 'third_party' / 'LoMa'
WEIGHTS_DIR = ROOT / 'weights' / 'loma'
# variant name -> upstream config class; B128: DeDoDe-B descriptor (VGG19, 128-d), B: DeDoDe-G (VGG19 + DINOv2 ViT-L/14)
VARIANTS = {'b128': 'LoMaB128', 'b': 'LoMaB'}


def import_upstream():
    """Import upstream `loma.loma`; its package lives in third_party/LoMa/src, which is only put on sys.path here."""
    src = UPSTREAM_DIR / 'src'
    if not (src / 'loma').is_dir():
        raise FileNotFoundError(f'{UPSTREAM_DIR} is empty, run: pixi run init-submodules')
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    import loma.loma
    return loma.loma


def upstream_commit():
    try:
        return subprocess.check_output(['git', '-C', str(UPSTREAM_DIR), 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def make_config(variant='b128', mp=True):
    if variant not in VARIANTS:
        raise ValueError(f'Unknown LoMa variant {variant!r}, choose from {list(VARIANTS)}')
    return getattr(import_upstream(), VARIANTS[variant])(mp=mp)


def checkpoint_path(variant='b128'):
    """Local copy of the released weights; upstream downloads them to the torch hub cache on first use."""
    return Path(torch.hub.get_dir()) / 'checkpoints' / make_config(variant).weights_url.rsplit('/', 1)[1]


def set_fp32(model):
    """Turn off upstream's mixed precision: the autocast of the detector and descriptor CNNs (their `amp` switch) and
    the cast of DINOv2 to the autocast dtype. DINOv2 keeps the values of its half-precision weights."""
    for module in model.modules():
        if isinstance(getattr(module, 'amp', None), bool):
            module.amp = False
    dinov2 = getattr(model._descriptor.encoder, 'frozen_dinov2', None)
    if dinov2 is not None:
        dinov2.dinov2_vitl14.float()
        dinov2.amp_dtype = torch.float32
    return model


def load_upstream(variant='b128', fp32=False):
    """Upstream LoMa with its released weights, on upstream's device (CUDA).

    Upstream runs the detector, descriptor and matcher under bf16 autocast (fp16 before Ampere) and casts DINOv2 to
    that dtype; fp32=True turns all of it off for an all-fp32 reference.
    """
    model = import_upstream().LoMa(make_config(variant, mp=not fp32))
    return set_fp32(model).eval() if fp32 else model.eval()


@torch.inference_mode()
def run_upstream(model, image0, image1, num_keypoints=None, threshold=None):
    """Upstream inference on [1, 3, H, W] RGB images in [0, 1] (its tensor API: detection and description at the
    given size). Returns the keypoints of both images (normalized [-1, 1] coordinates), the match of every keypoint of
    image0 (-1 if none) and its score."""
    loma = import_upstream()
    kpts0, desc0, _, _ = model.detect_and_describe(image0, num_keypoints)
    kpts1, desc1, _, _ = model.detect_and_describe(image1, num_keypoints)
    scores = model(kpts0, kpts1, desc0, desc1)['scores']
    m0, _, mscores0, _ = loma.filter_matches(scores, model.cfg.filter_threshold if threshold is None else threshold)
    return {'keypoints0': kpts0[0], 'keypoints1': kpts1[0], 'matches0': m0[0], 'scores0': mscores0[0]}


def to_pixels(normalized, height, width):
    """Normalized [-1, 1] coordinates to OpenCV pixels (pixel centres at integers): upstream to_pixel_coords - 0.5."""
    scale = torch.tensor([width / 2, height / 2], device=normalized.device)
    return (normalized + 1) * scale - 0.5
