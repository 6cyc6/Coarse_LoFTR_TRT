"""Loader for the pinned upstream EfficientLoFTR source (third_party/EfficientLoFTR, kept unmodified)."""
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_DIR = ROOT / 'third_party' / 'EfficientLoFTR'
WEIGHTS_DIR = ROOT / 'weights' / 'eloftr'
DEFAULT_CKPT = WEIGHTS_DIR / 'eloftr_outdoor.ckpt'
SAMPLE_IMAGES_DIR = UPSTREAM_DIR / 'assets' / 'phototourism_sample_images'


def import_upstream():
    """Import upstream `src.loftr`; upstream names its package `src`, so it is only put on sys.path here."""
    if not (UPSTREAM_DIR / 'src' / 'loftr').is_dir():
        raise FileNotFoundError(f'{UPSTREAM_DIR} is empty, run: pixi run init-submodules')
    if str(UPSTREAM_DIR) not in sys.path:
        sys.path.insert(0, str(UPSTREAM_DIR))
    import src.loftr
    return src.loftr


def upstream_commit():
    try:
        return subprocess.check_output(['git', '-C', str(UPSTREAM_DIR), 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def make_config(model_type='full', npe=None, mp=False, thr=None, border_rm=None):
    """Upstream config; thr and border_rm override the coarse matching threshold and border removal."""
    loftr = import_upstream()
    if model_type == 'full':
        config = deepcopy(loftr.full_default_cfg)
    elif model_type == 'opt':
        config = deepcopy(loftr.opt_default_cfg)
    else:
        raise ValueError(f'Unknown model type {model_type!r}, choose full or opt')
    if npe is not None:
        config['coarse']['npe'] = list(npe)
    if thr is not None:
        config['match_coarse']['thr'] = thr
    if border_rm is not None:
        config['match_coarse']['border_rm'] = border_rm
    config['mp'] = mp
    return config


def load_upstream(ckpt=DEFAULT_CKPT, model_type='full', npe=None, mp=False, device='cuda', thr=None, border_rm=None):
    """Upstream LoFTR in eval mode with RepVGG re-parameterization, as in the upstream README."""
    ckpt = Path(ckpt)
    if not ckpt.exists():
        raise FileNotFoundError(f'{ckpt} not found, run: pixi run download-weights')
    loftr = import_upstream()
    model = loftr.LoFTR(config=make_config(model_type, npe, mp, thr, border_rm))
    model.load_state_dict(torch.load(ckpt, map_location='cpu')['state_dict'])
    model = loftr.reparameter(model)
    return model.eval().to(device)


@torch.no_grad()
def run_upstream(model, image0, image1, mp=False, strict_fp32=False):
    """Run upstream inference and return its batch dict (mkpts0_f, mkpts1_f, mconf, i_ids, j_ids, ...).

    By default upstream computes the fine-level einsums under fp16 autocast even for an fp32 model;
    strict_fp32 turns that off (via the upstream `validate` switch) to get an all-fp32 reference.
    """
    model.fine_matching.validate = strict_fp32
    batch = {'image0': image0, 'image1': image1}
    with torch.autocast(device_type='cuda', enabled=mp):
        model(batch)
    model.fine_matching.validate = False
    return batch
