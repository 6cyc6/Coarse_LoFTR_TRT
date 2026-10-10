"""Loader for the original LoFTR as packaged in kornia (kornia.feature.LoFTR) with its released weights."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from kornia.feature.loftr.loftr import LoFTR, default_cfg, urls

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = ROOT / 'weights' / 'loftr'
PRETRAINED = tuple(urls)  # outdoor (MegaDepth), indoor_new and indoor (ScanNet)


def checkpoint_path(pretrained):
    """Local copy of the released `pretrained` checkpoint, downloaded by load_upstream on first use."""
    return WEIGHTS_DIR / urls[pretrained].rsplit('/', 1)[1]


def make_config(pretrained='outdoor'):
    if pretrained not in urls:
        raise ValueError(f'Unknown pretrained weights {pretrained!r}, choose from {PRETRAINED}')
    config = deepcopy(default_cfg)
    # indoor_new was trained with the fixed positional encoding; kornia sets this on the shared default_cfg
    config['coarse']['temp_bug_fix'] = pretrained == 'indoor_new'
    return config


def load_upstream(pretrained='outdoor', ckpt=None, device='cuda'):
    """kornia LoFTR in eval mode with the released `pretrained` weights, or with a local LoFTR checkpoint `ckpt`
    (e.g. zju3dv's outdoor_ds.ckpt; `pretrained` then only selects the positional encoding)."""
    model = LoFTR(pretrained=None, config=make_config(pretrained))
    if ckpt is None:
        state = torch.hub.load_state_dict_from_url(urls[pretrained], model_dir=str(WEIGHTS_DIR), map_location='cpu')
    else:
        state = torch.load(ckpt, map_location='cpu', weights_only=False)  # Lightning checkpoints pickle objects
    model.load_state_dict(state['state_dict'])
    return model.eval().to(device)


@torch.no_grad()
def run_upstream(model, image0, image1):
    """kornia inference: dict with keypoints0, keypoints1, confidence and batch_indexes."""
    return model({'image0': image0, 'image1': image1})


def upstream_matches(out, width, stride=8):
    """Matches in the format of eloftr.evaluation.compare. kornia returns no coarse ids, but keypoints0 are the
    coarse cell corners (i % w, i // w) * stride, which gives them back exactly."""
    kpts0 = out['keypoints0'].float().cpu().numpy()
    cells = np.round(kpts0 / stride).astype(np.int64)
    return {'i': cells[:, 1] * (width // stride) + cells[:, 0],
            'kpts0': kpts0,
            'kpts1': out['keypoints1'].float().cpu().numpy(),
            'conf': out['confidence'].float().cpu().numpy()}
