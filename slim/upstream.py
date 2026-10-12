"""Loader for the pinned upstream SLiM source (third_party/SLiM, kept unmodified)."""
import importlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_DIR = ROOT / 'third_party' / 'SLiM'
WEIGHTS_DIR = ROOT / 'weights' / 'slim'
DEFAULT_CKPT = UPSTREAM_DIR / 'ckpt' / 'megadepth_19epochs.ckpt'
_UPSTREAM = {}


def _isolated_import(names):
    """Import modules of upstream SLiM's `src` package.

    SLiM and upstream EfficientLoFTR both name their package `src` (and SLiM imports `src.` absolutely), so other
    `src` modules and the path entries providing them are set aside while SLiM's are imported, and restored
    afterwards (SLiM's `src` has no __init__.py, and a regular `src` package anywhere on sys.path would take
    precedence). SLiM imports everything it needs at module level, so its modules keep working once they are no
    longer registered under `src`.
    """
    stashed = {k: sys.modules.pop(k) for k in list(sys.modules) if k == 'src' or k.startswith('src.')}
    path = sys.path[:]
    sys.path[:] = [str(UPSTREAM_DIR)] + [p for p in path if not (Path(p or '.') / 'src').is_dir()]
    importlib.invalidate_caches()
    try:
        return [importlib.import_module(name) for name in names]
    finally:
        sys.path[:] = path
        for k in [k for k in sys.modules if k == 'src' or k.startswith('src.')]:
            del sys.modules[k]
        sys.modules.update(stashed)


def import_upstream():
    """Upstream modules: {'slim': src.slim, 'convmamba': src.backbone.convmamba, 'csm': .vssm.csm_triton}."""
    if not (UPSTREAM_DIR / 'src' / 'slim.py').is_file():
        raise FileNotFoundError(f'{UPSTREAM_DIR} is empty, run: pixi run init-submodules')
    if not _UPSTREAM:
        names = ['src.slim', 'src.backbone.convmamba', 'src.backbone.vssm.csm_triton', 'src.backbone.vssm.csms6s']
        _UPSTREAM.update(zip(('slim', 'convmamba', 'csm', 'csms6s'), _isolated_import(names)))
    return _UPSTREAM


def upstream_commit():
    try:
        return subprocess.check_output(['git', '-C', str(UPSTREAM_DIR), 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def make_config(config_name='indoor_test'):
    """Upstream config (configs/cfg_<config_name>.py), loaded by path: EfficientLoFTR also has a `configs` package."""
    path = UPSTREAM_DIR / 'configs' / f'cfg_{config_name}.py'
    spec = importlib.util.spec_from_file_location(f'slim_cfg_{config_name}', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_cfg_defaults()


def load_upstream(ckpt=DEFAULT_CKPT, config_name='indoor_test', device='cuda'):
    """Upstream SLiM in eval mode with the test settings of `config_name` (as test.py builds it, batch size 1)."""
    ckpt = Path(ckpt)
    if not ckpt.exists():
        raise FileNotFoundError(f'{ckpt} not found, run: pixi run init-submodules')
    config = make_config(config_name)
    model = import_upstream()['slim'].SLiM(config=config['MODEL'])
    model.load_state_dict(torch.load(ckpt, map_location='cpu')['state_dict'], strict=True)
    return model.eval().to(device)


@torch.no_grad()
def run_upstream(model, image0, image1, amp=False):
    """Upstream test inference (test_step), returns its data dict: fine_coord_0/1 [N, 2] in input pixels, ..."""
    data = {'image0': image0, 'image1': image1}
    with torch.autocast(device_type='cuda', enabled=amp):
        model(data)
    return data
