"""EfficientLoFTR export to fixed-shape ONNX and TensorRT engines."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sidecar_path(path):
    """Metadata JSON stored next to an ONNX model or engine: model.onnx -> model.onnx.json."""
    path = Path(path)
    return path.with_name(path.name + '.json')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plugin_paths(meta):
    """Plugin libraries listed in an ONNX/engine sidecar, relative paths taken from the repository root."""
    return [p if Path(p).is_absolute() else ROOT / p for p in meta.get('plugins', [])]
