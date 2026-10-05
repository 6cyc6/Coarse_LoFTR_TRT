"""EfficientLoFTR export to fixed-shape ONNX and TensorRT engines."""
import hashlib
from pathlib import Path


def sidecar_path(path):
    """Metadata JSON stored next to an ONNX model or engine: model.onnx -> model.onnx.json."""
    path = Path(path)
    return path.with_name(path.name + '.json')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
