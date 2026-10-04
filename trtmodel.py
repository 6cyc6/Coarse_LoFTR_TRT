import numpy as np
import torch

from eloftr.trt_runtime import TRTEngine


class TRTModel:
    """Legacy Coarse LoFTR engine runner on the TensorRT 10 tensor API (torch buffers instead of pycuda)."""

    def __init__(self, engine_path, dtype=np.float32):
        self.engine_path = engine_path
        self.dtype = dtype
        self.engine = TRTEngine(engine_path)

    def __call__(self, left_image: np.ndarray, right_image: np.ndarray):
        left, right = (torch.from_numpy(np.ascontiguousarray(x, dtype=self.dtype)) for x in (left_image, right_image))
        outputs = list(self.engine(left, right).values())
        return outputs[1].cpu().numpy().ravel()
