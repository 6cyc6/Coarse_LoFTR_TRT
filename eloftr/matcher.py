"""EfficientLoFTR matcher at a fixed input size with a TensorRT or torch backend."""
import json

import cv2
import numpy as np
import torch

from eloftr import sidecar_path
from eloftr.static_model import StaticELoFTR, precision_groups
from eloftr.upstream import DEFAULT_CKPT, load_upstream


def engine_metadata(engine_path):
    return json.loads(sidecar_path(engine_path).read_text())


class ELoFTRMatcher:
    def __init__(self, engine=None, height=480, width=640, model_type='full', precision='fp32', fp16=None,
                 ckpt=DEFAULT_CKPT, device='cuda', cuda_graph=False):
        """Use the TensorRT `engine` if given, otherwise StaticELoFTR in torch with the given settings."""
        self.device = torch.device(device)
        if engine is not None:
            from eloftr.trt_runtime import TRTEngine
            meta = engine_metadata(engine)
            # LoFTR engines (loftr_full) share the output contract and have no model_type
            self.height, self.width, self.model_type = meta['height'], meta['width'], meta.get('model_type')
            self.engine = TRTEngine(engine, device)
            if cuda_graph:
                self.engine.capture_cuda_graph()
            self.model = None
        else:
            self.height, self.width, self.model_type = height, width, model_type
            upstream = load_upstream(ckpt, model_type, device=device)
            self.model = StaticELoFTR(upstream, height, width, precision_groups(precision, fp16)).to(self.device)
            self.engine = None

    @torch.no_grad()
    def match_tensors(self, image0, image1):
        """[1, 1, H, W] float images in [0, 1] at the matcher size -> (kpts0, kpts1, conf) CUDA tensors."""
        if self.engine is not None:
            out = self.engine(image0, image1)
            keypoints0, keypoints1, confidence, valid = (out[k] for k in ('keypoints0', 'keypoints1', 'confidence', 'valid'))
        else:
            keypoints0, keypoints1, confidence, valid = self.model(image0, image1)
        return keypoints0[valid], keypoints1[valid], confidence[valid]

    def __call__(self, image0, image1):
        """Match two uint8 images (grayscale or BGR, any size); keypoints are in the original image pixels."""
        tensors, scales = [], []
        for image in (image0, image1):
            if image.ndim == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            scales.append(np.array([image.shape[1] / self.width, image.shape[0] / self.height], dtype=np.float32))
            image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)
            tensors.append(torch.from_numpy(image)[None, None].to(self.device, torch.float32) / 255.)
        kpts0, kpts1, conf = (t.cpu().numpy() for t in self.match_tensors(*tensors))
        # pixel coordinates are scaled about the image origin like upstream does for resized inputs
        return kpts0 * scales[0], kpts1 * scales[1], conf
