"""LoMa matcher at a fixed input size with a TensorRT or torch backend."""
import cv2
import numpy as np
import torch

from eloftr.evaluation import to_tensor
from eloftr.matcher import engine_metadata


def rgb_tensors(images, width, height, device):
    """uint8 BGR (or grayscale) images of any size -> [1, 3, H, W] RGB tensors in [0, 1] at the matcher size, and the
    (x, y) scale of each image relative to it."""
    tensors, scales = [], []
    for image in images:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB if image.ndim == 2 else cv2.COLOR_BGR2RGB)
        scales.append(np.array([image.shape[1] / width, image.shape[0] / height], dtype=np.float32))
        tensors.append(to_tensor(cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA), device))
    return tensors, scales


def rescale(keypoints, scale):
    """OpenCV pixels (centres at integers) of the resized image -> pixels of the original image."""
    return (keypoints + 0.5) * scale - 0.5


class LoMaMatcher:
    def __init__(self, engine=None, variant='b128', height=384, width=384, num_keypoints=2048, precision='fp16',
                 device='cuda', cuda_graph=False):
        """Use the TensorRT `engine` if given, otherwise StaticLoMa in torch with the given settings."""
        self.device = torch.device(device)
        if engine is not None:
            from eloftr.trt_runtime import TRTEngine
            meta = engine_metadata(engine)
            if meta.get('model') != 'loma':
                raise ValueError(f'{engine} is not a LoMa engine: {meta.get("model")}')
            self.height, self.width, self.variant = meta['height'], meta['width'], meta['variant']
            self.engine = TRTEngine(engine, device)
            if cuda_graph:
                self.engine.capture_cuda_graph()
            self.model = None
        else:
            from loma_trt.static_model import StaticLoMa
            from loma_trt.upstream import load_upstream
            self.height, self.width, self.variant = height, width, variant
            upstream = load_upstream(variant, fp32=True)
            self.model = StaticLoMa(upstream, height, width, num_keypoints, precision).to(self.device).eval()
            self.engine = None

    @torch.no_grad()
    def match_tensors(self, image0, image1):
        """[1, 3, H, W] RGB float images in [0, 1] at the matcher size -> (kpts0, kpts1, conf) CUDA tensors, keypoints
        in OpenCV pixels of the matcher input."""
        if self.engine is not None:
            out = self.engine(image0, image1)
            keypoints0, keypoints1, confidence, valid = (out[k] for k in ('keypoints0', 'keypoints1', 'confidence',
                                                                          'valid'))
        else:
            keypoints0, keypoints1, confidence, valid = self.model(image0, image1)
        return keypoints0[valid], keypoints1[valid], confidence[valid]

    def __call__(self, image0, image1):
        """Match two uint8 images (BGR or grayscale, any size); keypoints are in OpenCV pixels of the originals."""
        (tensor0, tensor1), scales = rgb_tensors((image0, image1), self.width, self.height, self.device)
        kpts0, kpts1, conf = (t.cpu().numpy() for t in self.match_tensors(tensor0, tensor1))
        return rescale(kpts0, scales[0]), rescale(kpts1, scales[1]), conf
