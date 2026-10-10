"""RoMa v2 matcher at a fixed input size with a TensorRT or torch backend.

The model predicts a dense warp and confidence; matches are sampled from them with upstream's `RoMaV2.sample`
(multinomial on the overlap, then balanced by a kernel density estimate), seeded for reproducibility.
"""
from types import SimpleNamespace

import torch

from eloftr.matcher import engine_metadata
from loma_trt.matcher import rescale, rgb_tensors
from romav2_trt.upstream import import_upstream


def sample_matches(outputs, num_matches, seed=0, threshold=None):
    """Upstream match post-processing and sampling on (warp_AB, confidence_AB[, warp_BA, confidence_BA]) [1, H, W, *]
    -> (kpts0, kpts1 [N, 2] in OpenCV pixels of the model input, overlap [N])."""
    romav2 = import_upstream()
    bidirectional = len(outputs) == 4
    preds = {}
    for direction, warp, confidence in zip(('AB', 'BA'), outputs[::2], outputs[1::2]):
        overlap, precision = romav2._map_confidence(confidence=confidence.float().clone(), threshold=threshold)
        preds |= {f'warp_{direction}': warp.float(), f'overlap_{direction}': overlap,
                  f'precision_{direction}': precision}
    height, width = outputs[0].shape[1:3]
    with torch.random.fork_rng(devices=[outputs[0].device]):
        torch.manual_seed(seed)
        matches, overlap, _, _ = romav2.RoMaV2.sample(SimpleNamespace(bidirectional=bidirectional), preds, num_matches)
    kpts0, kpts1 = romav2.RoMaV2.to_pixel_coordinates(matches, height, width, height, width)
    return kpts0 - 0.5, kpts1 - 0.5, overlap


class RoMaV2Matcher:
    def __init__(self, engine=None, height=384, width=384, precision='bf16', bidirectional=False, num_matches=5000,
                 device='cuda', cuda_graph=False, seed=0):
        """Use the TensorRT `engine` if given, otherwise StaticRoMaV2 in torch with the given settings."""
        self.device = torch.device(device)
        self.num_matches, self.seed = num_matches, seed
        if engine is not None:
            from eloftr.trt_runtime import TRTEngine
            meta = engine_metadata(engine)
            if meta.get('model') != 'romav2':
                raise ValueError(f'{engine} is not a RoMa v2 engine: {meta.get("model")}')
            self.height, self.width, self.outputs = meta['height'], meta['width'], meta['outputs']
            self.engine = TRTEngine(engine, device)
            if cuda_graph:
                self.engine.capture_cuda_graph()
            self.model = None
        else:
            from romav2_trt.static_model import StaticRoMaV2
            from romav2_trt.upstream import load_upstream
            self.height, self.width = height, width
            upstream = load_upstream(fp32=True, bidirectional=bidirectional)
            self.model = StaticRoMaV2(upstream, height, width, precision, bidirectional=bidirectional)
            self.model = self.model.to(self.device).eval()
            self.engine = None

    @torch.no_grad()
    def predict(self, image0, image1):
        """Dense outputs (warp_AB, confidence_AB[, warp_BA, confidence_BA]) for [1, 3, H, W] RGB images in [0, 1]."""
        if self.engine is not None:
            out = self.engine(image0, image1)
            return tuple(out[k] for k in self.outputs)
        return self.model(image0, image1)

    @torch.no_grad()
    def match_tensors(self, image0, image1, num_matches=None):
        """[1, 3, H, W] RGB float images in [0, 1] at the matcher size -> (kpts0, kpts1, overlap) CUDA tensors,
        keypoints in OpenCV pixels of the matcher input."""
        return sample_matches(self.predict(image0, image1), num_matches or self.num_matches, self.seed)

    def __call__(self, image0, image1, num_matches=None):
        """Match two uint8 images (BGR or grayscale, any size); keypoints are in OpenCV pixels of the originals."""
        (tensor0, tensor1), scales = rgb_tensors((image0, image1), self.width, self.height, self.device)
        kpts0, kpts1, conf = (t.cpu().numpy() for t in self.match_tensors(tensor0, tensor1, num_matches))
        return rescale(kpts0, scales[0]), rescale(kpts1, scales[1]), conf
