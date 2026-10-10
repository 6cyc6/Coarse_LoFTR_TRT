import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from eloftr.matcher import ELoFTRMatcher, engine_metadata
from eloftr.upstream import DEFAULT_CKPT


def draw_matches(image0, image1, kpts0, kpts1, conf):
    height = max(image0.shape[0], image1.shape[0])
    canvas = np.zeros((height, image0.shape[1] + image1.shape[1], 3), np.uint8)
    canvas[:image0.shape[0], :image0.shape[1]] = image0
    canvas[:image1.shape[0], image0.shape[1]:] = image1
    if len(conf):
        scores = (conf - conf.min()) / max(conf.max() - conf.min(), 1e-6)
        colors = cv2.applyColorMap((scores * 255).astype(np.uint8)[:, None], cv2.COLORMAP_JET)[:, 0]
        for p0, p1, color in zip(kpts0, kpts1, colors):
            p0 = (int(round(p0[0])), int(round(p0[1])))
            p1 = (int(round(p1[0])) + image0.shape[1], int(round(p1[1])))
            cv2.line(canvas, p0, p1, color.tolist(), 1, cv2.LINE_AA)
    return canvas


def make_matcher(opt):
    """The matcher of the engine's model (sidecar `model`), or of --model in torch."""
    model = engine_metadata(opt.engine).get('model') if opt.engine is not None else opt.model
    if model in ('loma', 'loma-b128', 'loma-b'):
        from loma_trt.matcher import LoMaMatcher
        variant = model.removeprefix('loma-') if model != 'loma' else None
        return LoMaMatcher(opt.engine, variant, opt.height, opt.width, precision=opt.precision or 'fp32')
    if model == 'romav2':
        from romav2_trt.matcher import RoMaV2Matcher
        return RoMaV2Matcher(opt.engine, opt.height, opt.width, opt.precision or 'fp32', num_matches=opt.num_matches)
    return ELoFTRMatcher(opt.engine, opt.height, opt.width, opt.model_type, opt.precision or 'fp32', ckpt=opt.ckpt)


def main():
    parser = argparse.ArgumentParser(description='Match two images with EfficientLoFTR, LoFTR, LoMa or RoMa v2.')
    parser.add_argument('image0', type=Path)
    parser.add_argument('image1', type=Path)
    parser.add_argument('--engine', type=Path, default=None,
                        help='TensorRT engine (its sidecar names the model); torch is used if omitted.')
    parser.add_argument('--model', choices=['eloftr', 'loma-b128', 'loma-b', 'romav2'], default='eloftr',
                        help='Torch model without --engine.')
    parser.add_argument('--height', type=int, default=480, help='Torch input height (engines carry their own).')
    parser.add_argument('--width', type=int, default=640, help='Torch input width (engines carry their own).')
    parser.add_argument('--model-type', choices=['full', 'opt'], default='full', help='EfficientLoFTR model type.')
    parser.add_argument('--precision', default=None,
                        help='Torch precision: fp32 (default), mixed (EfficientLoFTR), bf16 or fp16 (LoMa, RoMa v2).')
    parser.add_argument('--num-matches', type=int, default=5000, help='Matches sampled from the RoMa v2 warp.')
    parser.add_argument('--ckpt', type=Path, default=DEFAULT_CKPT)
    parser.add_argument('--out', type=Path, default=Path('outputs/matches.jpg'))
    opt = parser.parse_args()

    matcher = make_matcher(opt)
    image0, image1 = cv2.imread(str(opt.image0)), cv2.imread(str(opt.image1))
    if image0 is None or image1 is None:
        raise FileNotFoundError('Failed to read the input images')
    matcher(image0, image1)  # warm-up
    start = time.perf_counter()
    kpts0, kpts1, conf = matcher(image0, image1)
    elapsed = time.perf_counter() - start
    print(f'{len(conf)} matches in {elapsed * 1000:.1f} ms '
          f'({"TensorRT" if opt.engine else "torch"}, {matcher.height}x{matcher.width}, incl. resize and copies)')

    opt.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(opt.out), draw_matches(image0, image1, kpts0, kpts1, conf))
    print(f'Saved {opt.out}')


if __name__ == '__main__':
    main()
