# Coarse LoFTR TRT

[Google Colab demo notebook](https://colab.research.google.com/drive/1RFMAqfJeDaBoBQ7p5zXtJNXZE7DHGqlt?usp=sharing)

This project provides a deep learning model for the `Local Feature Matching` for two images that can be used on the embedded devices like NVidia Jetson Nano 2GB with a reasonable accuracy and performance - `5 FPS`. The algorithm is based on the `coarse part` of "LoFTR: Detector-Free Local Feature Matching with Transformers". But the model has a reduced number of ResNet and coarse transformer layers so there is the much lower memory consumption and the better performance. The required level of accuracy was achieved by applying the `Knowledge distillation` technique and training on the [BlendedMVS](https://github.com/YoYo000/BlendedMVS) dataset.

The code is based on the original [LoFTR](https://github.com/zju3dv/LoFTR) repository, but was adapted for compatibility with [TensorRT](https://developer.nvidia.com/tensorrt) technology, especially dependencies to `einsum` and `einops` were removed.

### Environment

The repository uses [pixi](https://pixi.sh) (Python 3.11, PyTorch 2.9.1 and torchvision 0.24.1 with CUDA 12.8, TensorRT 10.9 from PyPI, CUDA 12 driver):
```
pixi install
pixi run init-submodules   # fetches third_party/EfficientLoFTR, third_party/LoMa and third_party/RoMaV2
```

torch and TensorRT match the `trt` environment of the tracker that consumes the engines; engines only load with the TensorRT version that built them. torch 2.9 exports ONNX with the dynamo exporter by default, so the export scripts pass `dynamo=False` (the TorchScript exporter the models were validated with), and since torch 2.6 loads checkpoints with `weights_only=True`, so trusted local checkpoints that pickle Python objects (Lightning checkpoints) are loaded with `weights_only=False`.

Datasets are stored under `DATASET_DIR`, set in `run/env.sh` (default `~/dataset`; a value already exported in the shell takes precedence). `pixi run download-scannet` downloads the ScanNet-1500 test set (1500 indoor pairs from 100 scenes, 1.1 GB) from the LoFTR Google Drive folder to `$DATASET_DIR/scannet`.

### Repository layout

* `eloftr/`, `loftr_full/`: fixed-shape EfficientLoFTR and LoFTR for ONNX/TensorRT, the TensorRT runtime (`eloftr/trt_runtime.py`) and evaluation helpers (`eloftr/evaluation.py`, `eloftr/pose.py`).
* `loma_trt/`, `romav2_trt/`: fixed-shape LoMa and RoMa v2 (`upstream.py` loader, `static_model.py`, `evaluation.py` comparisons and engine smoke tests, `matcher.py`).
* `loftr/`, `train/`: the legacy Coarse LoFTR model and its distillation training; shared helpers are in `loftr/utils/helpers.py`.
* `scripts/trt/`: ONNX export and TensorRT engine building (`export_eloftr_onnx.py`, `export_loftr_onnx.py`, `export_loma_onnx.py`, `export_romav2_onnx.py`, legacy `export_onnx.py`, `build_trt_engine.py`).
* `scripts/eval/`: benchmarks (`eval_eloftr.py` on the sample images, `eval_scannet.py` on ScanNet-1500).
* `scripts/`: demos and training entry points (`match_eloftr.py`, `webcam.py`, `train_distill.py`, `compare.py`).
* `run/`: shell scripts (`env.sh`, `download_scannet.sh`, `export_trt.sh`, `start_tensorboard.sh`).

Scripts import the packages from the repository root, which pixi puts on `PYTHONPATH`; run them from the root with `pixi run python scripts/...` (or set `PYTHONPATH=.` in another environment).

### EfficientLoFTR + TensorRT

[EfficientLoFTR](https://github.com/zju3dv/EfficientLoFTR) is included as an unmodified, pinned git submodule in `third_party/EfficientLoFTR` (note its Project Registration License: research use is free, project use requires registration).
The `eloftr` package wraps it into `StaticELoFTR`, a fixed-shape version of the model for ONNX/TensorRT export: it reuses the upstream backbone, coarse transformer and fine FPN weights, and re-implements the coarse/fine matching without data-dependent shapes. Every coarse cell of the first image gets a candidate match, and a `valid` mask marks the matches upstream returns. Only cells outside upstream's border removal can be valid, so only they are refined at the fine level (exact; 8% faster at 384x384, 4% at 640x480).

```
pixi run download-weights   # weights/eloftr/eloftr_outdoor.ckpt
pixi run build-default      # full and opt, 640x480 and 384x384 engines in weights/eloftr
pixi run eval               # accuracy and speed vs the torch model
pixi run match <image0> <image1> --engine weights/eloftr/eloftr_full_mixed_480x640.engine
```

Engines for other sizes (multiples of 32), model types and precisions are built with `pixi run build-engine <height> <width> <full|opt> <mixed|fp32>`.

* `full` uses dual-softmax coarse matching (upstream default, recommended); `opt` thresholds the raw similarity at 25 instead (upstream "optimized" setting, same checkpoint). In TensorRT `opt` is 14% faster than `full` at 640x480 and 8% faster at 384x384. It is as accurate as `full` on the sample-image homographies, but about 2.5 AUC points lower on ScanNet-1500 (see below), as upstream `opt` in torch is. Upstream computes the `opt` coarse similarity in fp16 under mixed precision; `StaticELoFTR` keeps it in fp32 for both model types.
* `mixed` runs the backbone, transformer and fine FPN in fp16, and the coarse similarity, fine-window matmuls, LayerNorms and all softmaxes in fp32. Keeping the small fine-window matmuls in fp32 matters: in fp16 their rounded outputs create ties in the fine argmax that measurably lower the keypoint accuracy, at no speed gain. The precision is expressed by casts in the exported ONNX, which is built as a strongly-typed TensorRT network (the only mode in TensorRT 11) with TF32 disabled. `scripts/trt/export_eloftr_onnx.py --fp16 backbone,coarse,fine,fine_matching` selects the fp16 module groups explicitly.
* Engine outputs: `keypoints0`, `keypoints1` (`[L, 2]`, pixels of the engine input), `confidence` (`[L]`) and `valid` (`[L]`, bool), with `L = H/8 * W/8`. `eloftr.matcher.ELoFTRMatcher` resizes the input images, filters the matches and maps them back to the original image size.
* Every ONNX model and engine has a JSON sidecar (`*.onnx.json`, `*.engine.json`) with its shape, precision and versions. Engines only load with the TensorRT version that built them, so rebuild them from the ONNX models for other TensorRT versions.

`scripts/trt/export_eloftr_onnx.py` checks that `StaticELoFTR` in fp32 reproduces the upstream matches before exporting, and `scripts/trt/build_trt_engine.py` compares each engine with the torch model. `scripts/eval/eval_eloftr.py` compares the engines with the upstream torch model computed in fp32 on the 18 upstream sample image pairs and on synthetic homographies of the same images. It reports match agreement, RANSAC inliers, ground-truth homography precision and latency, and checks acceptance targets.

Results on an RTX 4070 Ti SUPER (`pixi run eval`, latency of image tensors on the GPU to filtered matches, without CUDA graphs; timings vary by about 15% between runs on a GPU that also drives a desktop):

| Model | Size | Latency | Matches | H@1px | H@3px | H error |
|---|---|---|---|---|---|---|
| `full` torch upstream fp32 | 640x480 | 29.2 ms | 459 | 0.9710 | 0.9950 | 0.337 px |
| `full` torch upstream mixed precision (`mp`) | 640x480 | 22.8 ms | 459 | 0.9699 | 0.9950 | 0.337 px |
| `full` TensorRT mixed | 640x480 | 11.3 ms | 460 | 0.9715 | 0.9950 | 0.335 px |
| `opt` torch upstream fp32 | 640x480 | 24.4 ms | 436 | 0.9733 | 0.9949 | 0.331 px |
| `opt` torch upstream `mp` | 640x480 | 17.2 ms | 437 | 0.9724 | 0.9950 | 0.332 px |
| `opt` TensorRT mixed | 640x480 | 9.7 ms | 436 | 0.9736 | 0.9949 | 0.329 px |
| `full` torch upstream fp32 | 384x384 | 14.1 ms | 212 | 0.9680 | 0.9941 | 0.343 px |
| `full` torch upstream `mp` | 384x384 | 10.7 ms | 212 | 0.9679 | 0.9939 | 0.343 px |
| `full` TensorRT mixed | 384x384 | 4.8 ms | 212 | 0.9687 | 0.9940 | 0.341 px |
| `opt` torch upstream fp32 | 384x384 | 13.1 ms | 186 | 0.9706 | 0.9936 | 0.337 px |
| `opt` torch upstream `mp` | 384x384 | 9.5 ms | 186 | 0.9702 | 0.9933 | 0.337 px |
| `opt` TensorRT mixed | 384x384 | 4.5 ms | 185 | 0.9708 | 0.9935 | 0.336 px |

H@1px/H@3px: fraction of matches within 1/3 px of the ground truth on the synthetic homographies, H error: mean error of the matches within 5 px. Note that the fine-level argmax picks one pixel pair inside an 8x8 window and is sensitive to precision: upstream's own default inference (fine-level einsums under fp16 autocast) already moves about 11% of the matches by more than 1 px relative to an all-fp32 run, without changing the accuracy. The TensorRT `mixed` engines stay within that variation.

### LoFTR + TensorRT

The original [LoFTR](https://github.com/zju3dv/LoFTR) (coarse and fine stages) is taken from `kornia.feature.LoFTR` with its released weights: `outdoor` (MegaDepth), `indoor_new` or `indoor` (ScanNet), downloaded to `weights/loftr` on first use. `loftr_full.StaticLoFTR` is its fixed-shape version, built like `StaticELoFTR`: it reuses the kornia modules and weights, computes the linear attention with batched matmuls instead of einsums, and re-implements the coarse/fine matching for every coarse cell. The engines have the same outputs and sidecars as the EfficientLoFTR engines, so `ELoFTRMatcher` and `scripts/match_eloftr.py` run them too.

```
pixi run build-loftr-engine   # weights/loftr/loftr_outdoor_mixed_480x640.engine
pixi run match <image0> <image1> --engine weights/loftr/loftr_outdoor_mixed_480x640.engine
```

Engines for other sizes (multiples of 8), weights and precisions are built with `pixi run build-loftr-engine <height> <width> <outdoor|indoor_new|indoor> <mixed|fp32>`. `mixed` runs the backbone and the linear layers of both transformers in fp16, and the attention itself (its normalizer sums over all coarse tokens and can overflow fp16), the coarse similarity, LayerNorms, softmaxes and the fine expectation in fp32. `scripts/trt/export_loftr_onnx.py` checks that `StaticLoFTR` in fp32 reproduces the kornia matches before exporting (it uses the sample images of the EfficientLoFTR submodule), and `scripts/trt/build_trt_engine.py` compares each engine with the torch model.

Results on an RTX 4070 Ti SUPER at 640x480, with the metrics of `scripts/eval/eval_eloftr.py` against kornia LoFTR in fp32 (measured once; `pixi run eval-scannet` below covers the LoFTR engines):

| Model | Latency | Matches | H@1px | H@3px | H error |
|---|---|---|---|---|---|
| kornia LoFTR fp32 | 52 ms | 411 | 0.9502 | 0.9978 | 0.363 px |
| TensorRT fp32 | 51 ms | 411 | 0.9502 | 0.9978 | 0.363 px |
| TensorRT mixed | 26 ms | 411 | 0.9501 | 0.9977 | 0.363 px |

The static model refines all 4800 coarse cells at the fine level, where kornia refines only the ~400 matches, which is why the fp32 engine is not faster than kornia.

### LoMa + TensorRT

[LoMa](https://github.com/davnords/LoMa) is included as an unmodified, pinned git submodule in `third_party/LoMa` (MIT; its matcher keeps LightGlue's Apache-2.0 license). Two variants are supported: LoMa-B128 (`LoMaB128`: DaD detector, DeDoDe-B descriptor with 128-d descriptions, 9-layer LightGlue-style matcher) and LoMa-B (`LoMaB`: the same with the DeDoDe-G descriptor, which adds DINOv2 ViT-L/14). Upstream downloads its weights to the torch hub cache on first use (LoMa-B128 150 MB; LoMa-B 760 MB, plus 1.2 GB of DINOv2 weights its constructor fetches and then overwrites).

`loma_trt.StaticLoMa` is upstream's tensor API at a fixed size: DaD detects 2048 keypoints per image (dense scoremap, 3x3 NMS, top-k, sub-pixel refinement), DeDoDe describes them at the same size, and the transformer matches them. Every keypoint of the first image keeps its best match, and a `valid` mask marks the mutual nearest neighbours above upstream's threshold 0.1, so the engines have the output contract of the EfficientLoFTR engines with `L = 2048`: `keypoints0`, `keypoints1` (`[L, 2]`), `confidence` and `valid` (`[L]`). Inputs are RGB, `[1, 3, H, W]` in [0, 1], and keypoints are OpenCV pixels of the engine input (pixel centres at integers; upstream's `to_pixel_coords` minus 0.5). `loma_trt.matcher.LoMaMatcher` resizes images, filters the matches and maps them back.

```
pixi run build-loma-default   # LoMa-B128 384x384 and 640x480, LoMa-B 392x392 and 672x504, fp16, in weights/loma
pixi run match <image0> <image1> --engine weights/loma/loma_b_fp16_392x392.engine
```

Other engines: `pixi run build-loma-engine <b128|b> <height> <width> <fp16|bf16|fp32>`, sizes multiples of 8 (of 56 for LoMa-B, whose DINOv2 has 14-pixel patches); `scripts/trt/export_loma_onnx.py --num-keypoints` changes the keypoint count (up to 3840, the TensorRT top-k limit).

* Precision: upstream runs the CNNs, DINOv2 and the transformer under bf16 autocast (fp16 before Ampere). The engines default to fp16 in the same places: all activations stay far below the fp16 range (DINOv2 peaks at about 400), TensorRT runs it faster, and it agrees better with fp32. Which 2048 keypoints survive the top-k, and where the sub-pixel refinement puts them, depends on the precision: on the sample images at 384x384, upstream's own bf16 keeps 80% of the fp32 keypoints and a Jaccard index of 0.70 between the match sets; the fp16 engine keeps 96% and 0.94, the bf16 engine 84% and 0.78. The scoremap logits, softmaxes, NMS, top-k, sampling and matching run in fp32 in every mode. TensorRT 10.9 has no bf16 Resize, so interpolations run in fp32 and round once, as PyTorch's half-precision kernels do.
* `scripts/trt/export_loma_onnx.py` checks that `StaticLoMa` in fp32 reproduces upstream in fp32 (autocast off, DINOv2 cast back) before exporting: keypoints are aligned by mutual nearest neighbour, then matches are compared like the EfficientLoFTR ones (Jaccard 0.993 to 1.0, zero median displacement error). The engine smoke test compares an engine with `StaticLoMa` in fp32 and accepts it if it agrees about as well as `StaticLoMa` in torch at the engine's precision does.
* Two exporter pitfalls are avoided in `StaticLoMa`: upstream's `unflatten(-1, ...)` exports as a wrong reshape, and TorchScript pools equal constants, after which the export rewrites a stack's `-1` dim in place and corrupts other ops sharing it. The model therefore uses explicit reshapes, positive dims and Python-int shapes.

On an RTX 3090 the fp16 engines take 17 ms (LoMa-B128, 384x384), 32 ms (LoMa-B128, 640x480), 41 ms (LoMa-B, 392x392) and 84 ms (LoMa-B, 672x504) per pair, 2.6-3.7x faster than upstream in its mixed precision, with the same ScanNet-1500 accuracy (within 0.25 AUC points, see [ScanNet benchmark](#scannet-benchmark)).

### RoMa v2 + TensorRT

[RoMa v2](https://github.com/Parskatt/RoMaV2) is included as an unmodified, pinned git submodule in `third_party/RoMaV2` (MIT; its DINOv3 backbone has the [DINOv3 license](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md)). Upstream fetches the DINOv3 code with torch.hub at a pinned commit and downloads its weights (1.1 GB, DINOv3 included) to the torch hub cache on first use.

`romav2_trt.StaticRoMaV2` is upstream's single-resolution forward, `RoMaV2.forward(img_A, img_B)`, at a fixed size: DINOv3 ViT-L/16 features (layers 11 and 17; the blocks after 17 are dropped), the multi-view transformer, global softmax matching, the DPT head, and the refiners at 1/4, 1/2 and full resolution. The engines output upstream's dense predictions, `warp_AB` (`[1, H, W, 2]`, normalized coordinates in image1 of every pixel of image0) and `confidence_AB` (`[1, H, W, 4]`, overlap logit and the parameters of a 2x2 precision matrix); `--bidirectional` adds `warp_BA` and `confidence_BA`. Matches are sampled from them outside the engine by upstream's `RoMaV2.sample` (multinomial on the overlap, balanced by a kernel density estimate), seeded, in `romav2_trt.matcher.RoMaV2Matcher`, which returns OpenCV pixels like the other matchers. Upstream's second, high-resolution refinement pass (its `precise` setting) is not part of the engines.

```
pixi run build-romav2-default   # 384x384 and 640x480, fp16 (DINOv3 in bf16), in weights/romav2
pixi run match <image0> <image1> --engine weights/romav2/romav2_fp16_384x384.engine --num-matches 2000
```

Other engines: `pixi run build-romav2-engine <height> <width> <fp16|bf16|fp32>`, sizes multiples of 16 (`pixi run build-romav2-engine 320 320` is upstream's `turbo` setting); `scripts/trt/export_romav2_onnx.py --bidirectional` exports both directions.

* Precision: upstream runs DINOv3 (cast to bf16), the matcher, the DPT head and the refiner convolutions in bf16. DINOv3's residual stream holds activations of about 1.6e5, beyond the fp16 range, so `fp16` engines keep DINOv3 in bf16 and run the rest in fp16; `bf16` engines follow upstream. On the sample images at 384x384, the fp16 engine's warp differs from fp32 by a median of 0.013 px (2.2% of the overlapping pixels by more than 1 px), the bf16 engine's by 0.032 px (3.2%), and the fp16 engine is 13% faster. Global matching, the refiner projections, grid sampling, local correlation, the heads and all warp and confidence arithmetic run in fp32, as in upstream. TensorRT 10.9 has no bf16 Resize or ConvTranspose, so those run in fp32 in bf16 engines. At 320x320 the precision matters more: with DINOv3 in bf16 alone, the warp of a hard pair can jump by pixels in whole regions, and on the sample images upstream's own bf16 and the fp16 engine both move 9% of the overlapping pixels by more than 1 px against fp32; the engine smoke test therefore compares that share relative to the noise of the torch model at the same precision.
* Upstream's matcher applies RoPE in bf16 even in an fp32 run, which turns rounding-level differences (e.g. running DINOv3 on both images as one batch, which the engines do) into differences of about 1e-3. The export self-check therefore runs DINOv3 per image, as upstream does, and then reproduces upstream fp32 exactly (median warp difference 3e-6 px, identical overlap masks).
* Local correlation is computed as upstream's native implementation (grid sampling of the 7x7 and 3x3 windows); the optional fused CUDA kernel of upstream (`fused-local-corr`, which pins another torch version) is not installed, so upstream also uses the native path in the comparisons. At 384x384 the 7x7 correlation takes about a fifth of the engine's time.

On an RTX 3090 the fp16 engines take 24 ms (320x320), 34 ms (384x384) and 74 ms (640x480) per pair, against 49 ms and 63 ms for upstream's forward at 320x320 and 384x384 (timings on this GPU, which also drives a desktop, vary by up to 20% with its clock and temperature). Sampling 5000 matches with upstream's sampler adds 15.5 ms (its kernel density estimate compares 20000 candidates), 1000 matches 1.1 ms. ScanNet-1500 accuracy is that of upstream (within 0.3 AUC points, see [ScanNet benchmark](#scannet-benchmark)).

### ScanNet benchmark

`scripts/eval/eval_scannet.py` measures relative pose accuracy and speed on ScanNet-1500 (`pixi run download-scannet` first) with the protocol of the upstream EfficientLoFTR test: images resized to 640x480, essential matrix by OpenCV RANSAC (0.5 px, confidence 0.99999) on 5 shuffles of the matches, AUC of the pose error max(rotation, translation angle) at 5/10/20 degrees, and precision as the fraction of matches with a symmetric epipolar distance below 5e-4. The metrics (`eloftr/pose.py`) reproduce upstream `src/utils/metrics.py`. It evaluates every engine in `weights/eloftr`, `weights/loftr`, `weights/loma`, `weights/romav2` and the legacy `weights/LoFTR_teacher.engine`, each next to the torch model it was exported from at the same input size and matching settings, and checks that every engine stays within 1 AUC point and 3% of the match count of its torch model.

```
pixi run eval-scannet                 # all engines and their torch models, report in outputs/eval/scannet.json
pixi run eval-scannet --paper         # also the published settings below
pixi run eval-scannet --limit 100 --engine weights/eloftr/eloftr_full_mixed_480x640.engine   # quick check
```

Results on an RTX 4070 Ti SUPER (`pixi run eval-scannet --paper`, about 30 minutes; latency is the median per pair from images on the GPU to filtered matches, engines without CUDA graphs):

| Model | Matches | AUC@5 | AUC@10 | AUC@20 | P@5e-4 | Latency |
|---|---|---|---|---|---|---|
| EfficientLoFTR torch fp32 | 950 | 19.86 | 37.68 | 53.99 | 72.40 | 30.0 ms |
| EfficientLoFTR torch mixed precision (`mp`) | 950 | 19.78 | 37.91 | 54.48 | 72.39 | 22.6 ms |
| EfficientLoFTR TensorRT mixed | 952 | 19.88 | 37.78 | 54.16 | 72.31 | 11.2 ms¹ |
| LoFTR outdoor, kornia fp32 | 812 | 17.57 | 34.45 | 50.60 | 69.82 | 51.3 ms |
| LoFTR outdoor, TensorRT fp32 | 812 | 17.46 | 34.30 | 50.56 | 69.82 | 51.5 ms |
| LoFTR outdoor, TensorRT mixed | 812 | 17.40 | 34.19 | 50.45 | 69.83 | 24.8 ms |
| Coarse LoFTR student, torch fp32 | 440 | 0.03 | 0.30 | 2.11 | 41.89 | 3.3 ms |
| Coarse LoFTR student, TensorRT fp16 | 440 | 0.07 | 0.37 | 2.19 | 41.89 | 0.9 ms |
| EfficientLoFTR torch `mp`, thr 0.1, no border removal (`--paper`) | 1335 | 18.60 | 36.63 | 53.59 | 65.05 | 23.4 ms |
| LoFTR indoor_new (ScanNet weights), kornia fp32 (`--paper`) | 962 | 21.69 | 40.62 | 57.57 | 88.11 | 53.9 ms |

* The engines bake in the default matching settings of their model (threshold 0.2, border removal 2), and the first eight rows use them. The published EfficientLoFTR ScanNet result, 19.2/37.0/53.6 ([paper](https://arxiv.org/abs/2403.04765), Table 1), uses threshold 0.1 without border removal; it and the published LoFTR results (outdoor weights 16.9/33.6/50.6 in the same table, ScanNet weights 22.06/40.80/57.62 in the LoFTR paper) are reproduced within 0.9 AUC points; for scale, torch fp32 and `mp` of the same model differ by up to 0.5.
* The TensorRT engines are within 0.3 AUC points and 0.2% of the match count of their torch models, at 2.7x (EfficientLoFTR, vs torch fp32) and 2.1x (LoFTR mixed) the speed.
* ¹ Measured with the full vs opt run below, after the fine stage was restricted to the interior cells (11.8 ms before, same matches).
* The Coarse LoFTR student matches cells of its 1/16 coarse grid (16 px) without refinement, which the 0.5 px RANSAC threshold of the protocol does not tolerate.

EfficientLoFTR `full` and `opt` at both engine sizes (`pixi run eval-scannet --engine weights/eloftr/*.engine`, a separate run; the 384x384 rows resize the 640x480 images to 384x384 and map the matches back):

| Model | Size | Matches | AUC@5 | AUC@10 | AUC@20 | P@5e-4 | Latency |
|---|---|---|---|---|---|---|---|
| `full` torch fp32 | 640x480 | 950 | 19.86 | 37.68 | 53.99 | 72.40 | 29.4 ms |
| `full` torch `mp` | 640x480 | 950 | 19.78 | 37.91 | 54.48 | 72.39 | 22.5 ms |
| `full` TensorRT mixed | 640x480 | 952 | 19.88 | 37.78 | 54.16 | 72.31 | 11.2 ms |
| `opt` torch fp32 | 640x480 | 1057 | 17.99 | 35.37 | 51.82 | 69.74 | 24.4 ms |
| `opt` torch `mp` | 640x480 | 1061 | 18.46 | 36.03 | 52.26 | 69.72 | 16.9 ms |
| `opt` TensorRT mixed | 640x480 | 1057 | 18.26 | 35.53 | 51.77 | 69.72 | 9.7 ms |
| `full` torch fp32 | 384x384 | 500 | 18.60 | 36.12 | 52.67 | 71.64 | 13.5 ms |
| `full` torch `mp` | 384x384 | 500 | 18.08 | 35.84 | 52.57 | 71.60 | 10.7 ms |
| `full` TensorRT mixed | 384x384 | 501 | 18.22 | 35.83 | 52.60 | 71.55 | 4.9 ms |
| `opt` torch fp32 | 384x384 | 517 | 16.99 | 33.48 | 49.31 | 69.82 | 12.4 ms |
| `opt` torch `mp` | 384x384 | 518 | 16.98 | 33.68 | 49.65 | 69.78 | 9.6 ms |
| `opt` TensorRT mixed | 384x384 | 517 | 17.21 | 34.04 | 50.08 | 69.87 | 4.5 ms |

The engines stay within 0.8 AUC points and 0.2% of the match count of their torch models; the torch rows themselves vary by up to 0.7 points between runs. `opt` returns about 10% more matches than `full` at a lower precision and loses 2.2-3.4 AUC@20 points across these rows, so `full` remains the recommended model type.
* LoFTR with the ScanNet weights finds no matches on 73 of the 1500 pairs; they count as failed poses, as upstream does.

LoMa and RoMa v2 at their engine sizes (`pixi run eval-scannet --engine weights/loma/*.engine weights/romav2/*.engine`, a separate run on an RTX 3090, about 45 minutes). Each engine is compared with upstream in its default mixed precision (`mp`, bf16 autocast) on the same RGB images, resized from the original images to the engine size; the matches are mapped back to 640x480. RoMa v2 samples 5000 matches with upstream's sampler from a fixed seed, and its latency includes the sampling:

| Model | Size | Matches | AUC@5 | AUC@10 | AUC@20 | P@5e-4 | Latency |
|---|---|---|---|---|---|---|---|
| LoMa-B128 torch `mp` | 384x384 | 567 | 24.55 | 46.23 | 64.88 | 86.21 | 70.8 ms |
| LoMa-B128 TensorRT fp16 | 384x384 | 572 | 24.59 | 46.00 | 64.92 | 86.15 | 19.1 ms |
| LoMa-B128 torch `mp` | 640x480 | 567 | 25.32 | 46.88 | 65.55 | 85.82 | 104.2 ms |
| LoMa-B128 TensorRT fp16 | 640x480 | 573 | 25.42 | 46.84 | 65.37 | 85.82 | 32.2 ms |
| LoMa-B torch `mp` | 392x392 | 590 | 26.99 | 48.92 | 67.66 | 87.49 | 137.5 ms |
| LoMa-B TensorRT fp16 | 392x392 | 594 | 27.07 | 48.79 | 67.48 | 87.43 | 40.8 ms |
| LoMa-B torch `mp` | 672x504 | 574 | 27.69 | 49.97 | 68.58 | 87.54 | 224.9 ms |
| LoMa-B TensorRT fp16 | 672x504 | 580 | 27.74 | 49.86 | 68.47 | 87.55 | 85.8 ms |
| RoMa v2 torch `mp` | 320x320 | 5000 | 29.29 | 51.82 | 70.38 | 87.97 | 69.5 ms |
| RoMa v2 TensorRT fp16 | 320x320 | 5000 | 29.46 | 51.88 | 70.49 | 88.01 | 40.1 ms |
| RoMa v2 torch `mp` | 320x320 | 1000 | 29.20 | 51.56 | 70.08 | 89.17 | 52.0 ms |
| RoMa v2 TensorRT fp16 | 320x320 | 1000 | 29.02 | 51.49 | 70.15 | 89.21 | 24.8 ms |
| RoMa v2 torch `mp` | 384x384 | 5000 | 31.49 | 54.40 | 72.38 | 89.12 | 96.7 ms |
| RoMa v2 TensorRT fp16 | 384x384 | 5000 | 31.78 | 54.48 | 72.45 | 89.13 | 54.8 ms |
| RoMa v2 torch `mp` | 640x480 | 5000 | 32.79 | 55.53 | 73.25 | 89.68 | 167.3 ms |
| RoMa v2 TensorRT fp16 | 640x480 | 5000 | 32.71 | 55.30 | 73.03 | 89.67 | 100.1 ms |

* The engines stay within 0.3 AUC points and 1.1% of the match count of upstream. The 320x320 rows (upstream's `turbo` size) are from separate runs (`--engine weights/romav2/romav2_fp16_320x320.engine`, and `--romav2-matches 1000` for the 1000-match rows); 320x320 gives up 2.0-2.6 AUC points against 384x384 for 27% lower latency (29% for the engine alone), and sampling 1000 instead of 5000 matches costs at most another 0.45 points (about the run-to-run spread) for 38% lower latency.
* These are not the published settings: LoMa's own benchmark detects at 1024 pixels on the long side and describes at 784x784, and RoMa v2's runs at 800x800 with a 1024x1024 refinement pass in both directions.

### Model weights
Weights for the PyTorch model, ONNX model and TensorRT engine files are located in the `weights` folder.
The committed `weights/LoFTR_teacher.trt` was built with TensorRT 8; rebuild the engine for the TensorRT of the pixi environment with `pixi run build-legacy`, which writes `weights/LoFTR_teacher.engine`.

Weights for original LoFTR coarse module can be downloaded using the original [url](https://drive.google.com/drive/folders/1DOcOPZb3-5cWxLqn256AhwUVjBPifhuf?usp=sharing) that was provider by paper authors, now only the `outdoor-ds` file is supported.

### Demo

There is a Demo application, that can be ran with the `scripts/webcam.py` script. There are following parameters:
* `--weights` - The path to PyTorch model weights, for example 'weights/LoFTR_teacher.pt' or 'weights/outdoor_ds.ckpt'                       
* `--trt` - The path to the TensorRT engine, for example 'weights/LoFTR_teacher.engine'
* `--onnx` - The path to the ONNX model, for example 'weights/LoFTR_teacher.onnx'
* `--original` - If specified the original LoFTR model will be used, can be used only with `--weights` parameter
* `--camid` - OpenCV webcam video capture ID, usually 0 or 1, default 0
* `--device` - Selects the runtime back-end CPU or CUDA, default is CUDA

Sample command line:
```
pixi run python scripts/webcam.py --trt=weights/LoFTR_teacher.engine --camid=0
```

Demo application shows a window with pair of images captured with a camera. Initially there will be the two same images. Then you can choose a view of interest and press the `s` button, the view will be remembered and will be visible as the left image. Then you can change the view and press the `p` button to make a snapshot of the feature matching result, the corresponding features will be marked with the same numbers at the two images. If you press the `p` button again then application will allow you to change the view and repeat the feature matching process. Also this application shows the real-time FPS counter so you can estimate the model performance.

### Training

To repeat the training procedure you should use the low-res set of the [BlendedMVS](https://github.com/YoYo000/BlendedMVS) dataset. After download you can use the `scripts/train_distill.py` script to run training process. There are following parameters for this script:
* `--path` - Path to the dataset
* `--checkpoint_path` - Where to store a log information and checkpoints, default value is 'weights'
* `--weights` - Path to the LoFTR teacher model weights, default value is 'weights/outdoor_ds.ckpt'
                        
Sample command line:
```
pixi run python scripts/train_distill.py --path=/home/user/datasets/BlendedMVS --checkpoint_path=weights/experiment1/
```

Please use the `train/settings.py` script to configure the training process. Please notice that by default the following parameters are enabled:

```
self.batch_size = 32
self.batch_size_divider = 8  # Used for gradient accumulation
self.use_amp = True
self.epochs = 35
self.epoch_size = 5000
```

This set of parameters was chosen for training with the Nvidia GTX1060 GPU, which is the low level consumer level card. The `use_amp` parameter means the [automatic mixed precision](https://pytorch.org/docs/stable/amp.html) will be used to reduce the memory consumption and the training time. Also, the gradient accumulation technique is enabled with the `batch_size_divider` parameter, it means the actual batch size will be `32/8` but for larger batch size simulation the 8 batches will be averaged. Moreover, the actual size of the epoch is reduced with the `epoch_size` parameter, it means that on every epoch only 5000 dataset elements will be randomly picked from the whole dataset.


[Paper](https://arxiv.org/abs/2202.00770)

```bibtex
@misc{kolodiazhnyi2022local,
      title={Local Feature Matching with Transformers for low-end devices}, 
      author={Kyrylo Kolodiazhnyi},
      year={2022},
      eprint={2202.00770},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

[LoFTR Paper:](https://arxiv.org/pdf/2104.00680.pdf)

```bibtex
@article{sun2021loftr,
  title={{LoFTR}: Detector-Free Local Feature Matching with Transformers},
  author={Sun, Jiaming and Shen, Zehong and Wang, Yuang and Bao, Hujun and Zhou, Xiaowei},
  journal={{CVPR}},
  year={2021}
}
```
