# Coarse LoFTR TRT

[Google Colab demo notebook](https://colab.research.google.com/drive/1RFMAqfJeDaBoBQ7p5zXtJNXZE7DHGqlt?usp=sharing)

This project provides a deep learning model for the `Local Feature Matching` for two images that can be used on the embedded devices like NVidia Jetson Nano 2GB with a reasonable accuracy and performance - `5 FPS`. The algorithm is based on the `coarse part` of "LoFTR: Detector-Free Local Feature Matching with Transformers". But the model has a reduced number of ResNet and coarse transformer layers so there is the much lower memory consumption and the better performance. The required level of accuracy was achieved by applying the `Knowledge distillation` technique and training on the [BlendedMVS](https://github.com/YoYo000/BlendedMVS) dataset.

The code is based on the original [LoFTR](https://github.com/zju3dv/LoFTR) repository, but was adapted for compatibility with [TensorRT](https://developer.nvidia.com/tensorrt) technology, especially dependencies to `einsum` and `einops` were removed.

### Environment

The repository uses [pixi](https://pixi.sh) (Python 3.11, PyTorch 2.4.1, TensorRT 10.9 from PyPI, CUDA 12 driver):
```
pixi install
pixi run init-submodules   # fetches third_party/EfficientLoFTR
```

Datasets are stored under `DATASET_DIR`, set in `run/env.sh` (default `~/dataset`; a value already exported in the shell takes precedence). `pixi run download-scannet` downloads the ScanNet-1500 test set (1500 indoor pairs from 100 scenes, 1.1 GB) from the LoFTR Google Drive folder to `$DATASET_DIR/scannet`.

### Repository layout

* `eloftr/`, `loftr_full/`: fixed-shape EfficientLoFTR and LoFTR for ONNX/TensorRT, the TensorRT runtime (`eloftr/trt_runtime.py`) and evaluation helpers (`eloftr/evaluation.py`, `eloftr/pose.py`).
* `loftr/`, `train/`: the legacy Coarse LoFTR model and its distillation training; shared helpers are in `loftr/utils/helpers.py`.
* `scripts/trt/`: ONNX export and TensorRT engine building (`export_eloftr_onnx.py`, `export_loftr_onnx.py`, legacy `export_onnx.py`, `build_trt_engine.py`).
* `scripts/eval/`: benchmarks (`eval_eloftr.py` on the sample images, `eval_scannet.py` on ScanNet-1500).
* `scripts/`: demos and training entry points (`match_eloftr.py`, `webcam.py`, `train_distill.py`, `compare.py`).
* `run/`: shell scripts (`env.sh`, `download_scannet.sh`, `export_trt.sh`, `start_tensorboard.sh`).

Scripts import the packages from the repository root, which pixi puts on `PYTHONPATH`; run them from the root with `pixi run python scripts/...` (or set `PYTHONPATH=.` in another environment).

### EfficientLoFTR + TensorRT

[EfficientLoFTR](https://github.com/zju3dv/EfficientLoFTR) is included as an unmodified, pinned git submodule in `third_party/EfficientLoFTR` (note its Project Registration License: research use is free, project use requires registration).
The `eloftr` package wraps it into `StaticELoFTR`, a fixed-shape version of the model for ONNX/TensorRT export: it reuses the upstream backbone, coarse transformer and fine FPN weights, and re-implements the coarse/fine matching without data-dependent shapes. Every coarse cell of the first image gets a candidate match, and a `valid` mask marks the matches upstream returns.

```
pixi run download-weights   # weights/eloftr/eloftr_outdoor.ckpt
pixi run build-default      # 640x480 and 384x384 engines in weights/eloftr
pixi run eval               # accuracy and speed vs the torch model
pixi run match <image0> <image1> --engine weights/eloftr/eloftr_full_mixed_480x640.engine
```

Engines for other sizes (multiples of 32), model types and precisions are built with `pixi run build-engine <height> <width> <full|opt> <mixed|fp32>`.

* `full` uses dual-softmax coarse matching (upstream default, recommended); `opt` thresholds the raw similarity (upstream "optimized" setting, same checkpoint; about 15% faster in TensorRT at 640x480 with the same ground-truth accuracy on the sample images).
* `mixed` runs the backbone, transformer and fine FPN in fp16, and the coarse similarity, fine-window matmuls, LayerNorms and all softmaxes in fp32. Keeping the small fine-window matmuls in fp32 matters: in fp16 their rounded outputs create ties in the fine argmax that measurably lower the keypoint accuracy, at no speed gain. The precision is expressed by casts in the exported ONNX, which is built as a strongly-typed TensorRT network (the only mode in TensorRT 11) with TF32 disabled. `scripts/trt/export_eloftr_onnx.py --fp16 backbone,coarse,fine,fine_matching` selects the fp16 module groups explicitly.
* Engine outputs: `keypoints0`, `keypoints1` (`[L, 2]`, pixels of the engine input), `confidence` (`[L]`) and `valid` (`[L]`, bool), with `L = H/8 * W/8`. `eloftr.matcher.ELoFTRMatcher` resizes the input images, filters the matches and maps them back to the original image size.
* Every ONNX model and engine has a JSON sidecar (`*.onnx.json`, `*.engine.json`) with its shape, precision and versions. Engines only load with the TensorRT version that built them, so rebuild them from the ONNX models for other TensorRT versions.

`scripts/trt/export_eloftr_onnx.py` checks that `StaticELoFTR` in fp32 reproduces the upstream matches before exporting, and `scripts/trt/build_trt_engine.py` compares each engine with the torch model. `scripts/eval/eval_eloftr.py` compares the engines with the upstream torch model computed in fp32 on the 18 upstream sample image pairs and on synthetic homographies of the same images. It reports match agreement, RANSAC inliers, ground-truth homography precision and latency, and checks acceptance targets.

Results on an RTX 3090 (`pixi run eval`, latency of image tensors on the GPU to filtered matches; timings vary by about 15% between runs on a GPU that also drives a desktop):

| Model | Size | Latency | Matches | H@1px | H@3px | H error |
|---|---|---|---|---|---|---|
| torch upstream fp32 | 640x480 | 37-39 ms | 459 | 0.9713 | 0.9951 | 0.337 px |
| torch upstream mixed precision (`mp`) | 640x480 | 28-29 ms | 459 | 0.9707 | 0.9950 | 0.336 px |
| TensorRT mixed | 640x480 | 14-17 ms | 459 | 0.9712 | 0.9950 | 0.335 px |
| torch upstream fp32 | 384x384 | 18-19 ms | 212 | 0.9683 | 0.9941 | 0.343 px |
| TensorRT mixed | 384x384 | 6-7 ms | 212 | 0.9675 | 0.9941 | 0.342 px |

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

### ScanNet benchmark

`scripts/eval/eval_scannet.py` measures relative pose accuracy and speed on ScanNet-1500 (`pixi run download-scannet` first) with the protocol of the upstream EfficientLoFTR test: images resized to 640x480, essential matrix by OpenCV RANSAC (0.5 px, confidence 0.99999) on 5 shuffles of the matches, AUC of the pose error max(rotation, translation angle) at 5/10/20 degrees, and precision as the fraction of matches with a symmetric epipolar distance below 5e-4. The metrics (`eloftr/pose.py`) reproduce upstream `src/utils/metrics.py`. It evaluates every engine in `weights/eloftr`, `weights/loftr` and the legacy `weights/LoFTR_teacher.engine`, each next to the torch model it was exported from at the same input size and matching settings, and checks that every engine stays within 1 AUC point and 3% of the match count of its torch model.

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
| EfficientLoFTR TensorRT mixed | 952 | 19.88 | 37.78 | 54.16 | 72.31 | 11.8 ms |
| LoFTR outdoor, kornia fp32 | 812 | 17.57 | 34.45 | 50.60 | 69.82 | 51.3 ms |
| LoFTR outdoor, TensorRT fp32 | 812 | 17.46 | 34.30 | 50.56 | 69.82 | 51.5 ms |
| LoFTR outdoor, TensorRT mixed | 812 | 17.40 | 34.19 | 50.45 | 69.83 | 24.8 ms |
| Coarse LoFTR student, torch fp32 | 440 | 0.03 | 0.30 | 2.11 | 41.89 | 3.3 ms |
| Coarse LoFTR student, TensorRT fp16 | 440 | 0.07 | 0.37 | 2.19 | 41.89 | 0.9 ms |
| EfficientLoFTR torch `mp`, thr 0.1, no border removal (`--paper`) | 1335 | 18.60 | 36.63 | 53.59 | 65.05 | 23.4 ms |
| LoFTR indoor_new (ScanNet weights), kornia fp32 (`--paper`) | 962 | 21.69 | 40.62 | 57.57 | 88.11 | 53.9 ms |

* The engines bake in the default matching settings of their model (threshold 0.2, border removal 2), and the first eight rows use them. The published EfficientLoFTR ScanNet result, 19.2/37.0/53.6 ([paper](https://arxiv.org/abs/2403.04765), Table 1), uses threshold 0.1 without border removal; it and the published LoFTR results (outdoor weights 16.9/33.6/50.6 in the same table, ScanNet weights 22.06/40.80/57.62 in the LoFTR paper) are reproduced within 0.9 AUC points; for scale, torch fp32 and `mp` of the same model differ by up to 0.5.
* The TensorRT engines are within 0.3 AUC points and 0.2% of the match count of their torch models, at 2.5x (EfficientLoFTR, vs torch fp32) and 2.1x (LoFTR mixed) the speed.
* The Coarse LoFTR student matches cells of its 1/16 coarse grid (16 px) without refinement, which the 0.5 px RANSAC threshold of the protocol does not tolerate.
* LoFTR with the ScanNet weights finds no matches on 73 of the 1500 pairs; they count as failed poses, as upstream does.

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
