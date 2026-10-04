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
* `mixed` runs the backbone, transformer and fine FPN in fp16, and the coarse similarity, fine-window matmuls, LayerNorms and all softmaxes in fp32. Keeping the small fine-window matmuls in fp32 matters: in fp16 their rounded outputs create ties in the fine argmax that measurably lower the keypoint accuracy, at no speed gain. The precision is expressed by casts in the exported ONNX, which is built as a strongly-typed TensorRT network (the only mode in TensorRT 11) with TF32 disabled. `export_eloftr_onnx.py --fp16 backbone,coarse,fine,fine_matching` selects the fp16 module groups explicitly.
* Engine outputs: `keypoints0`, `keypoints1` (`[L, 2]`, pixels of the engine input), `confidence` (`[L]`) and `valid` (`[L]`, bool), with `L = H/8 * W/8`. `eloftr.matcher.ELoFTRMatcher` resizes the input images, filters the matches and maps them back to the original image size.
* Every ONNX model and engine has a JSON sidecar (`*.onnx.json`, `*.engine.json`) with its shape, precision and versions. Engines only load with the TensorRT version that built them, so rebuild them from the ONNX models for other TensorRT versions.

`export_eloftr_onnx.py` checks that `StaticELoFTR` in fp32 reproduces the upstream matches before exporting, and `build_trt_engine.py` compares each engine with the torch model. `eval_eloftr.py` compares the engines with the upstream torch model computed in fp32 on the 18 upstream sample image pairs and on synthetic homographies of the same images. It reports match agreement, RANSAC inliers, ground-truth homography precision and latency, and checks acceptance targets.

Results on an RTX 3090 (`pixi run eval`, latency of image tensors on the GPU to filtered matches; timings vary by about 15% between runs on a GPU that also drives a desktop):

| Model | Size | Latency | Matches | H@1px | H@3px | H error |
|---|---|---|---|---|---|---|
| torch upstream fp32 | 640x480 | 37-39 ms | 459 | 0.9713 | 0.9951 | 0.337 px |
| torch upstream mixed precision (`mp`) | 640x480 | 28-29 ms | 459 | 0.9707 | 0.9950 | 0.336 px |
| TensorRT mixed | 640x480 | 14-17 ms | 459 | 0.9712 | 0.9950 | 0.335 px |
| torch upstream fp32 | 384x384 | 18-19 ms | 212 | 0.9683 | 0.9941 | 0.343 px |
| TensorRT mixed | 384x384 | 6-7 ms | 212 | 0.9675 | 0.9941 | 0.342 px |

H@1px/H@3px: fraction of matches within 1/3 px of the ground truth on the synthetic homographies, H error: mean error of the matches within 5 px. Note that the fine-level argmax picks one pixel pair inside an 8x8 window and is sensitive to precision: upstream's own default inference (fine-level einsums under fp16 autocast) already moves about 11% of the matches by more than 1 px relative to an all-fp32 run, without changing the accuracy. The TensorRT `mixed` engines stay within that variation.

### Model weights
Weights for the PyTorch model, ONNX model and TensorRT engine files are located in the `weights` folder.
The committed `weights/LoFTR_teacher.trt` was built with TensorRT 8; rebuild the engine for the TensorRT of the pixi environment with `pixi run build-legacy`, which writes `weights/LoFTR_teacher.engine`.

Weights for original LoFTR coarse module can be downloaded using the original [url](https://drive.google.com/drive/folders/1DOcOPZb3-5cWxLqn256AhwUVjBPifhuf?usp=sharing) that was provider by paper authors, now only the `outdoor-ds` file is supported.

### Demo

There is a Demo application, that can be ran with the `webcam.py` script. There are following parameters:
* `--weights` - The path to PyTorch model weights, for example 'weights/LoFTR_teacher.pt' or 'weights/outdoor_ds.ckpt'                       
* `--trt` - The path to the TensorRT engine, for example 'weights/LoFTR_teacher.engine'
* `--onnx` - The path to the ONNX model, for example 'weights/LoFTR_teacher.onnx'
* `--original` - If specified the original LoFTR model will be used, can be used only with `--weights` parameter
* `--camid` - OpenCV webcam video capture ID, usually 0 or 1, default 0
* `--device` - Selects the runtime back-end CPU or CUDA, default is CUDA

Sample command line:
```
pixi run python webcam.py --trt=weights/LoFTR_teacher.engine --camid=0
```

Demo application shows a window with pair of images captured with a camera. Initially there will be the two same images. Then you can choose a view of interest and press the `s` button, the view will be remembered and will be visible as the left image. Then you can change the view and press the `p` button to make a snapshot of the feature matching result, the corresponding features will be marked with the same numbers at the two images. If you press the `p` button again then application will allow you to change the view and repeat the feature matching process. Also this application shows the real-time FPS counter so you can estimate the model performance.

### Training

To repeat the training procedure you should use the low-res set of the [BlendedMVS](https://github.com/YoYo000/BlendedMVS) dataset. After download you can use the `train.py` script to run training process. There are following parameters for this script:
* `--path` - Path to the dataset
* `--checkpoint_path` - Where to store a log information and checkpoints, default value is 'weights'
* `--weights` - Path to the LoFTR teacher model weights, default value is 'weights/outdoor_ds.ckpt'
                        
Sample command line:
```
python3 train.py --path=/home/user/datasets/BlendedMVS --checkpoint_path=weights/experiment1/
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
