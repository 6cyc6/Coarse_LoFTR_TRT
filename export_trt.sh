#!/usr/bin/env bash
# Builds the legacy Coarse LoFTR engine with the TensorRT version of the pixi environment
# (engines only load with the TensorRT version that built them). Run inside the env: pixi run build-legacy

ONNX_MODEL=weights/LoFTR_teacher.onnx
TRT_MODEL=weights/LoFTR_teacher.engine

python build_trt_engine.py --onnx=$ONNX_MODEL --engine=$TRT_MODEL --fp16 --workspace-gib=8 "$@"
