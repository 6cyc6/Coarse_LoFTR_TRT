#!/usr/bin/env bash
# Builds the TensorRT plugin library of SLiM's selective scan (slim/plugin) with the CUDA toolkit in CUDA_HOME (set in
# pixi.toml) and the public headers of the installed TensorRT version, fetched from github.com/NVIDIA/TensorRT.
# Run inside the env: pixi run build-slim-plugin
set -euo pipefail

KERNELS=third_party/SLiM/src/backbone/vssm/kernels/selective_scan/csrc/selective_scan
OUT=build/slim_plugin/libslim_trt_plugins.so
TRT_VERSION=$(python -c "import tensorrt; print(tensorrt.__version__.rsplit('.', 1)[0])")
TRT_INCLUDE=build/tensorrt-$TRT_VERSION/include
# Ampere, Jetson Orin, Ada; PTX for newer GPUs
ARCHS=${SLIM_PLUGIN_ARCHS:-"80 86 87 89"}

if [ ! -f "$TRT_INCLUDE/NvInferRuntime.h" ]; then
  mkdir -p "$TRT_INCLUDE"
  for header in NvInfer.h NvInferImpl.h NvInferLegacyDims.h NvInferPlugin.h NvInferPluginBase.h NvInferPluginUtils.h \
                NvInferRuntime.h NvInferRuntimeBase.h NvInferRuntimeCommon.h NvInferRuntimePlugin.h NvInferVersion.h; do
    curl -sfL -o "$TRT_INCLUDE/$header" "https://raw.githubusercontent.com/NVIDIA/TensorRT/v$TRT_VERSION/include/$header"
  done
fi

GENCODE=()
for arch in $ARCHS; do
  GENCODE+=(-gencode "arch=compute_$arch,code=sm_$arch")
done
last=${ARCHS##* }
GENCODE+=(-gencode "arch=compute_$last,code=compute_$last")

mkdir -p "$(dirname "$OUT")"
# the kernel flags of the upstream setup.py
"$CUDA_HOME/bin/nvcc" -shared -Xcompiler -fPIC,-Wno-deprecated-declarations -cudart static -O3 -std=c++17 \
  --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math --threads 4 "${GENCODE[@]}" \
  -I slim/plugin/shim -I "$TRT_INCLUDE" -I "$KERNELS" -I "$KERNELS/cusoflex" \
  slim/plugin/selective_scan_plugin.cu -o "$OUT"
echo "Built $OUT (TensorRT $TRT_VERSION headers, sm_${ARCHS// /, sm_})"
