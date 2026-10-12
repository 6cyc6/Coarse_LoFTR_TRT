#!/usr/bin/env bash
# Builds VMamba's selective scan CUDA extension (selective_scan_cuda_oflex) that SLiM's torch model uses, from the
# sources in third_party/SLiM, with the CUDA toolkit in CUDA_HOME (set in pixi.toml). Run inside the env:
# pixi run build-slim-kernels
set -euo pipefail

SRC=third_party/SLiM/src/backbone/vssm/kernels/selective_scan
BUILD=build/selective_scan

# build from a copy: setup.py writes build/ and egg-info next to its sources, which would dirty the submodule
rm -rf "$BUILD"
mkdir -p "$BUILD"
cp -r "$SRC/setup.py" "$SRC/csrc" "$BUILD/"
python -m pip install --no-build-isolation --no-deps "$BUILD"
python -c "import torch, selective_scan_cuda_oflex; print('Built', selective_scan_cuda_oflex.__file__)"
