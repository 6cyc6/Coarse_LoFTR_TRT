#!/usr/bin/env bash
# Project environment variables, load with: source run/env.sh

# Root directory of the datasets (ScanNet, ...). A DATASET_DIR already set in the shell takes precedence.
export DATASET_DIR="${DATASET_DIR:-$HOME/dataset}"
