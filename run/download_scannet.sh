#!/usr/bin/env bash
# Downloads the ScanNet-1500 test set (the 1500 indoor pairs LoFTR and EfficientLoFTR are evaluated on) from the
# LoFTR Google Drive folder to $DATASET_DIR/scannet (DATASET_DIR is set in env.sh).
# Run inside the env: pixi run download-scannet
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

FILE_ID=1wtl-mNicxGlXZ-UQJxFnKuWPvvssQBwd  # testdata/scannet_test_1500.tar
TARGET="$DATASET_DIR/scannet"

if [ -d "$TARGET" ]; then
  echo "ScanNet-1500 already present: $TARGET"
  exit 0
fi

mkdir -p "$DATASET_DIR"
ARCHIVE="$DATASET_DIR/scannet.tar"
gdown --continue -O "$ARCHIVE" "$FILE_ID"

# extract next to the target and move it into place, so that an interrupted run leaves no partial dataset
TMP_DIR=$(mktemp -d "$DATASET_DIR/.scannet.XXXXXX")
trap 'rm -rf "$TMP_DIR"' EXIT
tar -xf "$ARCHIVE" -C "$TMP_DIR"
mv "$TMP_DIR/scannet_test_1500" "$TARGET"  # top-level folder of the archive
rm "$ARCHIVE"
echo "Saved $TARGET"
