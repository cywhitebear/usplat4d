#!/usr/bin/env bash
# prepare_for_rendering.sh — prep a USplat4D output dir for run_rendering.py
#
# Usage:
#   ./prepare_for_rendering.sh <usplat4d_out_dir> <som_dir>
#
# Example:
#   ./prepare_for_rendering.sh \
#     /media/ee904/DATA1/Yun/Outputs/usplat4d/iphone/backpack \
#     /media/ee904/DATA1/Yun/Outputs/shape-of-motion/iphone/backpack

set -e

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <usplat4d_out_dir> <som_dir>"
    exit 1
fi

OUT="$1"
SOM="$2"
CKPT="$OUT/usplat4d/final.ckpt"
SOM_CFG="$SOM/cfg.yaml"

if [[ ! -f "$CKPT" ]]; then
    echo "ERROR: final checkpoint not found at $CKPT"
    exit 1
fi
if [[ ! -f "$SOM_CFG" ]]; then
    echo "ERROR: SoM cfg.yaml not found at $SOM_CFG"
    exit 1
fi

mkdir -p "$OUT/checkpoints"
ln -sf "$CKPT" "$OUT/checkpoints/last.ckpt"
cp "$SOM_CFG" "$OUT/cfg.yaml"

echo "Ready. Run:"
echo "  cd /home/ee904/Yun/shape-of-motion"
echo "  python run_rendering.py --work-dir $OUT --port 8890"
