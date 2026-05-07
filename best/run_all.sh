#!/usr/bin/env bash
# Run the full HW3 code9 pipeline (code8 + PointRend mask head + Soft-NMS):
#   ensure COCO-pretrained PointRend Mask R-CNN X-101 weights are in
#   ./pretrained/ -> train -> inference -> visualisation.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# Pin to a single GPU. Adjust if you have a multi-GPU box.
export CUDA_VISIBLE_DEVICES=3

# ---------------------------------------------------------------------------
# 0. Ensure pretrained weights exist
# ---------------------------------------------------------------------------
# code9 defaults to USE_POINTREND=True, which needs the PointRend X-101
# checkpoint. The plain Mask R-CNN X-101 checkpoint is NOT a valid
# substitute (mask-head keys don't match), so we resolve the PointRend pkl
# specifically. If you flip USE_POINTREND=False in train.py, swap the
# WEIGHTS_FILE_NAME / WEIGHTS_URL block back to the code8 pair.
WEIGHTS_DIR="$HERE/pretrained"
WEIGHTS_FILE_NAME="pointrend_rcnn_X_101_32x8d_FPN_3x_coco.pkl"
WEIGHTS_FILE="$WEIGHTS_DIR/$WEIGHTS_FILE_NAME"
WEIGHTS_URL="https://dl.fbaipublicfiles.com/detectron2/PointRend/InstanceSegmentation/pointrend_rcnn_X_101_32x8d_FPN_3x_coco/28119989/model_final_ba17b9.pkl"
CODE8_WEIGHTS="$HERE/../code8/pretrained/$WEIGHTS_FILE_NAME"
CODE7_WEIGHTS="$HERE/../code7/pretrained/$WEIGHTS_FILE_NAME"

mkdir -p "$WEIGHTS_DIR"
if [ ! -f "$WEIGHTS_FILE" ]; then
    if [ -f "$CODE8_WEIGHTS" ]; then
        echo "[0/3] copying pretrained weights from $CODE8_WEIGHTS ..."
        cp "$CODE8_WEIGHTS" "$WEIGHTS_FILE"
    elif [ -f "$CODE7_WEIGHTS" ]; then
        echo "[0/3] copying pretrained weights from $CODE7_WEIGHTS ..."
        cp "$CODE7_WEIGHTS" "$WEIGHTS_FILE"
    else
        echo "[0/3] downloading pretrained weights to $WEIGHTS_FILE ..."
        if [ -x /usr/bin/curl ]; then
            /usr/bin/curl -L --fail --progress-bar -o "$WEIGHTS_FILE" "$WEIGHTS_URL"
        elif command -v wget >/dev/null 2>&1; then
            wget --show-progress -O "$WEIGHTS_FILE" "$WEIGHTS_URL"
        else
            echo "ERROR: neither curl nor wget found; install one or download manually:"
            echo "  $WEIGHTS_URL"
            echo "  -> $WEIGHTS_FILE"
            exit 1
        fi
    fi
else
    echo "[0/3] pretrained weights already at $WEIGHTS_FILE, skipping."
fi

LOG_FILE="$HERE/train_error.log"
STDERR_TMP=$(mktemp)

echo "[1/4] training (DCNv2 + PointRend) ..."
python train.py 2> >(tee "$STDERR_TMP" >&2)
TRAIN_EXIT=${PIPESTATUS[0]}
if [ "$TRAIN_EXIT" -ne 0 ]; then
    awk '/^Traceback/{found=1} found{print}' "$STDERR_TMP" > "$LOG_FILE"
    echo "ERROR: train.py failed. Traceback saved to $LOG_FILE"
    rm -f "$STDERR_TMP"
    exit 1
fi
rm -f "$STDERR_TMP" "$LOG_FILE"

echo "[2/4] inference (Soft-NMS + multi-scale TTA) ..."
python inference.py

echo "[3/4] visualisation (random subset of test images) ..."
python visualize.py

echo "[4/4] report figures (val PR/AP/loss/error-cases) ..."
python plots.py

echo "done."
