"""HW3 instance segmentation -- visualization (code9).

Single-file. Run:

    python visualize.py

Same visual format as code8/visualize.py; only EXP_NAME changes so
side-by-side PNGs land under runs/exp_d2_maskrcnn_x101_dcn_dice_pointrend/viz/.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np
import tifffile
from pycocotools import mask as coco_mask
from tqdm import tqdm


# ============================================================================
# CONFIG
# ============================================================================

HERE = Path(__file__).resolve().parent
EXP_NAME = "exp_d2_maskrcnn_x101_dcn_dice_pointrend"
RESULTS_JSON = HERE.parent / "runs" / EXP_NAME / "submit" / "test-results.json"
TEST_DIR = HERE.parent / "hw3-data-release" / "test_release"
ID_MAP_PATH = HERE.parent / "hw3-data-release" / "test_image_name_to_ids.json"
OUTPUT_DIR = HERE.parent / "runs" / EXP_NAME / "viz"

SCORE_THR = 0.3
ALPHA = 0.5
DRAW_BOXES = True
DRAW_LABELS = True
MAX_IMAGES = 12          # set to None to render every test image
SEED = 42
CLASS_NAMES = ["class1", "class2", "class3", "class4"]
CLASS_COLORS = [
    (255, 64, 64),
    (64, 255, 64),
    (64, 128, 255),
    (255, 200, 0),
]


# ============================================================================
# HELPERS
# ============================================================================

def load_image_rgb(path):
    img = tifffile.imread(str(path))
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def decode_rle(seg, h, w):
    if isinstance(seg.get("counts"), str):
        seg = {"size": seg["size"], "counts": seg["counts"].encode("ascii")}
    return coco_mask.decode(seg).astype(bool)


def random_color(rng):
    h = rng.random()
    s = 0.6 + rng.random() * 0.3
    v = 0.85 + rng.random() * 0.15
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    i %= 6
    rgb = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i]
    return tuple(int(c * 255) for c in rgb)


# ============================================================================
# MAIN
# ============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    with open(RESULTS_JSON) as f:
        records = json.load(f)
    with open(ID_MAP_PATH) as f:
        id_map = {item["id"]: item["file_name"] for item in json.load(f)}

    by_image = {}
    for r in records:
        by_image.setdefault(r["image_id"], []).append(r)

    image_ids = sorted(by_image.keys())
    if MAX_IMAGES is not None and MAX_IMAGES < len(image_ids):
        image_ids = rng.sample(image_ids, MAX_IMAGES)
        image_ids.sort()

    for image_id in tqdm(image_ids, desc="viz", dynamic_ncols=True):
        fname = id_map.get(image_id)
        if fname is None:
            continue
        img_path = TEST_DIR / fname
        if not img_path.exists():
            continue
        img = load_image_rgb(img_path)
        h, w = img.shape[:2]
        overlay = img.copy()
        outline_layer = img.copy()

        preds = sorted(
            by_image[image_id], key=lambda r: r["score"], reverse=True,
        )

        for r in preds:
            if r["score"] < SCORE_THR:
                continue
            cat = r["category_id"] - 1
            mask = decode_rle(r["segmentation"], h, w)
            if not mask.any():
                continue
            colour = random_color(rng)
            for c in range(3):
                overlay[..., c] = np.where(
                    mask,
                    overlay[..., c] * (1 - ALPHA) + colour[c] * ALPHA,
                    overlay[..., c],
                )
            cls_colour = CLASS_COLORS[cat % len(CLASS_COLORS)]
            contours, _ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(outline_layer, contours, -1, cls_colour[::-1], 1)

            if DRAW_BOXES:
                x, y, bw, bh = r["bbox"]
                x2, y2 = x + bw, y + bh
                cv2.rectangle(
                    outline_layer, (int(x), int(y)), (int(x2), int(y2)),
                    cls_colour[::-1], 1,
                )
            if DRAW_LABELS:
                label = f"{CLASS_NAMES[cat]} {r['score']:.2f}"
                x, y = int(r["bbox"][0]), int(r["bbox"][1])
                cv2.putText(
                    outline_layer, label, (x, max(10, y - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    cls_colour[::-1], 1, cv2.LINE_AA,
                )

        merged = (overlay.astype(np.float32) * 0.7
                  + outline_layer.astype(np.float32) * 0.3)
        merged = np.clip(merged, 0, 255).astype(np.uint8)

        side = np.concatenate([img, merged], axis=1)
        out_path = OUTPUT_DIR / f"{Path(fname).stem}.png"
        cv2.imwrite(str(out_path), side[:, :, ::-1])

    print(f"wrote {len(image_ids)} images to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
