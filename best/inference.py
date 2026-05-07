"""HW3 instance segmentation -- inference / submission builder.

Mirrors train.py: PointRend mask head + DCNv2 backbone config flags must
match what was used at training time so the checkpoint loads cleanly.

Multi-scale + horizontal-flip TTA. Detections are fused per class with
Soft-NMS (Bodla et al. ICCV 2017) instead of hard NMS, because cells
overlap heavily (one image holds up to ~770 instances) and hard NMS at
IoU>0.5 erases real overlapping cells. Run with:

    python inference.py
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import tifffile
import torch
import torchvision.ops as tv_ops

from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from pycocotools import mask as coco_mask
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
EXP_NAME = "exp_d2_maskrcnn_x101_dcn_dice_pointrend"
CKPT_PATH = HERE.parent / "runs" / EXP_NAME / "model_final.pth"
TEST_DIR = HERE.parent / "hw3-data-release" / "test_release"
ID_MAP_PATH = (
    HERE.parent / "hw3-data-release" / "test_image_name_to_ids.json"
)
OUTPUT_DIR = HERE.parent / "runs" / EXP_NAME / "submit"

NUM_CLASSES = 4
BASE_YAML_PLAIN = (
    "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
)
POINTREND_PROJECT_DIR = (
    HERE.parent / "detectron2" / "projects" / "PointRend"
)
POINTREND_BASE_YAML = (
    POINTREND_PROJECT_DIR / "configs" / "InstanceSegmentation"
    / "pointrend_rcnn_X_101_32x8d_FPN_3x_coco.yaml"
)

# Architecture flags MUST match train.py at the time the ckpt was saved.
USE_DCN = True
USE_POINTREND = True
DEFORM_ON_PER_STAGE = [False, True, True, True]
DEFORM_MODULATED = True
DEFORM_NUM_GROUPS = 1

ANCHOR_SIZES = [[8], [16], [32], [64], [128]]
ANCHOR_RATIOS = [[0.5, 1.0, 2.0]] * 5
RPN_PRE_NMS_TEST = 2000
RPN_POST_NMS_TEST = 2000
RPN_NMS_THRESH = 0.7
ROI_NMS_THRESH = 0.5
DETECTIONS_PER_IMG = 1000
SCORE_THRESH_TEST = 0.00

INPUT_MIN_SIZE_TEST = 1024
INPUT_MAX_SIZE_TEST = 1333

USE_HFLIP_TTA = True
TTA_SCALES = (768, 1024, 1280)

# Final detection fusion across TTA outputs.
USE_SOFT_NMS = True
FINAL_NMS_THR = 0.5
SOFT_NMS_METHOD = "gaussian"     # "linear" | "gaussian"
SOFT_NMS_SIGMA = 0.5
SOFT_NMS_LINEAR_THR = 0.3
SOFT_NMS_SCORE_THR = 0.001
SCORE_THR_OUT = 0.00


# ---------------------------------------------------------------------------
# PointRend import (registers ROI/mask heads as side-effect)
# ---------------------------------------------------------------------------

_point_rend_module = None


def _import_pointrend():
    global _point_rend_module
    if _point_rend_module is not None:
        return _point_rend_module
    if not POINTREND_PROJECT_DIR.exists():
        raise FileNotFoundError(
            f"PointRend project not found at {POINTREND_PROJECT_DIR}."
        )
    sys.path.insert(0, str(POINTREND_PROJECT_DIR))
    import point_rend  # noqa: F401  (registers heads)
    _point_rend_module = point_rend
    return point_rend


# ---------------------------------------------------------------------------
# Cfg builder (must match train.py architecture)
# ---------------------------------------------------------------------------

def build_cfg():
    cfg = get_cfg()

    if USE_POINTREND:
        point_rend = _import_pointrend()
        point_rend.add_pointrend_config(cfg)
        if not POINTREND_BASE_YAML.exists():
            raise FileNotFoundError(
                f"PointRend X-101 YAML not found at "
                f"{POINTREND_BASE_YAML}."
            )
        cfg.merge_from_file(str(POINTREND_BASE_YAML))
    else:
        cfg.merge_from_file(model_zoo.get_config_file(BASE_YAML_PLAIN))

    cfg.MODEL.WEIGHTS = ""
    cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675]
    cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0]

    if USE_DCN:
        cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE = DEFORM_ON_PER_STAGE
        cfg.MODEL.RESNETS.DEFORM_MODULATED = DEFORM_MODULATED
        cfg.MODEL.RESNETS.DEFORM_NUM_GROUPS = DEFORM_NUM_GROUPS

    if USE_POINTREND:
        cfg.MODEL.ROI_HEADS.NAME = "PointRendROIHeads"
        cfg.MODEL.POINT_HEAD.NUM_CLASSES = NUM_CLASSES

    cfg.MODEL.ANCHOR_GENERATOR.SIZES = ANCHOR_SIZES
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = ANCHOR_RATIOS

    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = RPN_PRE_NMS_TEST
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = RPN_POST_NMS_TEST
    cfg.MODEL.RPN.NMS_THRESH = RPN_NMS_THRESH

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESH_TEST
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = ROI_NMS_THRESH
    cfg.TEST.DETECTIONS_PER_IMAGE = DETECTIONS_PER_IMG
    cfg.INPUT.MASK_FORMAT = "bitmask"

    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def load_image_bgr(path):
    img = tifffile.imread(str(path))
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img[:, :, ::-1].copy()


def resize_bgr(img_bgr, short_edge, max_edge):
    h, w = img_bgr.shape[:2]
    scale = short_edge / min(h, w)
    if max(h, w) * scale > max_edge:
        scale = max_edge / max(h, w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    out = cv2.resize(
        img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR,
    )
    return out, scale


def encode_rle_str(mask_bool):
    rle = coco_mask.encode(np.asfortranarray(mask_bool.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


# ---------------------------------------------------------------------------
# Soft-NMS (Bodla et al. ICCV 2017)
# ---------------------------------------------------------------------------

def _box_iou_xyxy(boxes_a, boxes_b):
    inter_x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    inter_y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    inter_x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    inter_y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    inter_w = np.clip(inter_x2 - inter_x1, 0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0, None)
    inter = inter_w * inter_h
    area_a = (
        (boxes_a[:, 2] - boxes_a[:, 0])
        * (boxes_a[:, 3] - boxes_a[:, 1])
    )
    area_b = (
        (boxes_b[:, 2] - boxes_b[:, 0])
        * (boxes_b[:, 3] - boxes_b[:, 1])
    )
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def soft_nms_one_class(
    boxes, scores, method, sigma, linear_thr, score_thr,
):
    """Per-class Soft-NMS; returns (kept_indices, decayed_scores)."""
    n = boxes.shape[0]
    if n == 0:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
        )
    idx = np.arange(n)
    boxes_w = boxes.copy()
    scores_w = scores.astype(np.float32).copy()

    keep_orig_idx = []
    keep_scores = []

    for _ in range(n):
        m = int(np.argmax(scores_w))
        ms = float(scores_w[m])
        if ms < score_thr:
            break
        keep_orig_idx.append(int(idx[m]))
        keep_scores.append(ms)

        if scores_w.shape[0] == 1:
            break
        rest_mask = np.ones_like(scores_w, dtype=bool)
        rest_mask[m] = False
        rest_boxes = boxes_w[rest_mask]
        rest_scores = scores_w[rest_mask]
        rest_idx = idx[rest_mask]

        ious = _box_iou_xyxy(boxes_w[m:m + 1], rest_boxes)[0]

        if method == "linear":
            decay = np.where(ious >= linear_thr, 1.0 - ious, 1.0)
        elif method == "gaussian":
            decay = np.exp(-(ious ** 2) / max(sigma, 1e-6))
        else:
            raise ValueError(f"unknown soft-nms method: {method}")

        rest_scores = rest_scores * decay.astype(np.float32)

        survive = rest_scores > score_thr
        boxes_w = rest_boxes[survive]
        scores_w = rest_scores[survive]
        idx = rest_idx[survive]
        if scores_w.shape[0] == 0:
            break

    return (
        np.asarray(keep_orig_idx, dtype=np.int64),
        np.asarray(keep_scores, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Per-image prediction (multi-scale + hflip TTA)
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_one_scale(model, img_bgr_resized, orig_h, orig_w, hflip=False):
    img_t = torch.as_tensor(
        img_bgr_resized.transpose(2, 0, 1).copy()
    )
    if hflip:
        img_t = torch.flip(img_t, dims=[2])
    inputs = [{"image": img_t, "height": orig_h, "width": orig_w}]
    outputs = model(inputs)[0]["instances"].to("cpu")

    boxes = outputs.pred_boxes.tensor.numpy()
    scores = outputs.scores.numpy()
    labels = outputs.pred_classes.numpy()
    masks = outputs.pred_masks.numpy().astype(bool)

    if hflip:
        masks = masks[:, :, ::-1].copy()
        flipped = boxes.copy()
        flipped[:, 0] = orig_w - boxes[:, 2]
        flipped[:, 2] = orig_w - boxes[:, 0]
        boxes = flipped

    return boxes, scores, labels, masks


def per_class_nms(boxes, scores, labels, masks, iou_thr):
    keep_idx = []
    for c in np.unique(labels):
        mask_c = labels == c
        idx = np.where(mask_c)[0]
        if idx.size == 0:
            continue
        b = torch.as_tensor(boxes[idx], dtype=torch.float32)
        s = torch.as_tensor(scores[idx], dtype=torch.float32)
        kept = tv_ops.nms(b, s, iou_thr).numpy().tolist()
        keep_idx.extend(idx[kept].tolist())
    keep_idx = sorted(keep_idx)
    return (
        boxes[keep_idx],
        scores[keep_idx],
        labels[keep_idx],
        masks[keep_idx],
    )


def per_class_soft_nms(
    boxes, scores, labels, masks,
    method, sigma, linear_thr, score_thr,
):
    out_idx = []
    out_scores = []
    for c in np.unique(labels):
        mask_c = labels == c
        idx_c = np.where(mask_c)[0]
        if idx_c.size == 0:
            continue
        kept_local, decayed = soft_nms_one_class(
            boxes[idx_c], scores[idx_c],
            method=method, sigma=sigma,
            linear_thr=linear_thr, score_thr=score_thr,
        )
        if kept_local.size == 0:
            continue
        out_idx.extend(idx_c[kept_local].tolist())
        out_scores.extend(decayed.tolist())

    if not out_idx:
        return boxes[:0], scores[:0], labels[:0], masks[:0]
    out_idx = np.asarray(out_idx, dtype=np.int64)
    out_scores = np.asarray(out_scores, dtype=np.float32)
    return (
        boxes[out_idx],
        out_scores,
        labels[out_idx],
        masks[out_idx],
    )


@torch.no_grad()
def predict_tta(model, img_bgr):
    H, W = img_bgr.shape[:2]
    all_boxes, all_scores, all_labels, all_masks = [], [], [], []

    for short in TTA_SCALES:
        img_r, _ = resize_bgr(img_bgr, short, INPUT_MAX_SIZE_TEST)
        b, s, lab, m = predict_one_scale(model, img_r, H, W, hflip=False)
        all_boxes.append(b)
        all_scores.append(s)
        all_labels.append(lab)
        all_masks.append(m)
        if USE_HFLIP_TTA:
            b, s, lab, m = predict_one_scale(
                model, img_r, H, W, hflip=True,
            )
            all_boxes.append(b)
            all_scores.append(s)
            all_labels.append(lab)
            all_masks.append(m)

    if not all_boxes:
        return (
            np.zeros((0, 4)),
            np.zeros((0,)),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0, H, W), dtype=bool),
        )

    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    masks = np.concatenate(all_masks, axis=0)

    if USE_SOFT_NMS:
        boxes, scores, labels, masks = per_class_soft_nms(
            boxes, scores, labels, masks,
            method=SOFT_NMS_METHOD,
            sigma=SOFT_NMS_SIGMA,
            linear_thr=SOFT_NMS_LINEAR_THR,
            score_thr=SOFT_NMS_SCORE_THR,
        )
    else:
        boxes, scores, labels, masks = per_class_nms(
            boxes, scores, labels, masks, FINAL_NMS_THR,
        )
    return boxes, scores, labels, masks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if USE_POINTREND:
        _import_pointrend()

    cfg = build_cfg()
    cfg.MODEL.WEIGHTS = str(CKPT_PATH)
    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).load(str(CKPT_PATH))
    print(f"loaded {CKPT_PATH}")
    print(
        f"USE_POINTREND={USE_POINTREND}  USE_DCN={USE_DCN}  "
        f"USE_SOFT_NMS={USE_SOFT_NMS}  method={SOFT_NMS_METHOD}"
    )

    with open(ID_MAP_PATH) as f:
        id_map = {item["file_name"]: item["id"] for item in json.load(f)}
    test_files = sorted(TEST_DIR.glob("*.tif"))
    print(f"test images: {len(test_files)}")

    records = []
    for path in tqdm(test_files, desc="infer", dynamic_ncols=True):
        img_bgr = load_image_bgr(path)
        image_id = id_map.get(path.name)
        if image_id is None:
            print(f"[warn] {path.name} not in id map, skipping")
            continue

        boxes, scores, labels, masks = predict_tta(model, img_bgr)
        for i in range(boxes.shape[0]):
            if scores[i] < SCORE_THR_OUT:
                continue
            m = masks[i]
            if not m.any():
                continue
            x1, y1, x2, y2 = boxes[i].tolist()
            records.append({
                "image_id": int(image_id),
                "category_id": int(labels[i]) + 1,
                "bbox": [
                    float(x1),
                    float(y1),
                    float(x2 - x1),
                    float(y2 - y1),
                ],
                "score": float(scores[i]),
                "segmentation": encode_rle_str(m),
            })

    out_json = OUTPUT_DIR / "test-results.json"
    with open(out_json, "w") as f:
        json.dump(records, f)
    print(f"wrote {out_json}  ({len(records)} predictions)")

    zip_path = OUTPUT_DIR / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(out_json, arcname="test-results.json")
    print(f"wrote {zip_path}")


if __name__ == "__main__":
    main()
