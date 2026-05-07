"""HW3 instance segmentation -- diagnostic figures for the report.

Builds a COCO-format GT json for the val split, runs single-scale
inference, runs pycocotools COCOeval, and renders 11 figures plus a
summary.json. Figures land under runs/<exp>/figures/. Run with:

    python plots.py

Inference here is single-scale (no TTA, no Soft-NMS) -- this script is
for diagnostic plots, not the leaderboard submission.
"""
from __future__ import annotations

import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile
import torch
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

# Matplotlib backend must be set before pyplot import (headless safe).
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import cm  # noqa: E402


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
EXP_NAME = "exp_d2_maskrcnn_x101_dcn_dice_pointrend"
RUN_DIR = HERE.parent / "runs" / EXP_NAME
DATA_ROOT = HERE.parent / "hw3-data-release"
TRAIN_DIR = DATA_ROOT / "train"

SPLIT_PATH = RUN_DIR / "split.json"
CKPT_PATH = RUN_DIR / "model_final.pth"
METRICS_PATH = RUN_DIR / "metrics.json"
PLOT_DIR = RUN_DIR / "figures"

VAL_GT_JSON = RUN_DIR / "val_gt.json"
VAL_PRED_JSON = RUN_DIR / "val_predictions.json"

NUM_CLASSES = 4
CLASS_NAMES = [f"class{i + 1}" for i in range(NUM_CLASSES)]
CLASS_FILES = [f"class{i}.tif" for i in range(1, 5)]

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
_CMAP = cm.get_cmap("tab10")

INFER_SHORT = 1024
INFER_MAX = 1333
SCORE_THRESH_INFER = 0.05
DETECTIONS_PER_IMG = 1000

SEED = 42


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _needs_refresh(out_path: Path, src_path: Path) -> bool:
    if not out_path.exists():
        return True
    if not src_path.exists():
        return False
    return out_path.stat().st_mtime < src_path.stat().st_mtime


def load_image_rgb(path: Path) -> np.ndarray:
    img = tifffile.imread(str(path))
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def encode_rle_str(m: np.ndarray) -> dict:
    rle = coco_mask.encode(np.asfortranarray(m.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def _ensure_bytes_rle(seg: dict) -> dict:
    if isinstance(seg.get("counts"), str):
        return {
            "size": seg["size"],
            "counts": seg["counts"].encode("ascii"),
        }
    return seg


def decode_rle(seg: dict) -> np.ndarray:
    return coco_mask.decode(_ensure_bytes_rle(seg)).astype(bool)


def class_color_rgb(cls_idx_0based: int) -> tuple:
    rgb = _CMAP(cls_idx_0based % 10)[:3]
    return tuple(int(c * 255) for c in rgb)


# ---------------------------------------------------------------------------
# Build COCO-format val GT json
# ---------------------------------------------------------------------------

def build_val_gt_json():
    if VAL_GT_JSON.exists():
        print(f"[1/3] using cached {VAL_GT_JSON.name}")
        return

    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"{SPLIT_PATH} not found. Train first "
            "(it writes split.json)."
        )
    with open(SPLIT_PATH) as f:
        val_ids = json.load(f).get("val", [])
    if not val_ids:
        raise RuntimeError(
            "split.json has no 'val' ids -- did you train with "
            "TRAIN_ON_ALL=True? Plots need a held-out validation split."
        )

    images, annotations = [], []
    ann_id = 0
    for img_idx, sid in enumerate(val_ids):
        d = TRAIN_DIR / sid
        ip = d / "image.tif"
        if not ip.exists():
            continue
        img = tifffile.imread(str(ip))
        h, w = (img.shape[:2] if img.ndim >= 2 else img.shape)
        images.append({
            "id": img_idx,
            "file_name": f"{sid}/image.tif",
            "_sample_id": sid,
            "height": int(h),
            "width": int(w),
        })
        for cls_idx, fname in enumerate(CLASS_FILES):
            f = d / fname
            if not f.exists():
                continue
            raw = tifffile.imread(str(f)).astype(np.int32)
            for inst_id in np.unique(raw):
                if inst_id == 0:
                    continue
                m = (raw == inst_id)
                if m.sum() < 4:
                    continue
                rle = encode_rle_str(m)
                ys, xs = np.where(m)
                x1, y1 = int(xs.min()), int(ys.min())
                x2, y2 = int(xs.max() + 1), int(ys.max() + 1)
                annotations.append({
                    "id": ann_id,
                    "image_id": img_idx,
                    "category_id": cls_idx + 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": int(m.sum()),
                    "iscrowd": 0,
                    "segmentation": rle,
                })
                ann_id += 1

    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": i + 1, "name": CLASS_NAMES[i]}
            for i in range(NUM_CLASSES)
        ],
    }
    with open(VAL_GT_JSON, "w") as f:
        json.dump(coco_dict, f)
    print(
        f"[1/3] wrote {VAL_GT_JSON.name}  "
        f"({len(images)} images, {len(annotations)} annotations)"
    )


# ---------------------------------------------------------------------------
# Single-scale val inference
# ---------------------------------------------------------------------------

def run_val_inference():
    if not _needs_refresh(VAL_PRED_JSON, CKPT_PATH):
        print(f"[2/3] using cached {VAL_PRED_JSON.name}")
        return

    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"checkpoint not found: {CKPT_PATH}")

    sys.path.insert(0, str(HERE))
    import inference as inf  # noqa: WPS433

    if inf.USE_POINTREND:
        inf._import_pointrend()

    cfg = inf.build_cfg()
    cfg.MODEL.WEIGHTS = str(CKPT_PATH)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESH_INFER
    cfg.TEST.DETECTIONS_PER_IMAGE = DETECTIONS_PER_IMG

    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.modeling import build_model

    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).load(str(CKPT_PATH))
    print(f"[2/3] loaded {CKPT_PATH.name}")

    coco_gt = COCO(str(VAL_GT_JSON))
    img_records = coco_gt.loadImgs(coco_gt.getImgIds())

    records = []
    for info in tqdm(img_records, desc="val infer", dynamic_ncols=True):
        img_path = TRAIN_DIR / info["file_name"]
        img_bgr = inf.load_image_bgr(img_path)
        H, W = img_bgr.shape[:2]
        img_r, _ = inf.resize_bgr(img_bgr, INFER_SHORT, INFER_MAX)
        boxes, scores, labels, masks = inf.predict_one_scale(
            model, img_r, H, W, hflip=False,
        )
        for i in range(boxes.shape[0]):
            m = masks[i]
            if not m.any():
                continue
            x1, y1, x2, y2 = boxes[i].tolist()
            records.append({
                "image_id": int(info["id"]),
                "category_id": int(labels[i]) + 1,
                "bbox": [
                    float(x1), float(y1),
                    float(x2 - x1), float(y2 - y1),
                ],
                "score": float(scores[i]),
                "segmentation": encode_rle_str(m),
            })

    with open(VAL_PRED_JSON, "w") as f:
        json.dump(records, f)
    print(
        f"[2/3] wrote {VAL_PRED_JSON.name}  ({len(records)} predictions)"
    )


# ---------------------------------------------------------------------------
# COCOeval and per-prediction matches
# ---------------------------------------------------------------------------

def run_cocoeval(coco_gt: COCO):
    coco_dt = coco_gt.loadRes(str(VAL_PRED_JSON))
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="segm")
    coco_eval.params.maxDets = [100, 500, 1000]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval


def _mask_iou(pred_seg, gt_segs):
    if not gt_segs:
        return np.zeros(0, dtype=np.float32)
    pred_b = _ensure_bytes_rle(pred_seg)
    gt_b = [_ensure_bytes_rle(g) for g in gt_segs]
    iscrowd = [0] * len(gt_b)
    return (
        coco_mask.iou([pred_b], gt_b, iscrowd)
        .flatten().astype(np.float32)
    )


def match_predictions_mask(
    coco_gt: COCO, results, iou_thresholds=IOU_THRESHOLDS,
):
    """Greedy per-image mask-IoU matching at every IoU threshold.

    pred_class / gt_class_any are 1-indexed (matching COCO category_id).
    """
    iou_thresholds = np.asarray(iou_thresholds, dtype=np.float32)
    T = len(iou_thresholds)

    preds_by_img = defaultdict(list)
    for r in results:
        preds_by_img[r["image_id"]].append(r)

    out = []
    for img_id, preds in preds_by_img.items():
        preds = sorted(preds, key=lambda x: -x["score"])
        gts = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))

        gt_segs = [g["segmentation"] for g in gts]
        gt_classes = np.array([g["category_id"] for g in gts])

        matched_same_t = np.zeros((T, len(gts)), dtype=bool)
        matched_any = np.zeros(len(gts), dtype=bool)

        for p in preds:
            pc = int(p["category_id"])
            is_tp = np.zeros(T, dtype=bool)
            gt_any = None

            if len(gts) > 0:
                ious = _mask_iou(p["segmentation"], gt_segs)

                class_mask = (gt_classes == pc)
                for ti, th in enumerate(iou_thresholds):
                    avail = class_mask & (~matched_same_t[ti])
                    if avail.any():
                        cand = np.where(avail, ious, -1.0)
                        bi = int(cand.argmax())
                        if cand[bi] >= th:
                            is_tp[ti] = True
                            matched_same_t[ti, bi] = True

                avail_any = ~matched_any
                if avail_any.any():
                    cand_any = np.where(avail_any, ious, -1.0)
                    bi2 = int(cand_any.argmax())
                    if cand_any[bi2] >= 0.5:
                        matched_any[bi2] = True
                        gt_any = int(gt_classes[bi2])

            out.append({
                "image_id": img_id,
                "score": float(p["score"]),
                "pred_class": pc,
                "is_tp": is_tp,
                "gt_class_any": gt_any,
                "bbox": list(p["bbox"]),
                "segmentation": p["segmentation"],
            })

    out.sort(key=lambda x: -x["score"])
    return out, iou_thresholds


def read_metrics():
    if not METRICS_PATH.exists():
        return []
    rows = []
    with open(METRICS_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


# ---------------------------------------------------------------------------
# Mask alpha-blend overlay helper
# ---------------------------------------------------------------------------

def _alpha_blend_masks(
    img_rgb, segs, classes_0idx, alpha=0.5,
    random_color=False, rng=None,
):
    overlay = img_rgb.astype(np.float32).copy()
    for seg, cls in zip(segs, classes_0idx):
        m = decode_rle(seg)
        if not m.any():
            continue
        if random_color and rng is not None:
            color = np.array(
                [rng.randint(64, 255) for _ in range(3)],
                dtype=np.float32,
            )
        else:
            color = np.array(class_color_rgb(cls), dtype=np.float32)
        for c in range(3):
            overlay[..., c] = np.where(
                m,
                overlay[..., c] * (1 - alpha) + color[c] * alpha,
                overlay[..., c],
            )
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _draw_bbox(ax, box_xywh, color, label=None, linewidth=1.2):
    x, y, w, h = box_xywh
    rect = mpatches.Rectangle(
        (x, y), w, h,
        linewidth=linewidth, edgecolor=color, facecolor="none",
    )
    ax.add_patch(rect)
    if label:
        ax.text(
            x, max(0, y - 2), label, fontsize=7, color="black",
            bbox=dict(
                facecolor=color, alpha=0.7, pad=0.5, edgecolor="none",
            ),
        )


# ---------------------------------------------------------------------------
# A1 -- PR curve
# ---------------------------------------------------------------------------

def fig_pr_curve(coco_eval, out_path):
    precision = coco_eval.eval["precision"]
    recall_axis = np.linspace(0, 1, precision.shape[1])

    pr_iou50 = precision[0, :, :, 0, -1]
    pr_iou50_valid = np.where(pr_iou50 > -1, pr_iou50, np.nan)
    mean_iou50 = np.nanmean(pr_iou50_valid, axis=1)

    pr_all = precision[:, :, :, 0, -1]
    pr_all_valid = np.where(pr_all > -1, pr_all, np.nan)
    mean_all = np.nanmean(pr_all_valid, axis=(0, 2))

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for k in range(pr_iou50.shape[1]):
        ax.plot(
            recall_axis, pr_iou50_valid[:, k], color=_CMAP(k),
            alpha=0.5, linewidth=1,
            label=f"{CLASS_NAMES[k]} @IoU=0.5",
        )
    ax.plot(
        recall_axis, mean_iou50, color="black", linewidth=2.2,
        label="mean @IoU=0.5",
    )
    ax.plot(
        recall_axis, mean_all, color="red", linewidth=2.4,
        linestyle="--",
        label="mean @IoU=[0.5:0.95]  (= mAP grading)",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve  (segm, val)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# A2 -- per-class AP
# ---------------------------------------------------------------------------

def fig_per_class_ap(coco_eval, out_path):
    prec = coco_eval.eval["precision"]
    per_class = []
    for k in range(prec.shape[2]):
        p = prec[:, :, k, 0, -1]
        p_valid = p[p > -1]
        per_class.append(
            float(p_valid.mean()) if p_valid.size else 0.0
        )

    fig, ax = plt.subplots(figsize=(7, 5))
    xs = np.arange(NUM_CLASSES)
    colors = [_CMAP(i) for i in xs]
    bars = ax.bar(xs, per_class, color=colors, edgecolor="black")
    for b, v in zip(bars, per_class):
        ax.text(
            b.get_x() + b.get_width() / 2, v + 0.005,
            f"{v:.3f}", ha="center", fontsize=9,
        )
    mean_ap = float(np.mean(per_class))
    ax.axhline(
        mean_ap, color="red", linestyle="--", alpha=0.7,
        label=f"mean = {mean_ap:.3f}",
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_xlabel("Class")
    ax.set_ylabel("AP@[0.5:0.95]  (segm)")
    ax.set_title("Per-class AP")
    ax.set_ylim(0, max(per_class + [0.01]) * 1.18)
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# A3 -- AP breakdown
# ---------------------------------------------------------------------------

def fig_ap_breakdown(stats, out_path):
    names = [
        "mAP\n[.50:.95]", "AP50", "AP75",
        "AP_small", "AP_medium", "AP_large",
    ]
    vals = [max(float(stats[i]), 0.0) for i in range(6)]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, vals, color="#4C78A8", edgecolor="black")
    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2, v + 0.005,
            f"{v:.3f}", ha="center", fontsize=9,
        )
    ax.set_ylabel("AP")
    ax.set_title("COCO AP Breakdown  (segm, val)")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# A4 -- training loss curves
# ---------------------------------------------------------------------------

def fig_loss_curve(metrics, out_path):
    if not metrics:
        print("  [skip A4] no metrics.json")
        return
    keys = [
        "total_loss",
        "loss_mask", "loss_mask_point",
        "loss_cls", "loss_box_reg",
        "loss_rpn_cls", "loss_rpn_loc",
    ]
    rows = [m for m in metrics if "total_loss" in m]
    if not rows:
        print("  [skip A4] no total_loss in metrics.json")
        return

    iters = [m["iteration"] for m in rows]

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for k in keys:
        ys = [m.get(k) for m in rows]
        if all(v is None for v in ys):
            continue
        ys = [(v if v is not None else np.nan) for v in ys]
        if k == "total_loss":
            ax.plot(iters, ys, linewidth=2.2, color="black", label=k)
        else:
            ax.plot(iters, ys, linewidth=1.0, alpha=0.75, label=k)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Curves")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# A5 -- val AP vs iteration
# ---------------------------------------------------------------------------

def fig_map_vs_iter(metrics, out_path):
    if not metrics:
        print("  [skip A5] no metrics.json")
        return

    candidates = [
        ("segm/AP", "segm/AP50", "segm/AP75", "segm"),
        ("bbox/AP", "bbox/AP50", "bbox/AP75", "bbox"),
    ]
    chosen = None
    for kAP, k50, k75, label in candidates:
        rows = [m for m in metrics if kAP in m]
        if rows:
            chosen = (rows, kAP, k50, k75, label)
            break
    if chosen is None:
        print("  [skip A5] no segm/AP or bbox/AP in metrics.json")
        return

    rows, kAP, k50, k75, label = chosen
    iters = [m["iteration"] for m in rows]
    AP = [m.get(kAP, np.nan) for m in rows]
    AP50 = [m.get(k50, np.nan) for m in rows]
    AP75 = [m.get(k75, np.nan) for m in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iters, AP50, marker="o", linewidth=2, label=f"{label}/AP50")
    ax.plot(iters, AP75, marker="s", linewidth=2, label=f"{label}/AP75")
    ax.plot(
        iters, AP, marker="^", linewidth=2.4, color="red",
        label=f"{label}/AP@[0.5:0.95]  (grading)",
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("AP")
    ax.set_title(f"Validation AP vs Iteration  ({label})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# B1 -- detection vis (GT / pred@th_high / pred@th_low)
# ---------------------------------------------------------------------------

def fig_detection_vis(
    coco_gt: COCO, results, out_path,
    n_images=3, th_high=0.5, th_low=0.05,
):
    results_by_img = defaultdict(list)
    for r in results:
        results_by_img[r["image_id"]].append(r)

    candidate_ids = [
        i for i in coco_gt.getImgIds()
        if len(coco_gt.getAnnIds(imgIds=i)) > 0
    ]
    random.Random(SEED).shuffle(candidate_ids)
    sample_ids = candidate_ids[:n_images]
    if not sample_ids:
        print("  [skip B1] no val images with GT")
        return

    rng = random.Random(SEED)
    fig, axes = plt.subplots(3, n_images, figsize=(4.8 * n_images, 14))
    if n_images == 1:
        axes = axes.reshape(3, 1)
    row_titles = [
        "Ground Truth",
        f"Pred (score >= {th_high})",
        f"Pred (score >= {th_low})",
    ]

    for col, img_id in enumerate(sample_ids):
        info = coco_gt.loadImgs(img_id)[0]
        img_path = TRAIN_DIR / info["file_name"]
        img = load_image_rgb(img_path)
        gts = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))
        preds = results_by_img.get(img_id, [])

        for row, kind in enumerate(["gt", "high", "low"]):
            ax = axes[row, col]
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(row_titles[row], fontsize=11)

            if kind == "gt":
                segs = [g["segmentation"] for g in gts]
                cls = [g["category_id"] - 1 for g in gts]
                vis = _alpha_blend_masks(
                    img, segs, cls, alpha=0.55,
                    random_color=True, rng=rng,
                )
                ax.imshow(vis)
                ax.set_title(
                    f"img_id={img_id}  (n_gt={len(gts)})", fontsize=9,
                )
            else:
                th = th_high if kind == "high" else th_low
                filt = [p for p in preds if p["score"] >= th]
                segs = [p["segmentation"] for p in filt]
                cls = [p["category_id"] - 1 for p in filt]
                vis = _alpha_blend_masks(
                    img, segs, cls, alpha=0.55,
                    random_color=True, rng=rng,
                )
                ax.imshow(vis)
                ax.set_title(
                    f"img_id={img_id}  (n_pred={len(filt)})",
                    fontsize=9,
                )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# B2 -- confusion matrix
# ---------------------------------------------------------------------------

def fig_confusion_matrix(matches, out_path, score_th=0.3):
    cm_arr = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for m in matches:
        if m["score"] < score_th or m["gt_class_any"] is None:
            continue
        gt_idx = m["gt_class_any"] - 1
        pr_idx = m["pred_class"] - 1
        if 0 <= gt_idx < NUM_CLASSES and 0 <= pr_idx < NUM_CLASSES:
            cm_arr[gt_idx, pr_idx] += 1

    row_sum = cm_arr.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm = cm_arr / row_sum

    fig, ax = plt.subplots(figsize=(6.5, 5.8))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Ground-truth class")
    ax.set_title(
        f"Confusion Matrix  "
        f"(class-agnostic mask IoU>=0.5,  score>={score_th})"
    )
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            v = cm_norm[i, j]
            if cm_arr[i, j] > 0:
                ax.text(
                    j, i, f"{v:.2f}\n({cm_arr[i, j]})",
                    ha="center", va="center", fontsize=8,
                    color="white" if v > 0.5 else "black",
                )
    fig.colorbar(im, ax=ax, fraction=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# B3 -- TP/FP confidence histogram
# ---------------------------------------------------------------------------

def fig_confidence_hist(matches, iou_thresholds, out_path):
    iou50 = int(np.argmin(np.abs(iou_thresholds - 0.5)))
    tp_scores = [m["score"] for m in matches if m["is_tp"][iou50]]
    fp_scores = [m["score"] for m in matches if not m["is_tp"][iou50]]
    bins = np.linspace(0, 1, 41)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(
        [tp_scores, fp_scores], bins=bins, stacked=True,
        label=[
            f"TP (n={len(tp_scores)})",
            f"FP (n={len(fp_scores)})",
        ],
        color=["#2ca02c", "#d62728"],
        edgecolor="white", linewidth=0.3,
    )
    ax.set_yscale("log")
    ax.set_xlabel("Score")
    ax.set_ylabel("Count (log)")
    ax.set_title("Prediction Confidence Distribution  (mask IoU>=0.5)")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# B4 -- P / R / F1 vs score threshold
# ---------------------------------------------------------------------------

def fig_prf1_vs_threshold(matches, iou_thresholds, total_gts, out_path):
    if not matches:
        print("  [skip B4] no matches")
        return None
    scores = np.array([m["score"] for m in matches])
    is_tp = np.stack([m["is_tp"] for m in matches])
    T = is_tp.shape[1]

    score_grid = np.linspace(0.01, 0.99, 50)
    P = np.zeros((len(score_grid), T))
    R = np.zeros((len(score_grid), T))
    F = np.zeros((len(score_grid), T))
    for si, t in enumerate(score_grid):
        keep = scores >= t
        kept = int(keep.sum())
        for ti in range(T):
            tp = int(is_tp[keep, ti].sum()) if kept else 0
            fp = kept - tp
            fn = total_gts - tp
            p = tp / max(tp + fp, 1)
            r = tp / max(tp + fn, 1)
            f1 = 2 * p * r / max(p + r, 1e-9)
            P[si, ti] = p
            R[si, ti] = r
            F[si, ti] = f1

    P_mean = P.mean(axis=1)
    R_mean = R.mean(axis=1)
    F_mean = F.mean(axis=1)
    best_idx = int(np.argmax(F_mean))
    best_t = float(score_grid[best_idx])

    iou50 = int(np.argmin(np.abs(iou_thresholds - 0.5)))
    iou75 = int(np.argmin(np.abs(iou_thresholds - 0.75)))

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(score_grid, P_mean, linewidth=2, label="Precision (mean IoU)")
    ax.plot(score_grid, R_mean, linewidth=2, label="Recall (mean IoU)")
    ax.plot(
        score_grid, F_mean, linewidth=2.2, color="red",
        label="F1 (mean IoU=[0.5:0.95])",
    )
    ax.plot(
        score_grid, F[:, iou50], linestyle="--", alpha=0.6,
        label="F1 @IoU=0.5",
    )
    ax.plot(
        score_grid, F[:, iou75], linestyle="--", alpha=0.6,
        label="F1 @IoU=0.75",
    )
    ax.axvline(
        best_t, color="gray", linestyle=":", alpha=0.7,
        label=f"best mean-F1 @ score={best_t:.2f}",
    )
    ax.set_xlabel("Score Threshold")
    ax.set_ylabel("Value")
    ax.set_title(
        "P / R / F1 vs Score Threshold  "
        "(primary curve averaged over IoU=[0.5:0.95])"
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return best_t


# ---------------------------------------------------------------------------
# B5 -- error cases (high-conf FP / missed GT / class confusion)
# ---------------------------------------------------------------------------

def fig_error_cases(
    coco_gt: COCO, matches, iou_thresholds, out_path,
    score_th=0.3, n_each=3,
):
    iou50 = int(np.argmin(np.abs(iou_thresholds - 0.5)))

    fp_cases = [
        m for m in matches
        if (not m["is_tp"][iou50]) and m["score"] >= score_th
    ]
    fp_cases = sorted(fp_cases, key=lambda x: -x["score"])[:n_each]

    conf_cases = [
        m for m in matches
        if (not m["is_tp"][iou50]) and m["gt_class_any"] is not None
        and m["score"] >= 0.2
        and m["pred_class"] != m["gt_class_any"]
    ]
    conf_cases = sorted(conf_cases, key=lambda x: -x["score"])[:n_each]

    preds_by_img = defaultdict(list)
    for m in matches:
        if m["score"] >= score_th and m["is_tp"][iou50]:
            preds_by_img[m["image_id"]].append(m)
    fn_cases = []
    for img_id in coco_gt.getImgIds():
        gts = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))
        if not gts:
            continue
        preds = preds_by_img.get(img_id, [])
        for g in gts:
            matched = False
            for p in preds:
                ious = _mask_iou(
                    p["segmentation"], [g["segmentation"]],
                )
                if (
                    ious.size and ious[0] >= 0.5
                    and p["pred_class"] == g["category_id"]
                ):
                    matched = True
                    break
            if not matched:
                fn_cases.append((img_id, g))
        if len(fn_cases) >= 30:
            break
    random.Random(0).shuffle(fn_cases)
    fn_cases = fn_cases[:n_each]

    rows = [
        ("High-conf FP", fp_cases, "fp"),
        ("Missed GT (FN)", fn_cases, "fn"),
        ("Class confusion", conf_cases, "conf"),
    ]

    fig, axes = plt.subplots(3, n_each, figsize=(5 * n_each, 14))
    if n_each == 1:
        axes = axes.reshape(3, 1)

    for row, (title, cases, kind) in enumerate(rows):
        for col in range(n_each):
            ax = axes[row, col]
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(title, fontsize=11)
            if col >= len(cases):
                ax.axis("off")
                continue

            if kind == "fn":
                img_id, gt = cases[col]
            else:
                m = cases[col]
                img_id = m["image_id"]
            info = coco_gt.loadImgs(img_id)[0]
            img = load_image_rgb(TRAIN_DIR / info["file_name"])

            gts = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))
            gt_segs = [g["segmentation"] for g in gts]
            gt_cls = [g["category_id"] - 1 for g in gts]
            vis = _alpha_blend_masks(img, gt_segs, gt_cls, alpha=0.30)

            ax.imshow(vis)
            ax.set_title(f"img_id={img_id}", fontsize=9)

            for g in gts:
                _draw_bbox(ax, g["bbox"], "lime", linewidth=0.8)

            if kind == "fp":
                _draw_bbox(
                    ax, m["bbox"], "red",
                    label=(
                        f"P:{CLASS_NAMES[m['pred_class'] - 1]} "
                        f"{m['score']:.2f}"
                    ),
                    linewidth=2.0,
                )
            elif kind == "conf":
                _draw_bbox(
                    ax, m["bbox"], "orange",
                    label=(
                        f"P:{CLASS_NAMES[m['pred_class'] - 1]} "
                        f"(GT:"
                        f"{CLASS_NAMES[m['gt_class_any'] - 1]}) "
                        f"{m['score']:.2f}"
                    ),
                    linewidth=2.0,
                )
            elif kind == "fn":
                _draw_bbox(
                    ax, gt["bbox"], "blue",
                    label=f"GT:{CLASS_NAMES[gt['category_id'] - 1]}",
                    linewidth=2.5,
                )

    fig.suptitle(
        "Error Cases  (mask IoU=0.5;  GT bboxes in lime)",
        fontsize=12, y=1.0,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# B6 -- LR schedule
# ---------------------------------------------------------------------------

def fig_lr_schedule(metrics, out_path):
    if not metrics:
        print("  [skip B6] no metrics.json")
        return
    rows = [m for m in metrics if "lr" in m]
    if not rows:
        print("  [skip B6] no lr in metrics.json")
        return
    iters = [m["iteration"] for m in rows]
    lrs = [m["lr"] for m in rows]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(iters, lrs, linewidth=1.5, color="#1f77b4")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Learning rate")
    ax.set_title("Learning Rate Schedule")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import warnings
    warnings.filterwarnings("ignore")

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("[1/3] building val GT json ...")
    build_val_gt_json()

    print("[2/3] running val inference ...")
    run_val_inference()

    print("[3/3] running COCOeval(segm) ...")
    coco_gt = COCO(str(VAL_GT_JSON))
    with open(VAL_PRED_JSON) as f:
        results = json.load(f)
    coco_eval = run_cocoeval(coco_gt)

    print("[match] computing mask-IoU matches ...")
    matches, iou_ts = match_predictions_mask(coco_gt, results)
    total_gts = len(coco_gt.getAnnIds())

    metrics = read_metrics()

    print(f"\n=== Generating figures into {PLOT_DIR} ===")
    print("[A1] PR curve ...")
    fig_pr_curve(coco_eval, PLOT_DIR / "01_pr_curve.png")
    print("[A2] Per-class AP ...")
    fig_per_class_ap(coco_eval, PLOT_DIR / "02_per_class_ap.png")
    print("[A3] AP breakdown ...")
    fig_ap_breakdown(coco_eval.stats, PLOT_DIR / "03_ap_breakdown.png")
    print("[A4] Loss curve ...")
    fig_loss_curve(metrics, PLOT_DIR / "04_loss_curve.png")
    print("[A5] mAP vs iter ...")
    fig_map_vs_iter(metrics, PLOT_DIR / "05_map_vs_iter.png")
    print("[B1] Detection vis ...")
    fig_detection_vis(
        coco_gt, results, PLOT_DIR / "06_detection_vis.png",
    )
    print("[B2] Confusion matrix ...")
    fig_confusion_matrix(matches, PLOT_DIR / "07_confusion_matrix.png")
    print("[B3] Confidence hist ...")
    fig_confidence_hist(
        matches, iou_ts, PLOT_DIR / "08_confidence_hist.png",
    )
    print("[B4] P/R/F1 vs threshold ...")
    best_t = fig_prf1_vs_threshold(
        matches, iou_ts, total_gts,
        PLOT_DIR / "09_prf1_vs_threshold.png",
    )
    if best_t is not None:
        print(f"       best mean-F1 threshold: {best_t:.3f}")
    print("[B5] Error cases ...")
    fig_error_cases(
        coco_gt, matches, iou_ts, PLOT_DIR / "10_error_cases.png",
    )
    print("[B6] LR schedule ...")
    fig_lr_schedule(metrics, PLOT_DIR / "11_lr_schedule.png")

    summary = {
        "exp": EXP_NAME,
        "n_val_images": len(coco_gt.getImgIds()),
        "n_val_annotations": total_gts,
        "n_predictions": len(results),
        "mAP": float(coco_eval.stats[0]),
        "AP50": float(coco_eval.stats[1]),
        "AP75": float(coco_eval.stats[2]),
        "AP_small": float(coco_eval.stats[3]),
        "AP_medium": float(coco_eval.stats[4]),
        "AP_large": float(coco_eval.stats[5]),
        "best_f1_score_threshold": best_t,
    }
    with open(PLOT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nfigures + summary written to {PLOT_DIR}")


if __name__ == "__main__":
    main()
