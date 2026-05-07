"""HW3 instance segmentation -- training (best run).

Mask R-CNN X-101-32x8d FPN with three task-specific modifications:
    * DCNv2 on res3-res5
    * BCE + Dice hybrid mask loss
    * PointRend coarse-to-fine mask head

All three are guarded by independent flags (USE_DCN / USE_DICE_LOSS /
USE_POINTREND); turning all three off recovers the plain Mask R-CNN
baseline. Run with:

    python train.py
"""
from __future__ import annotations

import copy
import json
import os
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn.functional as F

import detectron2.data.transforms as T
import detectron2.layers.deform_conv as _d2_deform_conv
import detectron2.modeling.roi_heads.mask_head as _d2_mask_head
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.structures import BitMasks, Boxes, BoxMode, Instances
from detectron2.utils.logger import setup_logger
from pycocotools import mask as coco_mask


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE.parent / "hw3-data-release"
TRAIN_DIR = DATA_ROOT / "train"
EXP_NAME = "exp_d2_maskrcnn_x101_dcn_dice_pointrend"
OUTPUT_DIR = HERE.parent / "runs" / EXP_NAME

USE_DCN = True
USE_DICE_LOSS = True
USE_POINTREND = True
DICE_WEIGHT = 1.0
DICE_EPS = 1.0

MAX_ITER = 30000
BATCH_SIZE = 2
NUM_WORKERS = 4
BASE_LR = 0.0025
WARMUP_ITERS = 1000
LR_STEPS = (24000, 28000)
EVAL_PERIOD = 2000
CHECKPOINT_PERIOD = 2000
SEED = 42
USE_AMP = True

VAL_RATIO = 0.12
TRAIN_ON_ALL = False

NUM_CLASSES = 4

BASE_YAML_PLAIN = (
    "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
)
PRETRAINED_FILENAME_PLAIN = "mask_rcnn_X_101_32x8d_FPN_3x.pkl"
PRETRAINED_URL_PLAIN = (
    "https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/"
    "mask_rcnn_X_101_32x8d_FPN_3x/139653917/model_final_2d9806.pkl"
)

POINTREND_PROJECT_DIR = (
    HERE.parent / "detectron2" / "projects" / "PointRend"
)
POINTREND_BASE_YAML = (
    POINTREND_PROJECT_DIR / "configs" / "InstanceSegmentation"
    / "pointrend_rcnn_X_101_32x8d_FPN_3x_coco.yaml"
)
PRETRAINED_FILENAME_POINTREND = (
    "pointrend_rcnn_X_101_32x8d_FPN_3x_coco.pkl"
)
PRETRAINED_URL_POINTREND = (
    "https://dl.fbaipublicfiles.com/detectron2/PointRend/"
    "InstanceSegmentation/pointrend_rcnn_X_101_32x8d_FPN_3x_coco/"
    "28119989/model_final_ba17b9.pkl"
)

PRETRAINED_DIR = HERE / "pretrained"
PRETRAINED_DIR_FALLBACK_CODE8 = HERE.parent / "code8" / "pretrained"
PRETRAINED_DIR_FALLBACK_CODE7 = HERE.parent / "code7" / "pretrained"

ANCHOR_SIZES = [[8], [16], [32], [64], [128]]
ANCHOR_RATIOS = [[0.5, 1.0, 2.0]] * 5
RPN_PRE_NMS_TRAIN = 3000
RPN_POST_NMS_TRAIN = 2000
RPN_PRE_NMS_TEST = 2000
RPN_POST_NMS_TEST = 2000
RPN_NMS_THRESH = 0.7
ROI_NMS_THRESH = 0.5
DETECTIONS_PER_IMG = 1000
SCORE_THRESH_TEST = 0.05

DEFORM_ON_PER_STAGE = [False, True, True, True]
DEFORM_MODULATED = True
DEFORM_NUM_GROUPS = 1

INPUT_MIN_SIZE_TRAIN = (640, 768, 896, 1024)
INPUT_MAX_SIZE_TRAIN = 1333
INPUT_MIN_SIZE_TEST = 1024
INPUT_MAX_SIZE_TEST = 1333

HED_PROB = 0.8
HED_SIGMA = 0.05
HED_BIAS = 0.05

COCO_MAX_DETS = [100, 500, 1000]


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

CLASS_FILES = [f"class{i}.tif" for i in range(1, 5)]


def load_image_rgb(path):
    img = tifffile.imread(str(path))
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def load_instances(sample_dir):
    out = []
    for cls_idx, fname in enumerate(CLASS_FILES):
        f = sample_dir / fname
        if not f.exists():
            continue
        raw = tifffile.imread(str(f)).astype(np.int32)
        for inst_id in np.unique(raw):
            if inst_id == 0:
                continue
            m = (raw == inst_id)
            if m.sum() < 4:
                continue
            out.append((m, cls_idx))
    return out


def encode_rle(mask_bool):
    return coco_mask.encode(
        np.asfortranarray(mask_bool.astype(np.uint8))
    )


def get_dataset_dicts(sample_ids, root):
    root = Path(root)
    dicts = []
    for idx, sid in enumerate(sample_ids):
        d = root / sid
        img_path = d / "image.tif"
        if not img_path.exists():
            continue
        img = tifffile.imread(str(img_path))
        if img.ndim == 3:
            h, w = img.shape[:2]
        else:
            h, w = img.shape

        annos = []
        for mask, label in load_instances(d):
            ys, xs = np.where(mask)
            if ys.size == 0:
                continue
            x1, y1 = float(xs.min()), float(ys.min())
            x2, y2 = float(xs.max() + 1), float(ys.max() + 1)
            annos.append({
                "bbox": [x1, y1, x2, y2],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": int(label),
                "segmentation": encode_rle(mask),
            })

        dicts.append({
            "file_name": str(img_path),
            "image_id": idx,
            "height": int(h),
            "width": int(w),
            "annotations": annos,
            "_sample_id": sid,
            "_sample_dir": str(d),
        })
    return dicts


def stratified_split(root, val_ratio, seed):
    """Bucket images by which class*.tif files they contain, then split
    each bucket -- keeps rare classes balanced across train/val."""
    root = Path(root)
    sids = sorted(d.name for d in root.iterdir() if d.is_dir())
    buckets = {}
    for sid in sids:
        present = tuple(
            int((root / sid / f).exists()) for f in CLASS_FILES
        )
        buckets.setdefault(present, []).append(sid)
    rng = random.Random(seed)
    train, val = [], []
    for ids in buckets.values():
        rng.shuffle(ids)
        n = (
            max(1, int(round(len(ids) * val_ratio)))
            if len(ids) > 1 else 0
        )
        val.extend(ids[:n])
        train.extend(ids[n:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


# ---------------------------------------------------------------------------
# HED stain-jitter augmentation (H&E pathology)
# ---------------------------------------------------------------------------

_RGB_FROM_HED = np.array(
    [[0.65, 0.70, 0.29],
     [0.07, 0.99, 0.11],
     [0.27, 0.57, 0.78]],
    dtype=np.float32,
)
_HED_FROM_RGB = np.linalg.inv(_RGB_FROM_HED).astype(np.float32)


def hed_jitter_np(img, sigma, bias, rng):
    alpha = 1.0 + rng.uniform(-sigma, sigma, size=3).astype(np.float32)
    beta = rng.uniform(-bias, bias, size=3).astype(np.float32)
    x = img.astype(np.float32) / 255.0
    od = -np.log(np.clip(x, 1e-6, 1.0))
    hed = od @ _HED_FROM_RGB
    hed = hed * alpha + beta
    rgb = np.exp(-hed @ _RGB_FROM_HED)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


class HEDStainJitter(T.Augmentation):
    def __init__(self, prob=HED_PROB, sigma=HED_SIGMA, bias=HED_BIAS):
        super().__init__()
        self._init(locals())

    def get_transform(self, image):
        if np.random.rand() >= self.prob:
            return T.NoOpTransform()
        sigma, bias = self.sigma, self.bias
        rng = np.random.default_rng()

        class _HEDTransform(T.Transform):
            def apply_image(self, img):
                return hed_jitter_np(img, sigma, bias, rng)

            def apply_coords(self, coords):
                return coords

            def apply_segmentation(self, segm):
                return segm

            def inverse(self):
                return T.NoOpTransform()

        return _HEDTransform()


# ---------------------------------------------------------------------------
# Mapper
# ---------------------------------------------------------------------------

def build_train_augs():
    return [
        T.ResizeShortestEdge(
            short_edge_length=INPUT_MIN_SIZE_TRAIN,
            max_size=INPUT_MAX_SIZE_TRAIN,
            sample_style="choice",
        ),
        T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
        T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
        T.RandomRotation(
            angle=[0, 90, 180, 270],
            sample_style="choice",
            expand=False,
        ),
        T.RandomBrightness(0.85, 1.15),
        T.RandomContrast(0.85, 1.15),
        HEDStainJitter(prob=HED_PROB, sigma=HED_SIGMA, bias=HED_BIAS),
    ]


def build_test_augs():
    return [
        T.ResizeShortestEdge(
            short_edge_length=INPUT_MIN_SIZE_TEST,
            max_size=INPUT_MAX_SIZE_TEST,
            sample_style="choice",
        ),
    ]


def cell_mapper(dataset_dict, augmentations, is_train=True):
    dataset_dict = copy.deepcopy(dataset_dict)
    image = load_image_rgb(dataset_dict["file_name"])
    image = image[:, :, ::-1].copy()  # RGB -> BGR

    annos = dataset_dict.get("annotations", [])
    masks = []
    labels = []
    for a in annos:
        m = coco_mask.decode(a["segmentation"]).astype(bool)
        masks.append(m)
        labels.append(a["category_id"])

    aug_input = T.AugInput(image, sem_seg=None)
    transforms = T.AugmentationList(augmentations)(aug_input)
    image_aug = aug_input.image

    if masks:
        masks_aug = []
        labels_aug = []
        for m, lab in zip(masks, labels):
            mt = transforms.apply_segmentation(m.astype(np.uint8)) > 0
            if mt.sum() < 4:
                continue
            masks_aug.append(mt)
            labels_aug.append(lab)
        if masks_aug:
            masks_aug = np.stack(masks_aug, axis=0)
            labels_aug = np.asarray(labels_aug, dtype=np.int64)
            ys_xs = [np.where(m) for m in masks_aug]
            boxes_aug = np.array([
                [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
                for ys, xs in ys_xs
            ], dtype=np.float32)
        else:
            masks_aug = np.zeros(
                (0, image_aug.shape[0], image_aug.shape[1]),
                dtype=bool,
            )
            labels_aug = np.zeros((0,), dtype=np.int64)
            boxes_aug = np.zeros((0, 4), dtype=np.float32)
    else:
        masks_aug = np.zeros(
            (0, image_aug.shape[0], image_aug.shape[1]),
            dtype=bool,
        )
        labels_aug = np.zeros((0,), dtype=np.int64)
        boxes_aug = np.zeros((0, 4), dtype=np.float32)

    dataset_dict["image"] = torch.as_tensor(
        np.ascontiguousarray(image_aug.transpose(2, 0, 1))
    )

    if is_train:
        h, w = image_aug.shape[:2]
        target = Instances((h, w))
        target.gt_boxes = Boxes(
            torch.as_tensor(boxes_aug, dtype=torch.float32)
        )
        target.gt_classes = torch.as_tensor(
            labels_aug, dtype=torch.int64
        )
        target.gt_masks = BitMasks(
            torch.as_tensor(
                np.ascontiguousarray(masks_aug), dtype=torch.bool,
            )
        )
        dataset_dict["instances"] = target
        dataset_dict.pop("annotations", None)

    return dataset_dict


def train_mapper(d):
    return cell_mapper(d, build_train_augs(), is_train=True)


def test_mapper(d):
    return cell_mapper(d, build_test_augs(), is_train=False)


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


# AMP / DCNv2 dtype patch: cast offset/mask to input dtype so the kernel
# does not crash on fp16 input + fp32 offset/mask.

_orig_modulated_deform_conv = _d2_deform_conv.modulated_deform_conv


def _modulated_deform_conv_amp_safe(
    input, offset, mask, weight, bias=None, *args, **kwargs,
):
    if mask.dtype != input.dtype:
        mask = mask.to(input.dtype)
    if offset.dtype != input.dtype:
        offset = offset.to(input.dtype)
    return _orig_modulated_deform_conv(
        input, offset, mask, weight, bias, *args, **kwargs,
    )


# PointRend empty-batch patch: sample_point_labels() does torch.cat on an
# empty list when every image has zero foreground proposals; return an
# empty (0, P) tensor instead.

def _apply_pointrend_empty_batch_patch():
    import point_rend.mask_head as _pr_mh
    import point_rend.point_features as _pr_pf
    if getattr(_pr_pf, "_empty_batch_patched", False):
        return
    _orig = _pr_pf.sample_point_labels

    def _safe_sample_point_labels(instances, point_coords):
        if sum(len(x) for x in instances) == 0:
            P = (
                point_coords.shape[1]
                if point_coords.dim() >= 2 else 0
            )
            return point_coords.new_zeros((0, P))
        return _orig(instances, point_coords)

    _pr_pf.sample_point_labels = _safe_sample_point_labels
    _pr_mh.sample_point_labels = _safe_sample_point_labels
    _pr_pf._empty_batch_patched = True
    print("patched PointRend sample_point_labels for empty-batch safety")


# ---------------------------------------------------------------------------
# BCE + Dice mask loss (replaces detectron2's pixel-BCE)
# ---------------------------------------------------------------------------

def mask_rcnn_loss_bce_dice(pred_mask_logits, instances, vis_period=0):
    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    total_num_masks = pred_mask_logits.size(0)
    mask_side_len = pred_mask_logits.size(2)

    gt_classes = []
    gt_masks = []
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        if not cls_agnostic_mask:
            gt_classes_per_image = (
                instances_per_image.gt_classes.to(dtype=torch.int64)
            )
            gt_classes.append(gt_classes_per_image)
        gt_masks_per_image = (
            instances_per_image.gt_masks.crop_and_resize(
                instances_per_image.proposal_boxes.tensor,
                mask_side_len,
            ).to(device=pred_mask_logits.device)
        )
        gt_masks.append(gt_masks_per_image)

    if len(gt_masks) == 0:
        return pred_mask_logits.sum() * 0

    gt_masks = torch.cat(gt_masks, dim=0)

    if cls_agnostic_mask:
        pred_mask_logits = pred_mask_logits[:, 0]
    else:
        indices = torch.arange(
            total_num_masks, device=pred_mask_logits.device,
        )
        gt_classes = torch.cat(gt_classes, dim=0)
        pred_mask_logits = pred_mask_logits[indices, gt_classes]

    gt_masks_f = gt_masks.to(dtype=torch.float32)

    bce = F.binary_cross_entropy_with_logits(
        pred_mask_logits, gt_masks_f, reduction="mean",
    )

    p = pred_mask_logits.sigmoid().flatten(1)
    t = gt_masks_f.flatten(1)
    inter = (p * t).sum(dim=1)
    denom = p.sum(dim=1) + t.sum(dim=1)
    dice = 1.0 - (2.0 * inter + DICE_EPS) / (denom + DICE_EPS)
    dice = dice.mean()

    return bce + DICE_WEIGHT * dice


# ---------------------------------------------------------------------------
# Configuration builder
# ---------------------------------------------------------------------------

def _resolve_pretrained():
    """Look up the pretrained .pkl matching the current USE_POINTREND.

    Falls back to ../code7/ and ../code8/ pretrained dirs to avoid
    re-downloading the same file.
    """
    fname = (
        PRETRAINED_FILENAME_POINTREND
        if USE_POINTREND else PRETRAINED_FILENAME_PLAIN
    )
    primary = PRETRAINED_DIR / fname
    if primary.exists():
        return primary
    for fb in (
        PRETRAINED_DIR_FALLBACK_CODE8,
        PRETRAINED_DIR_FALLBACK_CODE7,
    ):
        candidate = fb / fname
        if candidate.exists():
            return candidate
    return primary


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

    cfg.MODEL.WEIGHTS = str(_resolve_pretrained())

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

    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = RPN_PRE_NMS_TRAIN
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = RPN_POST_NMS_TRAIN
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = RPN_PRE_NMS_TEST
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = RPN_POST_NMS_TEST
    cfg.MODEL.RPN.NMS_THRESH = RPN_NMS_THRESH

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESH_TEST
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = ROI_NMS_THRESH
    cfg.TEST.DETECTIONS_PER_IMAGE = DETECTIONS_PER_IMG

    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.INPUT.MIN_SIZE_TRAIN = INPUT_MIN_SIZE_TRAIN
    cfg.INPUT.MAX_SIZE_TRAIN = INPUT_MAX_SIZE_TRAIN
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
    cfg.INPUT.MIN_SIZE_TEST = INPUT_MIN_SIZE_TEST
    cfg.INPUT.MAX_SIZE_TEST = INPUT_MAX_SIZE_TEST

    cfg.SOLVER.IMS_PER_BATCH = BATCH_SIZE
    cfg.SOLVER.BASE_LR = BASE_LR
    cfg.SOLVER.WARMUP_ITERS = WARMUP_ITERS
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
    cfg.SOLVER.MAX_ITER = MAX_ITER
    cfg.SOLVER.STEPS = LR_STEPS
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.WEIGHT_DECAY = 1e-4
    cfg.SOLVER.CHECKPOINT_PERIOD = CHECKPOINT_PERIOD
    cfg.SOLVER.AMP.ENABLED = USE_AMP
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0

    cfg.DATASETS.TRAIN = ("hw3_cells_train",)
    if TRAIN_ON_ALL:
        cfg.DATASETS.TEST = ()
        cfg.TEST.EVAL_PERIOD = 0
    else:
        cfg.DATASETS.TEST = ("hw3_cells_val",)
        cfg.TEST.EVAL_PERIOD = EVAL_PERIOD
    cfg.DATALOADER.NUM_WORKERS = NUM_WORKERS

    cfg.OUTPUT_DIR = str(OUTPUT_DIR)
    cfg.SEED = SEED
    return cfg


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class CellTrainer(DefaultTrainer):

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=train_mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(
            cfg, dataset_name, mapper=test_mapper,
        )

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "eval")
        os.makedirs(output_folder, exist_ok=True)
        return COCOEvaluator(
            dataset_name,
            tasks=("bbox", "segm"),
            distributed=False,
            output_dir=output_folder,
            max_dets_per_image=COCO_MAX_DETS[-1],
        )


def register_datasets(train_ids, val_ids):
    cat_names = [f"class{i}" for i in range(1, 5)]

    if "hw3_cells_train" in DatasetCatalog.list():
        DatasetCatalog.remove("hw3_cells_train")
    if "hw3_cells_val" in DatasetCatalog.list():
        DatasetCatalog.remove("hw3_cells_val")

    DatasetCatalog.register(
        "hw3_cells_train",
        lambda: get_dataset_dicts(train_ids, TRAIN_DIR),
    )
    MetadataCatalog.get("hw3_cells_train").set(thing_classes=cat_names)

    if val_ids:
        DatasetCatalog.register(
            "hw3_cells_val",
            lambda: get_dataset_dicts(val_ids, TRAIN_DIR),
        )
        MetadataCatalog.get("hw3_cells_val").set(
            thing_classes=cat_names,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    warnings.filterwarnings("ignore")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_logger(
        output=str(OUTPUT_DIR), name="d2_cells_dcn_dice_pointrend",
    )

    if USE_POINTREND:
        _import_pointrend()
        _apply_pointrend_empty_batch_patch()

    weights = _resolve_pretrained()
    if not weights.exists():
        url = (
            PRETRAINED_URL_POINTREND
            if USE_POINTREND else PRETRAINED_URL_PLAIN
        )
        raise FileNotFoundError(
            f"Pretrained weights not found at {weights}.\n"
            f"Run run_all.sh to download, or fetch manually:\n"
            f"  curl -L -o {weights} \\\n    {url}"
        )
    print(f"using pretrained weights: {weights}")

    if USE_DICE_LOSS:
        _d2_mask_head.mask_rcnn_loss = mask_rcnn_loss_bce_dice
        if USE_POINTREND:
            _import_pointrend()
            import point_rend.mask_head as _pr_mask_head
            _pr_mask_head.mask_rcnn_loss = mask_rcnn_loss_bce_dice
            print(
                f"patched mask_rcnn_loss (d2 + PointRend) "
                f"-> BCE + {DICE_WEIGHT} * Dice"
            )
        else:
            print(
                f"patched mask_rcnn_loss "
                f"-> BCE + {DICE_WEIGHT} * Dice"
            )

    if USE_DCN and USE_AMP:
        _d2_deform_conv.modulated_deform_conv = (
            _modulated_deform_conv_amp_safe
        )
        print(
            "patched modulated_deform_conv for AMP fp16/fp32 "
            "compatibility"
        )

    print(
        f"USE_DCN={USE_DCN}  USE_DICE_LOSS={USE_DICE_LOSS}  "
        f"USE_POINTREND={USE_POINTREND}  USE_AMP={USE_AMP}"
    )

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    split_path = OUTPUT_DIR / "split.json"
    if split_path.exists():
        with open(split_path) as f:
            d = json.load(f)
        train_ids = d["train"]
        val_ids = d.get("val", [])
        if TRAIN_ON_ALL and val_ids:
            train_ids = train_ids + val_ids
            val_ids = []
    else:
        if TRAIN_ON_ALL:
            train_ids = sorted(
                d.name for d in TRAIN_DIR.iterdir() if d.is_dir()
            )
            val_ids = []
        else:
            train_ids, val_ids = stratified_split(
                TRAIN_DIR, VAL_RATIO, SEED,
            )
        with open(split_path, "w") as f:
            json.dump({"train": train_ids, "val": val_ids}, f, indent=2)
    print(
        f"split: train={len(train_ids)} val={len(val_ids)}  "
        f"TRAIN_ON_ALL={TRAIN_ON_ALL}"
    )

    register_datasets(train_ids, val_ids)

    cfg = build_cfg()
    with open(OUTPUT_DIR / "config.yaml", "w") as f:
        f.write(cfg.dump())

    from detectron2.modeling import build_model
    _model = build_model(cfg)
    n_params = sum(
        p.numel() for p in _model.parameters() if p.requires_grad
    )
    print(f"trainable params: {n_params / 1e6:.2f}M  (limit: 200M)")
    assert n_params < 200_000_000, "model exceeds 200M trainable params"
    del _model

    trainer = CellTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    main()
