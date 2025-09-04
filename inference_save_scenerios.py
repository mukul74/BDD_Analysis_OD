import os
import cv2
import glob
import json
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from collections import defaultdict

# --------------------
# CONFIG
# --------------------
MODEL_PATH = "best.pt"
IMG_DIR = r"datasets/BDD/images/val"
LABEL_DIR = r"datasets/BDD/labels/val"
META_JSON = r"bdd100k_labels_release/bdd100k_labels_images_val.json"
OUTPUT_DIR = r"runs/compare_preds_gt"
CONF_THRESH = 0.25  # inference confidence threshold for model
IOU_MATCH = 0.5  # IoU threshold to call a detection TP
MAX_SAVE_PER_BUCKET = 50

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model
model = YOLO(MODEL_PATH)
class_names = model.names

# Colors
PRED_COLOR = (0, 255, 0)  # green
GT_COLOR = (0, 0, 255)  # red


# --------------------
# Helpers
# --------------------
def iou_xyxy(box_a, box_b):
    """
    box = [x1,y1,x2,y2]
    Returns IoU float
    """
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)

    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def sanitize_folder_name(s):
    return str(s).replace(" ", "_").replace("/", "_")


# --------------------
# Load metadata
# --------------------
with open(META_JSON, "r") as f:
    meta_data = json.load(f)

meta_lookup = {}
for entry in meta_data:
    name = entry["name"].replace(".jpg", "")
    attrs = entry.get("attributes", {}) or {}
    meta_lookup[name] = {
        "weather": attrs.get("weather", "unknown"),
        "scene": attrs.get("scene", "unknown"),
        "timeofday": attrs.get("timeofday", "unknown"),
    }

# --------------------
# Storage for detection metrics
# --------------------
# Per-bucket: tp, fp, fn, total_gt, saved_images_count
metrics_data = defaultdict(
    lambda: {"tp": 0, "fp": 0, "fn": 0, "total_gt": 0, "saved": 0, "n_images": 0}
)

# --------------------
# Process images
# --------------------
image_paths = glob.glob(os.path.join(IMG_DIR, "*.jpg"))
image_paths.sort()

for img_path in tqdm(image_paths, desc="Processing", unit="img"):
    img = cv2.imread(img_path)
    if img is None:
        continue
    h, w, _ = img.shape
    base = os.path.splitext(os.path.basename(img_path))[0]

    # Run inference using image path (keeps behaviour consistent with your base code)
    # returns a Results object list; we take the first (single image)
    results = model(img_path, conf=CONF_THRESH, verbose=False)
    res = results[0]

    # Extract predicted boxes (xyxy), class and conf robustly (handle zero detections)
    if hasattr(res, "boxes") and len(res.boxes) > 0:
        try:
            preds_xyxy = res.boxes.xyxy.cpu().numpy()  # shape (N,4)
            pred_cls = res.boxes.cls.cpu().numpy().astype(int)
            pred_conf = res.boxes.conf.cpu().numpy()
        except Exception:
            # fallback if attributes have slightly different names/types
            # convert to numpy via list comprehensions
            preds_xyxy = (
                np.array([b.xyxy[0].cpu().numpy() for b in res.boxes])
                if len(res.boxes) > 0
                else np.zeros((0, 4))
            )
            pred_cls = (
                np.array([int(b.cls) for b in res.boxes])
                if len(res.boxes) > 0
                else np.array([], dtype=int)
            )
            pred_conf = (
                np.array([float(b.conf) for b in res.boxes])
                if len(res.boxes) > 0
                else np.array([])
            )
    else:
        preds_xyxy = np.zeros((0, 4))
        pred_cls = np.array([], dtype=int)
        pred_conf = np.array([])

    # Parse ground-truth boxes from YOLO txt labels (convert to xyxy)
    gt_boxes = []  # list of (x1,y1,x2,y2,cls)
    label_file = os.path.join(LABEL_DIR, base + ".txt")
    if os.path.exists(label_file):
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls, cx, cy, bw, bh = map(float, parts[:5])
                cls = int(cls)
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                # clamp
                x1, y1, x2, y2 = map(
                    int, [max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)]
                )
                gt_boxes.append([x1, y1, x2, y2, cls])

    # Prepare preds list (x1,y1,x2,y2,cls,conf)
    preds = []
    for (x1, y1, x2, y2), cls, conf in zip(preds_xyxy, pred_cls, pred_conf):
        # ensure ints for drawing and limits
        x1i, y1i, x2i, y2i = (
            int(round(x1)),
            int(round(y1)),
            int(round(x2)),
            int(round(y2)),
        )
        preds.append([x1i, y1i, x2i, y2i, int(cls), float(conf)])

    # Get metadata bucket key
    meta = meta_lookup.get(
        base, {"weather": "unknown", "scene": "unknown", "timeofday": "unknown"}
    )
    key = (meta["weather"], meta["scene"], meta["timeofday"])
    bucket = metrics_data[key]
    bucket["n_images"] += 1
    bucket["total_gt"] += len(gt_boxes)

    # Match preds -> gt using greedy by pred confidence (descending)
    matched_gt = set()
    tp = 0
    fp = 0

    preds_sorted = sorted(preds, key=lambda x: x[5], reverse=True)  # sort by conf

    for p in preds_sorted:
        p_box = p[:4]
        p_cls = p[4]
        best_iou = 0.0
        best_idx = -1
        for gi, g in enumerate(gt_boxes):
            if gi in matched_gt:
                continue
            g_box = g[:4]
            g_cls = g[4]
            if g_cls != p_cls:
                continue  # require class match
            iou = iou_xyxy(p_box, g_box)
            if iou >= IOU_MATCH and iou > best_iou:
                best_iou = iou
                best_idx = gi
        if best_idx >= 0:
            tp += 1
            matched_gt.add(best_idx)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)

    # Accumulate per-bucket
    bucket["tp"] += tp
    bucket["fp"] += fp
    bucket["fn"] += fn

    # Draw predictions and GT (same style as your base code)
    # Draw preds
    for x1, y1, x2, y2, cls, conf in preds:
        cv2.rectangle(img, (x1, y1), (x2, y2), PRED_COLOR, 2)
        cv2.putText(
            img,
            f"P:{class_names.get(cls,cls)} {conf:.2f}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            PRED_COLOR,
            1,
            cv2.LINE_AA,
        )
    # Draw GT
    for x1, y1, x2, y2, cls in gt_boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), GT_COLOR, 2)
        cv2.putText(
            img,
            f"GT:{class_names.get(cls,cls)}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            GT_COLOR,
            1,
            cv2.LINE_AA,
        )

    # Save annotated image only up to MAX_SAVE_PER_BUCKET
    if bucket["saved"] < MAX_SAVE_PER_BUCKET:
        safe_dir = os.path.join(
            OUTPUT_DIR,
            "images",
            sanitize_folder_name(
                f"{meta['weather']}_{meta['scene']}_{meta['timeofday']}"
            ),
        )
        os.makedirs(safe_dir, exist_ok=True)
        cv2.imwrite(os.path.join(safe_dir, base + ".jpg"), img)
        bucket["saved"] += 1

# --------------------
# Write CSV metrics per bucket
# --------------------
csv_file = os.path.join(OUTPUT_DIR, "metrics_by_condition.csv")
with open(csv_file, "w") as f:
    f.write(
        "weather,scene,time,precision,recall,f1,tp,fp,fn,total_gt,n_images,saved_images\n"
    )
    for key, vals in metrics_data.items():
        w, sc, t = key
        tp = vals["tp"]
        fp = vals["fp"]
        fn = vals["fn"]
        total_gt = vals["total_gt"]
        n_images = vals["n_images"]
        saved = vals["saved"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f.write(
            f"{w},{sc},{t},{prec:.4f},{rec:.4f},{f1:.4f},{tp},{fp},{fn},{total_gt},{n_images},{saved}\n"
        )

print(
    f"✅ Saved annotated images (<= {MAX_SAVE_PER_BUCKET} per bucket) to: {os.path.join(OUTPUT_DIR,'images')}"
)
print(f"✅ Metrics CSV: {csv_file}")
