import os
import cv2
import glob
from tqdm import tqdm
from ultralytics import YOLO

# --------------------
# CONFIG
# --------------------
MODEL_PATH = "best.pt"
IMG_DIR = "datasets/BDD/images/val"  # images folder
LABEL_DIR = "datasets/BDD/labels/val"  # YOLO format GT labels
OUTPUT_DIR = "runs/compare_preds_gt"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model
model = YOLO(MODEL_PATH)

# Colors
PRED_COLOR = (0, 255, 0)  # green
GT_COLOR = (0, 0, 255)  # red

# Class names
class_names = model.names

# Process images
image_paths = glob.glob(os.path.join(IMG_DIR, "*.jpg"))

for img_path in tqdm(image_paths, desc="Processing", unit="img"):
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    base = os.path.splitext(os.path.basename(img_path))[0]

    # 1. Predictions
    results = model(img_path, conf=0.25, verbose=False)
    preds = results[0].boxes.xyxy.cpu().numpy()  # [x1,y1,x2,y2]
    pred_cls = results[0].boxes.cls.cpu().numpy().astype(int)
    pred_conf = results[0].boxes.conf.cpu().numpy()

    # Draw predictions
    for (x1, y1, x2, y2), cls, conf in zip(preds, pred_cls, pred_conf):
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), PRED_COLOR, 2)
        cv2.putText(
            img,
            f"P:{class_names[cls]} {conf:.2f}",
            (int(x1), int(y1) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            PRED_COLOR,
            2,
        )

    # 2. Ground Truth
    label_file = os.path.join(LABEL_DIR, base + ".txt")
    if os.path.exists(label_file):
        with open(label_file, "r") as f:
            for line in f:
                cls, cx, cy, bw, bh = map(float, line.strip().split())
                cls = int(cls)
                # convert YOLO format → xyxy
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), GT_COLOR, 2)
                cv2.putText(
                    img,
                    f"GT:{class_names[cls]}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    GT_COLOR,
                    2,
                )

    # Save combined image
    cv2.imwrite(os.path.join(OUTPUT_DIR, base + ".jpg"), img)

print(f"✅ Done! Results saved in {OUTPUT_DIR}")
