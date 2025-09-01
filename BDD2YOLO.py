"""
=====================================================
Project : BDD Dataset Analysis
Author  : Mukul Agarwal
Date    : 31-Aug-2025
=====================================================
"""
import json
import os
import shutil
from tqdm import tqdm 

# =========================
# CONFIG
# =========================
BDD_CLASSES = [
    "car",
    "traffic sign",
    "traffic light",
    "person",
    "truck",
    "bus",
    "bike",
    "rider",
    "motor",
    "train",
]
class2id = {cls: i for i, cls in enumerate(BDD_CLASSES)}

IMG_WIDTH = 1280
IMG_HEIGHT = 720

# Dataset paths
DATASET_ROOT = r"datasets/BDD"
TRAIN_JSON = r"bdd100k_labels_release\bdd100k_labels_images_train.json"
VAL_JSON = r"bdd100k_labels_release\bdd100k_labels_images_val.json"
TRAIN_IMG_DIR = r"bdd100k_images_100k/100k/train"
VAL_IMG_DIR = r"bdd100k_images_100k/100k/val"


# =========================
# CONVERTER
# =========================
def convert_bdd_to_yolo(bdd_json, img_src_dir, out_img_dir, out_label_dir):
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)

    with open(bdd_json, "r") as f:
        data = json.load(f)

    print(f"Starting conversion of {len(data)} images from {bdd_json}")

    num_skipped = 0
    for entry in tqdm(data, desc=f"Converting {os.path.basename(bdd_json)}"):
        img_name = entry["name"]
        labels = entry.get("labels", [])

        yolo_lines = []
        for label in labels:
            category = label["category"]
            if category not in class2id:
                num_skipped += 1
                continue

            box2d = label.get("box2d")
            if not box2d:
                num_skipped += 1
                continue

            x1, y1, x2, y2 = box2d["x1"], box2d["y1"], box2d["x2"], box2d["y2"]
            w = x2 - x1
            h = y2 - y1
            x_center = x1 + w / 2
            y_center = y1 + h / 2

            # Normalize
            x_center /= IMG_WIDTH
            y_center /= IMG_HEIGHT
            w /= IMG_WIDTH
            h /= IMG_HEIGHT

            class_id = class2id[category]
            yolo_lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
            )

        # Save YOLO label
        label_path = os.path.join(out_label_dir, os.path.splitext(img_name)[0] + ".txt")
        with open(label_path, "w") as f:
            f.write("\n".join(yolo_lines))

        # Copy image
        src_img = os.path.join(img_src_dir, img_name)
        dst_img = os.path.join(out_img_dir, img_name)
        if os.path.exists(src_img):
            shutil.copy2(src_img, dst_img)

    print(
        f"Finished {bdd_json}. Skipped {num_skipped} annotations not in target classes."
    )


# =========================
# YAML WRITER
# =========================
def write_yaml(path="bdd.yaml"):
    yaml_content = f"""path: {DATASET_ROOT}
train: images/train
val: images/val

nc: {len(BDD_CLASSES)}
names:
"""
    for cls in BDD_CLASSES:
        yaml_content += f"  - {cls}\n"

    with open(path, "w") as f:
        f.write(yaml_content)
    print(f"Wrote dataset config → {path}")


# =========================
# MAIN
# =========================
def main():
    # Create structure
    os.makedirs(os.path.join(DATASET_ROOT, "images/train"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_ROOT, "images/val"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_ROOT, "labels/train"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_ROOT, "labels/val"), exist_ok=True)

    # Convert Train
    convert_bdd_to_yolo(
        TRAIN_JSON,
        TRAIN_IMG_DIR,
        os.path.join(DATASET_ROOT, "images/train"),
        os.path.join(DATASET_ROOT, "labels/train"),
    )

    # Convert Val
    convert_bdd_to_yolo(
        VAL_JSON,
        VAL_IMG_DIR,
        os.path.join(DATASET_ROOT, "images/val"),
        os.path.join(DATASET_ROOT, "labels/val"),
    )

    # Write YAML
    write_yaml(os.path.join(DATASET_ROOT, "bdd.yaml"))


if __name__ == "__main__":
    main()
