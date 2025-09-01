import os
import cv2
import json
import random
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from tabulate import tabulate
from tqdm import tqdm
from itertools import combinations

# ========================
# Utility functions
# ========================


def load_json(filepath):
    with open(filepath, "r") as file:
        return json.load(file)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_histogram_plot(data, title, xlabel, ylabel, filename, bins=None):
    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=bins, edgecolor="black")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def save_bar_plot(counter, title, xlabel, ylabel, filename):
    sorted_items = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    keys, values = zip(*sorted_items) if sorted_items else ([], [])
    plt.figure(figsize=(8, 5))
    plt.bar(keys, values, edgecolor="black")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def draw_bboxes(image, labels, excluded_categories=None):
    excluded_categories = excluded_categories or []
    for label in labels:
        category = label.get("category")
        if category in excluded_categories:
            continue

        box2d = label.get("box2d")
        if not box2d:
            # skip labels without box2d (e.g., segmentation-only annotations)
            continue

        x1, y1, x2, y2 = map(int, box2d.values())

        color = (0, 255, 0)  # green
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)
        cv2.putText(
            image,
            category,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return image


def save_random_images(
    dataset,
    img_root,
    rows=5,
    cols=5,
    excluded_categories=None,
    save_path="plots/random_grid.png",
):
    total_available = len(dataset)
    print(f"\nSaving {rows * cols} random samples to {save_path}")

    if total_available == 0:
        print("Dataset is empty.")
        return

    samples = random.sample(dataset, min(rows * cols, total_available))
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
    axes = axes.flatten()

    for ax, entry in zip(axes, samples):
        img_path = os.path.join(img_root, entry["name"])
        if not os.path.exists(img_path):
            ax.axis("off")
            continue
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        image = draw_bboxes(image, entry.get("labels", []), excluded_categories)
        ax.imshow(image)
        ax.set_title(entry["name"], fontsize=8)
        ax.axis("off")

    for ax in axes[len(samples) :]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def generate_stats(dataset, excluded_categories):
    """
        Compute counts of objects, scenes, weather, and time of day.

        Args:
            data (list of dict): Dataset annotations (train or val) and exculded categories

        Returns:
            dict: Counters for objects, scenes, weather, time of day etc.
    """
    objects_per_image = []
    weather_counter, scene_counter, timeofday_counter = Counter(), Counter(), Counter()
    category_counter, occluded_counter, truncated_counter = Counter(), Counter(), Counter()

    for entry in tqdm(dataset, desc="Processing dataset"):
        valid_object_count = 0
        attrs = entry.get("attributes", {})
        weather_counter[attrs.get("weather", "unknown")] += 1
        scene_counter[attrs.get("scene", "unknown")] += 1
        timeofday_counter[attrs.get("timeofday", "unknown")] += 1

        for label in entry.get("labels", []):
            category = label.get("category", "unknown")
            if category in excluded_categories:
                continue
            valid_object_count += 1
            label_attrs = label.get("attributes", {})
            occluded_counter[label_attrs.get("occluded", "unknown")] += 1
            truncated_counter[label_attrs.get("truncated", "unknown")] += 1
            category_counter[category] += 1

            objects_per_image.append(valid_object_count)

    return {
        "weather": weather_counter,
        "scene": scene_counter,
        "timeofday": timeofday_counter,
        "category": category_counter,
        "occluded": occluded_counter,
        "truncated": truncated_counter,
        "objects_per_image": objects_per_image,
    }


def analyze_bboxes(dataset, excluded_categories, img_width=1280, img_height=720):
    widths, heights, areas, aspect_ratios = [], [], [], []
    centers_x, centers_y, category_sizes = [], [], {}

    for entry in dataset:
        for label in entry.get("labels", []):
            category = label.get("category", "unknown")
            if category in excluded_categories:
                continue
            box2d = label.get("box2d")
            if not box2d:
                continue
            x1, y1, x2, y2 = box2d["x1"], box2d["y1"], box2d["x2"], box2d["y2"]
            w, h = max(0, x2 - x1), max(0, y2 - y1)
            if w == 0 or h == 0:
                continue
            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / h)
            areas.append((w * h) / (img_width * img_height))
            centers_x.append((x1 + x2) / 2)
            centers_y.append((y1 + y2) / 2)
            category_sizes.setdefault(category, []).append((w, h))

    return widths, heights, aspect_ratios, areas, centers_x, centers_y, category_sizes


def save_bbox_analysis(
    widths, heights, aspect_ratios, areas, category_sizes, prefix, folder
):
    save_histogram_plot(
        widths,
        f"{prefix}: BBox Widths",
        "Width (px)",
        "Count",
        os.path.join(folder, f"{prefix}_bbox_width.png"),
    )
    save_histogram_plot(
        heights,
        f"{prefix}: BBox Heights",
        "Height (px)",
        "Count",
        os.path.join(folder, f"{prefix}_bbox_height.png"),
    )
    save_histogram_plot(
        aspect_ratios,
        f"{prefix}: BBox Aspect Ratios",
        "W/H",
        "Count",
        os.path.join(folder, f"{prefix}_bbox_aspect_ratio.png"),
    )
    save_histogram_plot(
        areas,
        f"{prefix}: BBox Areas",
        "Area (normalized)",
        "Count",
        os.path.join(folder, f"{prefix}_bbox_area.png"),
    )

    avg_sizes = {
        cat: np.mean([w * h for w, h in sizes]) for cat, sizes in category_sizes.items()
    }
    top15 = dict(sorted(avg_sizes.items(), key=lambda x: x[1], reverse=True)[:15])
    plt.figure(figsize=(10, 6))
    plt.bar(list(top15.keys()), list(top15.values()), edgecolor="black")
    plt.title(f"{prefix}: Top 15 Categories by Avg BBox Area")
    plt.xticks(rotation=45)
    plt.ylabel("Avg Area (px²)")
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f"{prefix}_top15_bbox_area.png"))
    plt.close()


# ========================
# Summary functions (added back)
# ========================


def generate_category_summary(
    train_stats, val_stats, excluded_categories, summary_folder="plots/summaries"
):
    """
    Create CSV & pretty print comparing category counts between train and val.
    """
    ensure_dir(summary_folder)
    all_categories = sorted(set(train_stats.keys()) | set(val_stats.keys()))
    total_train = sum(train_stats.values())
    total_val = sum(val_stats.values())

    rows = []
    for cat in all_categories:
        t = train_stats.get(cat, 0)
        v = val_stats.get(cat, 0)
        t_pct = 100 * t / total_train if total_train > 0 else 0
        v_pct = 100 * v / total_val if total_val > 0 else 0
        rows.append(
            {
                "Category": cat,
                "Train Count": t,
                "Val Count": v,
                "Train %": round(t_pct, 2),
                "Val %": round(v_pct, 2),
                "Difference %": round(t_pct - v_pct, 2),
            }
        )

    df = pd.DataFrame(rows).sort_values(by="Train Count", ascending=False)
    save_path = os.path.join(summary_folder, "category_summary.csv")
    df.to_csv(save_path, index=False)

    print("\n=== Category Summary (Train vs Val) ===")
    print(tabulate(df, headers="keys", tablefmt="fancy_grid", showindex=False))
    print(f"\nSaved CSV to {save_path}")
    # return df


# keep this helper from your code
def ensure_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

# your original function (unchanged)
# def generate_category_summary(
#     train_stats, val_stats, excluded_categories, summary_folder="plots/summaries"
# ):
#     """
#     Create CSV & pretty print comparing category counts between train and val.
#     """
#     ensure_dir(summary_folder)
#     all_categories = sorted(set(train_stats.keys()) | set(val_stats.keys()))
#     total_train = sum(train_stats.values())
#     total_val = sum(val_stats.values())

#     rows = []
#     for cat in all_categories:
#         t = train_stats.get(cat, 0)
#         v = val_stats.get(cat, 0)
#         t_pct = 100 * t / total_train if total_train > 0 else 0
#         v_pct = 100 * v / total_val if total_val > 0 else 0
#         rows.append(
#             {
#                 "Category": cat,
#                 "Train Count": t,
#                 "Val Count": v,
#                 "Train %": round(t_pct, 2),
#                 "Val %": round(v_pct, 2),
#                 "Difference %": round(t_pct - v_pct, 2),
#             }
#         )

#     df = pd.DataFrame(rows).sort_values(by="Train Count", ascending=False)
#     save_path = os.path.join(summary_folder, "category_summary.csv")
#     df.to_csv(save_path, index=False)

#     print("\n=== Category Summary (Train vs Val) ===")
#     print(tabulate(df, headers="keys", tablefmt="fancy_grid", showindex=False))
#     print(f"\nSaved CSV to {save_path}")
    
#     return df  # return DataFrame for Gradio

def generate_attribute_summary(
    train_stats,
    val_stats,
    attribute_name,
    excluded_categories,
    summary_folder="plots/summaries",
):
    """
    Compare a top-level image attribute (e.g., 'weather', 'scene', 'timeofday') between train & val.
    """
    ensure_dir(summary_folder)

    all_values = sorted(set(train_stats.keys()) | set(val_stats.keys()))
    total_train = sum(train_stats.values())
    total_val = sum(val_stats.values())

    rows = []
    for v in all_values:
        t = train_stats.get(v, 0)
        val = val_stats.get(v, 0)
        t_pct = 100 * t / total_train if total_train > 0 else 0
        val_pct = 100 * val / total_val if total_val > 0 else 0
        rows.append(
            {
                attribute_name.capitalize(): v,
                "Train Count": t,
                "Val Count": val,
                "Train %": round(t_pct, 2),
                "Val %": round(val_pct, 2),
                "Difference %": round(t_pct - val_pct, 2),
            }
        )

    df = pd.DataFrame(rows).sort_values(by="Train Count", ascending=False)
    save_path = os.path.join(summary_folder, f"{attribute_name}_summary.csv")
    df.to_csv(save_path, index=False)

    print(f"\n=== {attribute_name.capitalize()} Summary (Train vs Val) ===")
    print(tabulate(df, headers="keys", tablefmt="fancy_grid", showindex=False))
    print(f"\nSaved CSV to {save_path}")
    # return df


def generate_bbox_summary(train_bboxes, val_bboxes, summary_folder="plots/summaries"):
    """
    train_bboxes / val_bboxes: tuple or list of (widths, heights, aspect_ratios, areas)
    Produces a CSV containing mean/median/std for each metric across train & val.
    """
    ensure_dir(summary_folder)
    save_path = os.path.join(summary_folder, "bbox_summary.csv")

    def safe_stats(arr):
        if not arr:
            return 0.0, 0.0, 0.0
        a = np.array(arr)
        return float(np.mean(a)), float(np.median(a)), float(np.std(a))

    rows = []
    for name, b in [("Train", train_bboxes), ("Val", val_bboxes)]:
        widths, heights, aspect_ratios, areas = b
        w_mean, w_med, w_std = safe_stats(widths)
        h_mean, h_med, h_std = safe_stats(heights)
        ar_mean, ar_med, ar_std = safe_stats(aspect_ratios)
        a_mean, a_med, a_std = safe_stats(areas)
        rows.append(
            {
                "Dataset": name,
                "Width Mean": round(w_mean, 2),
                "Width Median": round(w_med, 2),
                "Width Std": round(w_std, 2),
                "Height Mean": round(h_mean, 2),
                "Height Median": round(h_med, 2),
                "Height Std": round(h_std, 2),
                "Aspect Ratio Mean": round(ar_mean, 2),
                "Aspect Ratio Median": round(ar_med, 2),
                "Aspect Ratio Std": round(ar_std, 2),
                "Area Mean": round(a_mean, 6),
                "Area Median": round(a_med, 6),
                "Area Std": round(a_std, 6),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print("\n=== Bounding Box Summary (Train vs Val) ===")
    print(tabulate(df, headers="keys", tablefmt="fancy_grid", showindex=False))
    print(f"\nSaved CSV to {save_path}")
    # return df


def run_dataset_analysis(name, dataset, img_root, excluded, out_folder):
    """

    """
    
    print(f"\n--- {name} Data Analysis ---")
    stats = generate_stats(dataset, excluded)

    # Save Bar Plots and Histogram plots
    for key, counter in stats.items():
        if key == "objects_per_image":
            if counter:
                save_histogram_plot(
                    counter,
                    f"{name}: Objects per Image",
                    "Objects",
                    "Count",
                    os.path.join(out_folder, "objects_per_image.png"),
                    bins=range(0, max(counter) + 2),
                )
        else:
            save_bar_plot(
                counter,
                f"{name}: {key.capitalize()} Distribution",
                key.capitalize(),
                "Image Count",
                os.path.join(out_folder, f"{key}_distribution.png"),
            )

    # Save random grid
    # ensure_dir("plots/random_grids")
    save_random_images(
        dataset,
        img_root,
        save_path=os.path.join("plots/random_grids", f"{name.lower()}_grid.png"),
    )

    if stats["objects_per_image"]:
        print(
            f"{name} Objects/Image Stats: Mean={np.mean(stats['objects_per_image']):.2f}, Median={np.median(stats['objects_per_image']):.2f}"
        )

    # BBox analysis
    widths, heights, aspect_ratios, areas, _, _, cat_sizes = analyze_bboxes(
        dataset, excluded
    )
    save_bbox_analysis(
        widths, heights, aspect_ratios, areas, cat_sizes, prefix=name, folder=out_folder
    )
    return stats,(widths, heights, aspect_ratios, areas)


# ========================
# Find Anomolies
# ========================

def visualize_bad_bboxes(bad_df, img_dir, out_dir, num_samples=100):
    """
    Save images with bad bounding boxes overlayed (instead of showing).
    """
    os.makedirs(out_dir, exist_ok=True)

    for idx, row in bad_df.head(num_samples).iterrows():
        img_name = row["image"]  # str
        img_path = os.path.join(img_dir, img_name)

        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Draw bbox
        x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Add label text
        cv2.putText(
            img,
            f"{row['reason']}:{row['category']}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
        )

        # Save instead of show
        out_path = os.path.join(out_dir, f"{idx}_{os.path.basename(img_name)}")
        plt.imsave(out_path, img)

    print(f"✅ Saved {min(num_samples, len(bad_df))} bad bbox samples to {out_dir}")


def compute_iou(b1, b2):
    """Compute IoU between two boxes: (x1, y1, x2, y2)."""
    x1 = max(b1["x1"], b2["x1"])
    y1 = max(b1["y1"], b2["y1"])
    x2 = min(b1["x2"], b2["x2"])
    y2 = min(b1["y2"], b2["y2"])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (b1["x2"] - b1["x1"]) * (b1["y2"] - b1["y1"])
    area2 = (b2["x2"] - b2["x1"]) * (b2["y2"] - b2["y1"])

    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0


def find_bad_bboxes(
    dataset,
    img_width=1280,
    img_height=720,
    min_width=1,
    min_height=1,
    min_area_ratio=0.00001,
    max_aspect_ratio=10,
    overlap_iou_thresh=0.9,
    out_csv="plots/summaries/bad_bboxes.csv",
):
    """
    Detect bad bounding boxes:
    - Invalid (w <= 0 or h <= 0)
    - Too small (w < min_width, h < min_height, or area_ratio < min_area_ratio)
    - Out of bounds (coords outside image size)
    - Weird aspect ratio (w >> h or h >> w)
    - Duplicate overlaps (same class boxes IoU > overlap_iou_thresh)
    Saves both full records and per-class summary.
    """
    records = []
    total_area = img_width * img_height

    for entry in dataset:
        img_name = entry.get("name", "unknown")
        boxes = []

        for label in entry.get("labels", []):
            box2d = label.get("box2d")
            if not box2d:
                continue
            x1, y1, x2, y2 = box2d["x1"], box2d["y1"], box2d["x2"], box2d["y2"]
            w, h = x2 - x1, y2 - y1
            area_ratio = (w * h) / total_area if total_area > 0 else 0

            reason = []
            if w <= 0 or h <= 0:
                reason.append("invalid")
            elif w < min_width or h < min_height or area_ratio < min_area_ratio:
                reason.append("too_small")

            if x1 < 0 or y1 < 0 or x2 > img_width or y2 > img_height:
                reason.append("out_of_bounds")

            if w > 0 and h > 0:
                aspect_ratio = max(w / h, h / w)
                if aspect_ratio > max_aspect_ratio:
                    reason.append("weird_aspect_ratio")

            record = {
                "image": img_name,
                "category": label.get("category", "unknown"),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "width": w,
                "height": h,
                "area_ratio": round(area_ratio, 6),
                "reason": ",".join(reason) if reason else "",
            }
            boxes.append(record)
            if reason:  # store anomalies
                records.append(record)

        # Check duplicate overlaps (same-category, same-image)
        for b1, b2 in combinations(boxes, 2):
            if b1["category"] == b2["category"]:
                iou = compute_iou(b1, b2)
                if iou > overlap_iou_thresh:
                    for b in (b1, b2):
                        dup_record = b.copy()
                        dup_record["reason"] = "duplicate_overlap"
                        records.append(dup_record)

    if records:
        df = pd.DataFrame(records)
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        df.to_csv(out_csv, index=False)

        # Per-class summary
        summary = df.groupby(["category", "reason"]).size().reset_index(name="count")

        print(
            f"\nFound {len(records)} anomalous bounding boxes. Saved full list to {out_csv}"
        )
        print("=== Per-Class Bad BBox Summary ===")
        print(tabulate(summary, headers="keys", tablefmt="fancy_grid", showindex=False))

        return df, summary
    else:
        print("No bad bounding boxes found.")
        return None, None
