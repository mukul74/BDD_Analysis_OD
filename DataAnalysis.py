"""
=====================================================
Project : BDD Dataset Analysis
Author  : Mukul Agarwal
Date    : 30-Aug-2025
=====================================================
"""

from utilities import *

# ========================
# Main
# ========================

if __name__ == "__main__":
    print("Start Analysis")

    # Loading the dataset
    # JSON files
    train_json = os.path.join(
        "bdd100k_labels_release", "bdd100k_labels_images_train.json"
    )
    val_json = os.path.join("bdd100k_labels_release", "bdd100k_labels_images_val.json")

    # Image folders
    train_img_root = os.path.join("bdd100k_images_100k", "100k", "train")
    val_img_root = os.path.join("bdd100k_images_100k", "100k", "val")

    # Reading Json
    train_data = load_json(train_json)
    val_data = load_json(val_json)

    print(f"Train images: {len(train_data)} | Val images: {len(val_data)}")
    excluded = {"drivable area", "lane"}

    # Creating the directories for summaries and images
    ensure_dir("plots/train")
    ensure_dir("plots/val")
    ensure_dir("plots/random_grids")

    print("Running Analysis")
    train_stats, train_bboxes = run_dataset_analysis(
        "Train", train_data, train_img_root, excluded, "plots/train"
    )
    val_stats, val_bboxes = run_dataset_analysis(
        "Val", val_data, val_img_root, excluded, "plots/val"
    )

    print("Generating Summaries")
    # Summaries
    generate_category_summary(train_stats["category"], val_stats["category"], excluded)
    generate_attribute_summary(
        train_stats["weather"], val_stats["weather"], "weather", excluded
    )
    generate_attribute_summary(
        train_stats["scene"], val_stats["scene"], "scene", excluded
    )
    generate_attribute_summary(
        train_stats["timeofday"], val_stats["timeofday"], "timeofday", excluded
    )
    generate_bbox_summary(train_bboxes, val_bboxes, "plots/summaries")

    print("Detecting Anomalies")
    bad_train, bad_train_summary = find_bad_bboxes(train_data)
    visualize_bad_bboxes(bad_train, img_dir=train_img_root, out_dir="./BadData/Train/")

    bad_val, bad_val_summary = find_bad_bboxes(val_data)
    visualize_bad_bboxes(bad_val, img_dir=val_img_root, out_dir="./BadData/Val/")

    print("Analysis Complete")
