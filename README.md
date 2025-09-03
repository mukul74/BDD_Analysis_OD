# Analysis of the BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning for Object Detection  
*Mukul Agarwal*  

This repository contains code and resources for analyzing the **BDD100K dataset**, one of the largest and most diverse open driving datasets. The focus is on object detection tasks, with an emphasis on understanding dataset composition, annotation quality, and category-level insights that can inform model training and evaluation.

---

## Data Analysis for Object Detection

The analysis is based on the following dataset components:

### Image Data
- `bdd100k_images_100k/100k/train/`  
- `bdd100k_images_100k/100k/val/`  

### Metadata Files
- `bdd100k_labels_release/bdd100k_labels_images_train.json`  
- `bdd100k_labels_release/bdd100k_labels_images_val.json`  

These metadata files provide object annotations (bounding boxes, categories, attributes) corresponding to each image in the dataset.

---

## Codebase

The following Python scripts are used in this analysis:

- `DataAnalysis.py` – main script for reading and analyzing the dataset.  
- `utilities.py` – helper functions for data parsing and statistics extraction.  

---

## Main Functionalities

The analysis covers dataset-level statistics, including:

- **Scene** distribution  
- **Weather** conditions  
- **Time of day**  
- **Category** frequencies  
- **Occlusion** levels  
- **Truncation** levels  
- **Number of objects per image**  

No. of Images in the Train Folder:  
- Train images: 69,863  
- Validation images: 10,000  

---

## Example Visualizations

Images showing different scenes with ground-truth bounding boxes:

<p align="center">
  <img src="plots/random_grids/train_grid.png" alt="Random Scenes Train" width="30%"><br>
  <sub>Training Set</sub>
</p>

<p align="center">
  <img src="plots/random_grids/val_grid.png" alt="Random Scenes Validation" width="30%"><br>
  <sub>Validation Set</sub>
</p>


# Statistics 
## Category Summary (Train vs Validation)

| Category      | Train Count | Val Count | Train % | Val % | Difference % |
|---------------|-------------|-----------|---------|-------|--------------|
| car           | 713,211     | 102,506   | 55.42   | 55.25 | 0.17         |
| traffic sign  | 239,686     | 34,908    | 18.63   | 18.82 | -0.19        |
| traffic light | 186,117     | 26,885    | 14.46   | 14.49 | -0.03        |
| person        | 91,349      | 13,262    | 7.10    | 7.15  | -0.05        |
| truck         | 29,971      | 4,245     | 2.33    | 2.29  | 0.04         |
| bus           | 11,672      | 1,597     | 0.91    | 0.86  | 0.05         |
| bike          | 7,210       | 1,007     | 0.56    | 0.54  | 0.02         |
| rider         | 4,517       | 649       | 0.35    | 0.35  | 0.00         |
| motor         | 3,002       | 452       | 0.23    | 0.24  | -0.01        |
| train         | 136         | 15        | 0.01    | 0.01  | 0.00         |

📁 Saved CSV: `plots/summaries/category_summary.csv`

## Weather Summary (Train vs Validation)

| Weather       | Train Count | Val Count | Train % | Val % | Difference % |
|---------------|-------------|-----------|---------|-------|--------------|
| clear         | 37,344      | 5,346     | 53.45   | 53.46 | -0.01        |
| overcast      | 8,770       | 1,239     | 12.55   | 12.39 | 0.16         |
| undefined     | 8,119       | 1,157     | 11.62   | 11.57 | 0.05         |
| snowy         | 5,549       | 769       | 7.94    | 7.69  | 0.25         |
| rainy         | 5,070       | 738       | 7.26    | 7.38  | -0.12        |
| partly cloudy | 4,881       | 738       | 6.99    | 7.38  | -0.39        |
| foggy         | 130         | 13        | 0.19    | 0.13  | 0.06         |

📁 Saved CSV: `plots/summaries/weather_summary.csv`

---

## Scene Summary (Train vs Validation)

| Scene        | Train Count | Val Count | Train % | Val % | Difference % |
|--------------|-------------|-----------|---------|-------|--------------|
| city street  | 43,516      | 6,112     | 62.29   | 61.12 | 1.17         |
| highway      | 17,379      | 2,499     | 24.88   | 24.99 | -0.11        |
| residential  | 8,074       | 1,253     | 11.56   | 12.53 | -0.97        |
| parking lot  | 377         | 49        | 0.54    | 0.49  | 0.05         |
| undefined    | 361         | 53        | 0.52    | 0.53  | -0.01        |
| tunnel       | 129         | 27        | 0.18    | 0.27  | -0.09        |
| gas stations | 27          | 7         | 0.04    | 0.07  | -0.03        |

📁 Saved CSV: `plots/summaries/scene_summary.csv`

---

## Time of Day Summary (Train vs Validation)

| Time of Day | Train Count | Val Count | Train % | Val % | Difference % |
|-------------|-------------|-----------|---------|-------|--------------|
| daytime     | 36,728      | 5,258     | 52.57   | 52.58 | -0.01        |
| night       | 27,971      | 3,929     | 40.04   | 39.29 | 0.75         |
| dawn/dusk   | 5,027       | 778       | 7.20    | 7.78  | -0.58        |
| undefined   | 137         | 35        | 0.20    | 0.35  | -0.15        |

📁 Saved CSV: `plots/summaries/timeofday_summary.csv`

## Bounding Box Statistics (Train vs Validation)

| Dataset | Width Mean | Width Median | Width Std | Height Mean | Height Median | Height Std | Aspect Ratio Mean | Aspect Ratio Median | Aspect Ratio Std | Area Mean  | Area Median | Area Std  |
|---------|------------|--------------|-----------|-------------|---------------|------------|-------------------|---------------------|------------------|------------|-------------|-----------|
| Train   | 56.60      | 28.70        | 75.69     | 49.92       | 29.09         | 61.69      | 1.22              | 1.10                | 0.95             | 0.007353   | 0.000887    | 0.024794  |
| Val     | 56.15      | 28.70        | 74.83     | 49.59       | 28.99         | 61.24      | 1.22              | 1.10                | 0.74             | 0.007247   | 0.000878    | 0.024326  |

📁 Saved CSV: `plots/summaries/bbox_statistics.csv`

## Comments
