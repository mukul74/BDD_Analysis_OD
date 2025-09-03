from ultralytics import YOLO
from tabulate import tabulate  

# Load your trained model
model = YOLO("best.pt")

# Run evaluation on val set
metrics = model.val(data=r"datasets\BDD\bdd.yaml", imgsz=640, plots=True)

# Class names from model
class_names = model.names

# Extract per-class metrics
precisions = metrics.box.mp_per_class if metrics.box else []
recalls = metrics.box.mr_per_class if metrics.box else []
map50s = metrics.box.map50_per_class if metrics.box else []
map5095s = metrics.box.map_per_class if metrics.box else []

# Build per-class table
class_table = []
for i, cls in enumerate(class_names.values()):
    class_table.append(
        [
            cls,
            f"{precisions[i]:.4f}",
            f"{recalls[i]:.4f}",
            f"{map50s[i]:.4f}",
            f"{map5095s[i]:.4f}",
        ]
    )

print("\n=== Per-Class Metrics ===")
print(
    tabulate(
        class_table,
        headers=["Class", "Precision", "Recall", "mAP@50", "mAP@50-95"],
        tablefmt="pretty",
    )
)

# Extract overall metrics
overall_table = [
    ["Precision", f"{metrics.results_dict['metrics/precision(B)']:.4f}"],
    ["Recall", f"{metrics.results_dict['metrics/recall(B)']:.4f}"],
    ["mAP@50", f"{metrics.results_dict['metrics/mAP50(B)']:.4f}"],
    ["mAP@50-95", f"{metrics.results_dict['metrics/mAP50-95(B)']:.4f}"],
]

print("\n=== Overall Metrics ===")
print(tabulate(overall_table, headers=["Metric", "Value"], tablefmt="pretty"))
