import os
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# --- Step 1: Load Metrics from results.csv ---
results_csv = "runs/detect/BurnGain/results.csv"
df = pd.read_csv(results_csv)
df.columns = df.columns.str.strip()

print("Available columns:", df.columns.tolist())

# --- Step 2: Plot Training Metrics ---
plt.figure(figsize=(12, 6))
plt.plot(df["metrics/precision(B)"], label="Precision")
plt.plot(df["metrics/recall(B)"], label="Recall")
plt.plot(df["metrics/mAP50(B)"], label="mAP@0.5")
plt.plot(df["metrics/mAP50-95(B)"], label="mAP@0.5:0.95")
plt.xlabel("Epoch")
plt.ylabel("Metric Score")
plt.title("BurnGain: Training Metrics Over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("burngain_training_metrics.png")
plt.show()
print("✅ Training metrics plot saved as burngain_training_metrics.png")

# --- Step 3: Generate Confusion Matrix ---
model = YOLO("runs/detect/BurnGain/weights/best.pt")
val_images_path = "yolo_dataset/images/train"  # adjust if using a separate val folder
val_results = model(val_images_path)

gt_labels, pred_labels = [], []
names = model.names

for result in val_results:
    img_path = result.path
    lbl_path = img_path.replace("/images/", "/labels/").rsplit(".",1)[0] + ".txt"
    if os.path.exists(lbl_path):
        with open(lbl_path) as f:
            gt_labels.extend(int(line.split()[0]) for line in f)
    pred_labels.extend(int(cls) for cls in result.boxes.cls)

if gt_labels and pred_labels:
    cm = confusion_matrix(gt_labels, pred_labels, labels=range(len(names)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=names)
    disp.plot(include_values=True, xticks_rotation=45, cmap="Blues")
    plt.title("BurnGain: Confusion Matrix")
    plt.tight_layout()
    plt.savefig("burngain_confusion_matrix.png")
    plt.show()
    print("✅ Confusion matrix saved as burngain_confusion_matrix.png")
else:
    print("⚠️ Could not generate confusion matrix — no predictions or labels found.")
