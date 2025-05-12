# train_model.py

from ultralytics import YOLO

# --- Step 1: Train YOLOv8 Model ---
model = YOLO("yolov8n.pt")  # Change to yolov8s.pt for better accuracy

model.train(
    data="data.yaml",         # Path to your dataset config
    epochs=2,
    imgsz=256,
    batch=4,
    name="BurnGain"           # Output folder name under runs/detect/
)

print("Training completed. Metrics saved to 'runs/detect/BurnGain/results.csv'")
