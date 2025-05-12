import yaml

# Load class names from classes.txt
with open("yolo_dataset/classes.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Build data.yaml dictionary
data_yaml = {
    'train': 'yolo_dataset/images/train',
    'val': 'yolo_dataset/images/train',  # use same folder if no val split
    'nc': len(class_names),
    'names': class_names
}

# Save data.yaml
with open("data.yaml", "w") as f:
    yaml.dump(data_yaml, f, sort_keys=False)

print("data.yaml created!")
