import json
import os
from tqdm import tqdm
from PIL import Image
import shutil

# Input files
json_path = 'dataset/annotations.json'
image_dir = 'dataset/images'

# Output folders
output_dir = 'yolo_dataset'
os.makedirs(f'{output_dir}/images/train', exist_ok=True)
os.makedirs(f'{output_dir}/labels/train', exist_ok=True)

# Load JSON
with open(json_path) as f:
    data = json.load(f)

# Map categories
categories = {cat['id']: cat['name'] for cat in data['categories']}
category_map = {cat_id: i for i, cat_id in enumerate(categories)}

# Save category names
with open(f'{output_dir}/classes.txt', 'w') as f:
    for name in categories.values():
        f.write(name + '\n')

# Image info
image_map = {img['id']: img['file_name'] for img in data['images']}

# Group annotations
annotations_per_image = {}
for ann in data['annotations']:
    img_id = ann['image_id']
    annotations_per_image.setdefault(img_id, []).append(ann)

# Convert annotations
for img_id, anns in tqdm(annotations_per_image.items()):
    img_name = image_map[img_id]
    img_path = os.path.join(image_dir, img_name)

    # Copy image
    shutil.copy(img_path, os.path.join(output_dir, 'images/train', img_name))

    # Get image size
    with Image.open(img_path) as img:
        w, h = img.size

    yolo_lines = []
    for ann in anns:
        cat_id = ann['category_id']
        x, y, bw, bh = ann['bbox']
        x_c = (x + bw / 2) / w
        y_c = (y + bh / 2) / h
        bw /= w
        bh /= h
        class_id = category_map[cat_id]
        yolo_lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}")

    with open(os.path.join(output_dir, 'labels/train', img_name.replace('.jpg', '.txt')), 'w') as f:
        f.write('\n'.join(yolo_lines))
