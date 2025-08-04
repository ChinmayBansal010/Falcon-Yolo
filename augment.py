import os
import cv2
import random
import albumentations as A
from glob import glob
from collections import defaultdict

label_dir = 'data/train/labels'
image_dir = 'data/train/images'
class_names = ['FireExtinguisher', 'ToolBox', 'OxygenTank']
target_per_class = 500

# --- Define the image extension used in your dataset ---
IMAGE_EXTENSION = '.png'

# Albumentations pipeline for object detection
augment = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(limit=10, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomScale(scale_limit=0.1, p=0.5)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# Count images per class
class_counts = defaultdict(list)
for label_path in glob(os.path.join(label_dir, '*.txt')):
    if '_aug' in os.path.basename(label_path): continue # Skip already augmented files
    with open(label_path, 'r') as f:
        labels = f.readlines()
    if not labels: continue
    classes_in_file = set([int(line.split()[0]) for line in labels if line.strip()])
    for cls_id in classes_in_file:
        class_counts[cls_id].append(label_path)

# Augment images to balance all classes
for cls_id, label_files in class_counts.items():
    count = len(label_files)
    to_add = target_per_class - count
    if to_add <= 0:
        continue
    print(f"[{class_names[cls_id]}] Augmenting {to_add} images...")

    for i in range(to_add):
        label_path = random.choice(label_files)
        
        # --- MODIFIED: Use the IMAGE_EXTENSION constant ---
        img_name = os.path.basename(label_path).replace('.txt', IMAGE_EXTENSION)
        img_path = os.path.join(image_dir, img_name)

        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Read original bboxes
        bboxes = []
        class_labels = []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                class_labels.append(int(parts[0]))
                bboxes.append([float(x) for x in parts[1:]])
        
        # Apply augmentation to image and bboxes
        augmented = augment(image=img, bboxes=bboxes, class_labels=class_labels)
        
        if not augmented['bboxes']:
            continue
        
        aug_img_bgr = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)
        
        # --- MODIFIED: Use the IMAGE_EXTENSION constant for saving ---
        new_name_base = os.path.basename(label_path).replace('.txt', f'_aug{i}')
        new_img_path = os.path.join(image_dir, f"{new_name_base}{IMAGE_EXTENSION}")
        new_lbl_path = os.path.join(label_dir, f"{new_name_base}.txt")

        cv2.imwrite(new_img_path, aug_img_bgr)
        
        with open(new_lbl_path, 'w') as f:
            for new_cls, new_bbox in zip(augmented['class_labels'], augmented['bboxes']):
                x_center, y_center, width, height = new_bbox
                f.write(f"{new_cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")