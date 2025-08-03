import os
import cv2
import random
import albumentations as A
from glob import glob
from collections import defaultdict
import numpy as np

label_dir = 'data/train/labels'
image_dir = 'data/train/images'
class_names = ['FireExtinguisher', 'ToolBox', 'OxygenTank']
target_per_class = 2000
IMAGE_EXTENSION = '.png'

augment = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Affine(
        scale=(0.85, 1.15),
        translate_percent=(-0.06, 0.06),
        rotate=(-20, 20),
        p=0.8,
        border_mode=cv2.BORDER_CONSTANT
    ),
    A.RandomSizedBBoxSafeCrop(width=416, height=416, erosion_rate=0.2, p=0.4),
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=1.0),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
        A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=1.0),
    ], p=0.85),
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        A.MotionBlur(blur_limit=(3, 7), p=1.0),
        A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
    ], p=0.6),
    A.CoarseDropout(max_holes=10, max_height=40, max_width=40, fill_value=0, p=0.3),
], bbox_params=A.BboxParams(
    format='yolo',
    label_fields=['class_labels'],
    min_area=100,
    min_visibility=0.3
))

os.makedirs(image_dir, exist_ok=True)
os.makedirs(label_dir, exist_ok=True)

print("🔍 Counting original images per class...")
class_counts = defaultdict(list)
for label_path in glob(os.path.join(label_dir, '*.txt')):
    if '_aug' in os.path.basename(label_path):
        continue
    try:
        with open(label_path, 'r') as f:
            labels = f.readlines()
        if not labels:
            continue
        classes_in_file = set([int(line.split()[0]) for line in labels if line.strip()])
        for cls_id in classes_in_file:
            class_counts[cls_id].append(label_path)
    except (ValueError, IndexError):
        print(f"⚠️ Skipping malformed label file: {label_path}")
        continue

print("🚀 Starting augmentation process...")
for cls_id, label_files in class_counts.items():
    if not label_files:
        continue
    
    current_count = len(label_files)
    to_add = target_per_class - current_count
    
    if to_add <= 0:
        print(f"✅ Class '{class_names[cls_id]}' already has {current_count} images. No augmentation needed.")
        continue
        
    print(f"➡️ Class '{class_names[cls_id]}': Found {current_count} images. Generating {to_add} new images...")

    for i in range(to_add):
        if (i + 1) % 50 == 0 or i == to_add - 1:
            print(f"    -> Progress: {i + 1}/{to_add}", end='\r')

        label_path = random.choice(label_files)
        base_name = os.path.basename(label_path).replace('.txt', '')
        img_path = os.path.join(image_dir, f"{base_name}{IMAGE_EXTENSION}")

        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        with open(label_path, 'r') as f:
            lines = f.readlines()

        bboxes, class_labels = [], []
        for line in lines:
            try:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, x, y, bw, bh = map(float, parts)
                bboxes.append([x, y, bw, bh])
                class_labels.append(int(cls))
            except (ValueError, IndexError):
                continue
        
        if not bboxes:
            continue

        try:
            augmented = augment(image=img, bboxes=bboxes, class_labels=class_labels)
            if not augmented['bboxes']:
                continue
        except Exception:
            continue

        new_name_base = f"{base_name}_aug{cls_id}_{i}"
        new_img_path = os.path.join(image_dir, f"{new_name_base}{IMAGE_EXTENSION}")
        new_lbl_path = os.path.join(label_dir, f"{new_name_base}.txt")

        aug_img_bgr = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)
        cv2.imwrite(new_img_path, aug_img_bgr)
        
        with open(new_lbl_path, 'w') as f:
            for cls, bbox in zip(augmented['class_labels'], augmented['bboxes']):
                x, y, w, h = bbox
                f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    print(f"\n✅ Finished augmenting for class '{class_names[cls_id]}'.")

print("\nAugmentation complete! 🎉 Your dataset is now larger and more balanced.")