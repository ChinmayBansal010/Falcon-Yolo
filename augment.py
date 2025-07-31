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

# Albumentations pipeline
augment = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(limit=10, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomScale(scale_limit=0.1, p=0.5)
])

# Count images per class
class_counts = defaultdict(list)

for label_path in glob(os.path.join(label_dir, '*.txt')):
    with open(label_path, 'r') as f:
        labels = f.readlines()
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
        img_name = os.path.basename(label_path).replace('.txt', '.jpg')
        img_path = os.path.join(image_dir, img_name)

        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        aug = augment(image=img)['image']

        # Save new files
        new_name = img_name.replace('.jpg', f'_aug{i}.jpg')
        new_img_path = os.path.join(image_dir, new_name)
        new_lbl_path = os.path.join(label_dir, new_name.replace('.jpg', '.txt'))

        cv2.imwrite(new_img_path, aug)
        os.system(f'cp {label_path} {new_lbl_path}')
