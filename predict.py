from ultralytics import YOLO
from pathlib import Path
import cv2
import os
import yaml
import logging

# === LOGGING CONFIGURATION ===
logs_dir = Path(__file__).parent / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)
log_file_path = logs_dir / "prediction_log.txt"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, mode='a'),
        logging.StreamHandler()
    ]
)

# === PREDICT AND SAVE FUNCTION ===
def predict_and_save(model, image_path, output_path, output_path_txt):
    results = model.predict(source = image_path, conf=0.4, iou=0.5, max_det=100, imgsz = 800, augment = True)
    result = results[0]
    img = result.plot()

    cv2.imwrite(str(output_path), img)
    with open(output_path_txt, 'w') as f:
        for box in result.boxes:
            if box.conf < 0.4:
                continue
            cls_id = int(box.cls)
            x_center, y_center, width, height = box.xywh[0].tolist()
            f.write(f"{cls_id} {x_center} {y_center} {width} {height}\n")

# === MAIN EXECUTION ===
if __name__ == '__main__': 
    this_dir = Path(__file__).parent
    os.chdir(this_dir)

    with open(this_dir / 'yolo_params.yaml', 'r') as file:
        data = yaml.safe_load(file)
        if 'test' not in data or data['test'] is None:
            logging.error("Missing 'test' field in yolo_params.yaml")
            exit()
        images_dir = Path(data['test']) / 'images'

    if not images_dir.exists() or not images_dir.is_dir():
        logging.error(f"Images directory {images_dir} not found or not a directory")
        exit()

    if not any(images_dir.iterdir()):
        logging.warning(f"Images directory {images_dir} is empty")
        exit()

    detect_path = this_dir / "runs" / "detect"
    train_folders = [f for f in os.listdir(detect_path) if os.path.isdir(detect_path / f) and f.startswith("train")]

    if not train_folders:
        logging.error("No training folders found")
        exit()

    idx = 0
    if len(train_folders) > 1:
        print("Select the training folder:")
        for i, folder in enumerate(train_folders):
            print(f"{i}: {folder}")
        while True:
            choice = input("Choice: ")
            if choice.isdigit() and int(choice) in range(len(train_folders)):
                idx = int(choice)
                break
            else:
                print("Invalid choice, try again.")

    model_path = detect_path / train_folders[idx] / "weights" / "best_map50.pt"
    model = YOLO(model_path)

    output_dir = this_dir / "predictions"
    images_output_dir = output_dir / 'images'
    labels_output_dir = output_dir / 'labels'
    images_output_dir.mkdir(parents=True, exist_ok=True)
    labels_output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in images_dir.glob('*'):
        if img_path.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
            continue
        output_path_img = images_output_dir / img_path.name
        output_path_txt = labels_output_dir / img_path.with_suffix('.txt').name
        predict_and_save(model, img_path, output_path_img, output_path_txt)

    logging.info(f"Predicted images saved in {images_output_dir}")
    logging.info(f"Bounding box labels saved in {labels_output_dir}")
    logging.info(f"Model parameters loaded from {this_dir / 'yolo_params.yaml'}")

    val_dir = this_dir / "runs" / "detect" / "val"
    metrics = model.val(
        data=str(this_dir / 'yolo_params.yaml'),
        split="test",
        project=str(val_dir.parent), 
        name=val_dir.name,          
        exist_ok=True,
        imgsz=800,
        iou = 0.5,
        augment = True,
        max_det=300,
    )

    map_50 = metrics.box.map50
    map_5095 = metrics.box.map
    precision = metrics.box.p
    recall = metrics.box.r

    log_str = (
        f"\n==== Evaluation Log ====\n"
        f"Model Path: {model_path}\n"
        f"Evaluation Data: {images_dir.resolve()}\n"
        f"mAP@0.5 ....................................... {map_50.mean():.3f} ({map_50.mean()*100:.1f}%)\n"
        f"mAP@0.5:0.95 ................................... {map_5095.mean():.3f}\n"
        f"Precision ...................................... {precision.mean():.3f}\n"
        f"Recall ......................................... {recall.mean():.3f}\n"
        f"Predictions Folder ............................. {output_dir.resolve()}\n"
        f"Validation Results Saved At .................... {val_dir.resolve()}\n"
        f"=========================\n"
    )

    logging.info(log_str.strip())
