import argparse
import torch
import logging
import os
import shutil
from ultralytics import YOLO

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "train.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

DEFAULTS = {
    'epochs': 200,
    'mosaic': 1.0,
    'optimizer': 'AdamW',
    'momentum': 0.937,
    'lr0': 0.001,
    'lrf': 0.1,
    'batch': 8,
    'imgsz': 640,
    'single_cls': False,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'data': 'yolo_params.yaml',
    'weights': 'yolov8m.pt',
    'save' : True
}

best_map50 = 0.0

def on_train_epoch_end(trainer):
    global best_map50
    metrics = trainer.metrics or {}
    epoch = trainer.epoch + 1
    total_epochs = trainer.epochs
    map50 = metrics.get('metrics/mAP50(B)', 0)
    map5095 = metrics.get('metrics/mAP50-95(B)', -1)

    logging.info(
        f"Epoch {epoch}/{total_epochs} | "
        f"mAP@0.5: {map50:.4f}, "
        f"mAP@0.5-95: {map5095:.4f}, "
        f"Precision: {metrics.get('metrics/precision(B)', -1):.4f}, "
        f"Recall: {metrics.get('metrics/recall(B)', -1):.4f}, "
        f"Loss: {metrics.get('train/loss', -1):.4f}"
    )

    if map50 > best_map50:
        best_map50 = map50
        logging.info(f"🚀 New best mAP@0.5: {best_map50:.4f}! Saving model...")
        source_path = os.path.join(trainer.save_dir, 'weights', 'last.pt')
        dest_path = os.path.join(trainer.save_dir, 'weights', 'best_map50.pt')
        shutil.copy(source_path, dest_path)

    if 'per_class_metrics' in metrics:
        logging.info("Per-class metrics:")
        for cls_idx, cls_metrics in metrics['per_class_metrics'].items():
            cls_name = cls_metrics.get('name', f'class_{cls_idx}')
            logging.info(
                f"  Class '{cls_name}': mAP@0.5 {cls_metrics.get('map50', -1):.4f}, "
                f"mAP@all {cls_metrics.get('map', -1):.4f}, "
                f"P {cls_metrics.get('p', -1):.4f}, R {cls_metrics.get('r', -1):.4f}"
            )

def on_train_end(trainer):
    logging.info(f"Training finished after {trainer.epochs} epochs. Best metrics: {trainer.metrics}")

def on_fit_epoch_end(trainer):
    logging.info(f"Fit epoch end — including validation. Metrics: {trainer.metrics}")

def main():
    parser = argparse.ArgumentParser(description="Train a YOLOv8 object detection model.")
    parser.add_argument('--data', type=str, default=DEFAULTS['data'])
    parser.add_argument('--weights', type=str, default=DEFAULTS['weights'])
    parser.add_argument('--epochs', type=int, default=DEFAULTS['epochs'])
    parser.add_argument('--mosaic', type=float, default=DEFAULTS['mosaic'])
    parser.add_argument('--optimizer', type=str, default=DEFAULTS['optimizer'])
    parser.add_argument('--momentum', type=float, default=DEFAULTS['momentum'])
    parser.add_argument('--lr0', type=float, default=DEFAULTS['lr0'])
    parser.add_argument('--lrf', type=float, default=DEFAULTS['lrf'])
    parser.add_argument('--single_cls', action='store_true')
    parser.add_argument('--device', type=str, default=DEFAULTS['device'])
    parser.add_argument('--batch', type=int, default=DEFAULTS['batch'])
    parser.add_argument('--imgsz', type=int, default=DEFAULTS['imgsz'])
    parser.add_argument('--save', action='store_true', default = DEFAULTS['save'])
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    logging.info(f"Training started on {args.device} with weights {args.weights}")
    logging.info(f"Hyperparameters: {args}")

    global best_map50
    best_map50 = 0.0

    model = YOLO(args.weights)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    model.add_callback("on_train_end", on_train_end)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        single_cls=args.single_cls,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        warmup_epochs=5.0,
        warmup_bias_lr=0.05,
        warmup_momentum=0.8,
        weight_decay=0.0005,
        amp=True,
        mosaic=args.mosaic,
        mixup=0.15,
        copy_paste=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        close_mosaic=20,
        label_smoothing=0.0,
        patience=50,
        workers=4,
        verbose=False,
        val=True,
        cache="disk",
        save=args.save,
        project="runs/detect",
        name="train",
        exist_ok=True,
        cos_lr=True,
        resume = args.resume
    )

    logging.info("Training complete. Evaluating on test set...")

    try:
        metrics = model.val(data=args.data, split='val', imgsz=1280, augment=True)
        logging.info(f"Test mAP50: {metrics.box.map50:.4f}, mAP50-95: {metrics.box.map:.4f}")
    except Exception as e:
        logging.warning(f"Test evaluation failed: {e}")

if __name__ == '__main__':
    main()
