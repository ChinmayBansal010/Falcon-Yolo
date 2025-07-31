import argparse
import torch
import logging
import os
from ultralytics import YOLO

# Logging setup
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "train.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Default hyperparameters
DEFAULTS = {
    'epochs': 100,
    'mosaic': 0.7,
    'optimizer': 'AdamW',
    'momentum': 0.937,
    'lr0': 0.001,
    'lrf': 0.01,
    'batch': 8,
    'imgsz': 512,
    'single_cls': False,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'data': 'yolo_params.yaml',
    'weights': 'yolov8s.pt'
}

def main():
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()

    logging.info(f"Training started on {args.device} with weights {args.weights}")
    model = YOLO(args.weights)

    # Train the model and collect results
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
        mosaic=args.mosaic,
        mixup=0.15,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        flipud=0.0,
        fliplr=0.5,
        workers=4,
        patience=20,
        warmup_epochs=3.0,
        warmup_bias_lr=0.1,
        warmup_momentum=0.8,
        close_mosaic=10,
        verbose=False,
        val=True,
        save=args.save,
        project="runs/detect",
        name="train",
        exist_ok=True
    )

    logging.info("Training complete. Evaluating on test set...")

    # You can extract and log metrics from the results object after training
    try:
        metrics = model.val(data=args.data, split='test')
        logging.info(f"Test mAP50: {metrics.box.map50:.4f}, mAP50-95: {metrics.box.map:.4f}")
    except Exception as e:
        logging.warning(f"Test evaluation failed: {e}")

if __name__ == '__main__':
    main()
