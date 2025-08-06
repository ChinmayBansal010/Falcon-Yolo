import argparse
import torch
import logging
import os
import shutil
import yaml
from ultralytics import YOLO

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "train.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

TUNE_RESULTS_DIR = os.path.join("runs", "detect", "tune")
TUNED_PARAMS_FILE = os.path.join(TUNE_RESULTS_DIR, "best_hyperparameters.yaml")

HYPERPARAMETERS = {
    'optimizer': 'AdamW',
    'lr0': 0.001,
    'lrf': 0.1,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 5.0,
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.05,
    'box': 7.5,
    'cls': 0.5,
    'dfl': 1.5,
    'cos_lr': True,
    'patience': 50,
    'label_smoothing': 0.0,
    'mosaic': 1.0,
    'mixup': 0.1,
    'copy_paste': 0.1,
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    'degrees': 0.0,
    'translate': 0.1,
    'scale': 0.5,
    'shear': 0.0,
    'perspective': 0.0,
    'flipud': 0.0,
    'fliplr': 0.5,
    'close_mosaic': 20,
    'amp': True,
    'save': True,
    'verbose': False
}

best_map50 = 0.0

def on_train_epoch_end(trainer):
    """
    An Ultralytics callback function that executes at the end of each training epoch.
    It logs the current epoch's mAP@0.5 and loss. It also checks if the current
    mAP@0.5 is the best seen so far and, if so, saves a copy of the latest model
    weights as 'best_map50.pt'.
    """
    global best_map50
    metrics = trainer.metrics or {}
    epoch = trainer.epoch + 1
    total_epochs = trainer.epochs
    map50 = metrics.get('metrics/mAP50(B)', 0)

    logging.info(
        f"Epoch {epoch}/{total_epochs} | "
        f"mAP@0.5: {map50:.4f}, "
        f"Loss: {metrics.get('train/loss', -1):.4f}"
    )

    if map50 > best_map50:
        best_map50 = map50
        logging.info(f"🚀 New best mAP@0.5: {best_map50:.4f}! Saving model...")
        source_path = os.path.join(trainer.save_dir, 'weights', 'last.pt')
        dest_path = os.path.join(trainer.save_dir, 'weights', 'best_map50.pt')
        shutil.copy(source_path, dest_path)

def main():
    """
    The main entry point for the script. It performs the following steps:
    1.  Parses command-line arguments for training configuration like data path,
        epochs, batch size, etc.
    2.  Initializes a set of default hyperparameters.
    3.  If not in tuning mode, it checks for a 'best_hyperparameters.yaml' file
        and loads it to override the defaults.
    4.  Initializes a YOLO model and attaches the 'on_train_epoch_end' callback.
    5.  If the '--tune' flag is used, it runs YOLO's built-in hyperparameter
        tuner and saves the best discovered parameters to a YAML file.
    6.  Proceeds with the main model training using the determined hyperparameters.
    7.  After training, it evaluates the best-performing model on the test set.
    """
    parser = argparse.ArgumentParser(description="Train a YOLOv8 object detection model.")
    parser.add_argument('--data', type=str, default='yolo_params.yaml', help="Path to the dataset configuration file.")
    parser.add_argument('--weights', type=str, default='yolov8l.pt')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--tune', action='store_true', help="Run hyperparameter tuning and save results.")
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    hyp = HYPERPARAMETERS.copy()
    
    if not args.tune and os.path.exists(TUNED_PARAMS_FILE):
        logging.info(f"Found {TUNED_PARAMS_FILE}. Loading optimized hyperparameters.")
        with open(TUNED_PARAMS_FILE, 'r') as f:
            tuned_hyp = yaml.safe_load(f)
            if tuned_hyp:
                hyp.update(tuned_hyp)
    
    global best_map50
    best_map50 = 0.0

    model = YOLO(args.weights)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)

    if args.tune:
        logging.info("🚀 Starting hyperparameter tuning...")
        tune_results = model.tune(
            data=args.data, epochs=30, iterations=20, optimizer='AdamW',
            project=os.path.dirname(TUNE_RESULTS_DIR),
            name=os.path.basename(TUNE_RESULTS_DIR),
            val=True, batch=args.batch,
            imgsz=args.imgsz, device=args.device, workers=args.workers,
            patience=10
        )
        
        best_hyp = tune_results.best_hyp
        logging.info(f"Tuning complete. Best hyperparameters found: {best_hyp}")
        
        os.makedirs(os.path.dirname(TUNED_PARAMS_FILE), exist_ok=True)
        
        with open(TUNED_PARAMS_FILE, 'w') as f:
            yaml.dump(best_hyp, f, indent=4, sort_keys=False)
        logging.info(f"✅ Saved best hyperparameters to {TUNED_PARAMS_FILE}")

        hyp.update(best_hyp)

    logging.info(f"Starting main training on {args.device}")
    logging.info(f"Using Hyperparameters: {hyp}")
    
    results = model.train(
        data=args.data, epochs=args.epochs, batch=args.batch, imgsz=args.imgsz,
        device=args.device, project="runs/detect", name="train", exist_ok=True,
        resume=args.resume, workers=args.workers, cache='disk',
        **hyp 
    )

    logging.info("Training complete. Evaluating on test set...")
    try:
        best_model_path = os.path.join(results.save_dir, 'weights', 'best_map50.pt')
        if not os.path.exists(best_model_path):
            best_model_path = os.path.join(results.save_dir, 'weights', 'best.pt')

        best_model = YOLO(best_model_path)
        metrics = best_model.val(data=args.data, split='test', imgsz=1280, augment=True)
        logging.info(f"Test mAP50: {metrics.box.map50:.4f}, mAP50-95: {metrics.box.map:.4f}")
    except Exception as e:
        logging.warning(f"Test evaluation failed: {e}")

if __name__ == '__main__':
    main()