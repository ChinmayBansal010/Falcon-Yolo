# Object Detection with YOLOv8

This project provides a comprehensive solution for object detection using the YOLOv8 model, encompassing data augmentation, model training, prediction, and visualization tools.

## Table of Contents
- [Environment Setup](#environment-setup)
- [Project Structure](#project-structure)
- [Step-by-Step Instructions to Run and Test Your Model](#step-by-step-instructions-to-run-and-test-your-model)
  - [1. Prepare your dataset](#1-prepare-your-dataset)
  - [2. Data Augmentation (Optional but Recommended for Class Balancing)](#2-data-augmentation-optional-but-recommended-for-class-balancing)
  - [3. Train the Model](#3-train-the-model)
  - [4. Make Predictions](#4-make-predictions)
  - [5. Visualize Dataset Annotations](#5-visualize-dataset-annotations)
  - [6. Run Live Webcam Detection (Web Application)](#6-run-live-webcam-detection-web-application)
- [How to Reproduce Final Results](#how-to-reproduce-final-results)
- [Notes on Expected Outputs and How to Interpret Them](#notes-on-expected-outputs-and-how-to-interpret-them)

## Environment Setup

To get started, simply run the provided batch file to create and set up the Conda environment named 'EDU'.

1.  **Run the setup script:**
    ```bash
    .\setup_env.bat
    ```

This script will create a new Conda environment named 'EDU', install all necessary dependencies (including PyTorch with CUDA support and Ultralytics), and activate the environment for you.

Install the python libraries using the below command for any import errors

2. **For Dependency Issues.**
    ```bash
    pip install -r requirements.txt
    ```

## Project Structure

The project is organized into the following directories and files:

* `data/`: Contains the images and labels for training, validation, and testing.
    * `train/images/`: Images for training the model.
    * `train/labels/`: YOLO format label files corresponding to training images.
    * `val/images/`: Images for validating the model during training.
    * `val/labels/`: YOLO format label files corresponding to validation images.
    * `test/images/`: Images for final model evaluation and prediction.
* `classes.txt`: A plain text file listing the names of the object classes, one per line.   
* `yolo_params.yaml`: Configuration file for YOLO, specifying dataset paths and class information.
* `augment.py`: Python script for performing data augmentation to balance class distribution.
* `train.py`: Python script to initiate and manage the YOLOv8 model training process.
* `predict.py`: Python script for running inference on test images, saving predictions, and evaluating performance.
* `visualize.py`: Python script to visualize the bounding box annotations on dataset images.
* `app.py`: **(UPDATED)** Python script for a Flask-based web application for real-time object detection using a webcam.
* `templates/`: Directory to hold HTML templates for the web application.
    * `index.html`: The main page for the web application.
* `runs/`: Directory automatically created by Ultralytics to store training and validation results.
* `predictions/`: Directory where predicted images with bounding boxes and their corresponding label files are saved.
* `logs/`: Directory for storing training and prediction logs.
* `setup_env.bat`, `create_env.bat`, `install_packages.bat`: These are the setup files for creating and installing the Conda environment and dependencies in folder `ENV_SETUP/`.

## Step-by-Step Instructions to Run and Test Your Model

Follow these steps sequentially to set up, train, and test your object detection model.

### 1. Prepare your dataset

Organize your dataset according to the `Project Structure` described above. Ensure your images are in `.jpg`, `.jpeg`, or `.png` format and their corresponding labels are in YOLO format (`.txt` files) with the same base filename.

Create a `yolo_params.yaml` file in the root directory of your project. This file will tell YOLO where to find your data and what classes it should detect. An example structure is shown below; adjust paths and class names as per your dataset:

```yaml
train: data/train/images
val: data/val/images
test: data/test/images

nc: 3  # Number of classes
names: ['FireExtinguisher', 'ToolBox', 'OxygenTank'] # Class names
```

### 2. Data Augmentation (Optional but Recommended for Class Balancing)

To address potential class imbalances in your training data, you can use the `augment.py` script. This script augments images to achieve a target count per class (default is 500) by applying transformations such as random brightness/contrast, rotation, horizontal flips, and scaling.

Run the augmentation script:

```bash
python augment.py
```

Augmented images and their corresponding label files will be saved in `data/train/images/` and `data/train/labels/` respectively, with `_augX` appended to their filenames.

### 3. Train the Model

The `train.py` script is used to train the YOLOv8 model. You can customize various training parameters using command-line arguments.

To start training, execute:

```bash
python train.py --epochs 100 --batch 8 --imgsz 512 --weights yolov8s.pt --save
```

* `--epochs`: Specifies the number of training epochs (default: 100).
* `--batch`: Sets the batch size for training (default: 8).
* `--imgsz`: Defines the input image size for the model (default: 512).
* `--weights`: Path to initial model weights. `yolov8s.pt` is a good starting point for a small YOLOv8 model.
* `--save`: Flag to save the `best.pt` and `last.pt` model weights after training.

Training progress and details will be logged in `logs/train.log`. The trained models will be stored in `runs/detect/train/weights/`.

### 4. Make Predictions

Once your model is trained, use `predict.py` to perform inference on your test dataset.

```bash
python predict.py
```

The script will:
* Prompt you to select a training run if multiple exist in `runs/detect/`.
* Load the `best.pt` weights from the selected training run.
* Process all images found in the `test` directory specified in `yolo_params.yaml`.
* Save predicted images with detected bounding boxes to `predictions/images/`.
* Save the corresponding YOLO format label files for the predictions to `predictions/labels/`.
* Perform a validation step on the test set and record performance metrics (mAP, precision, recall) in `logs/prediction_log.txt`.

### 5. Visualize Dataset Annotations

To visually inspect the ground truth annotations on your dataset images (training or validation), use the `visualize.py` script.

```bash
python visualize.py
```

Control the visualization using the following keys:
* `d`: Navigate to the next image.
* `a`: Navigate to the previous image.
* `t`: Switch to visualizing the training set annotations.
* `v`: Switch to visualizing the validation set annotations.
* `q` or `Esc`: Quit the visualizer.

### 6. Run Live Webcam Detection (Web Application)

For real-time object detection using your webcam, execute the new `app.py` script. This requires a `best.pt` model file to be present in `runs/detect/train/weights/`.

1.  **Ensure you have Flask installed** (see "Environment Setup").
2.  **Run the Flask application:**
    ```bash
    python app.py
    ```
3.  **Open your browser** and navigate to `http://127.0.0.1:5000`. You will see the live webcam feed with object detections.

The web page provides a "Start/Stop" button to control the video stream and displays confidence scores for each detection.

## How to Reproduce Final Results

To reproduce the exact trained model and prediction results:

1.  **Ensure Data is Ready**: Confirm your dataset is correctly organized in the `data/train`, `data/val`, and `data/test` directories, and `yolo_params.yaml` is configured as described in "1. Prepare your dataset."
2.  **Run Augmentation (Optional)**: If the original results were generated with data augmentation, run `python augment.py` to apply the same augmentation process.
3.  **Train the Model**: Execute the training script with the same parameters used for the original results. For example:
    ```bash
    python train.py --epochs 100 --batch 8 --imgsz 512 --weights yolov8s.pt --save
    ```
    The `best.pt` model, essential for prediction, will be saved in `runs/detect/train/weights/`.
4.  **Run Prediction**: Execute the prediction script:
    ```bash
    python predict.py
    ```
    This will generate the predicted images and labels, and an evaluation log (`logs/prediction_log.txt`) based on the trained model.

## Notes on Expected Outputs and How to Interpret Them

* **`augment.py`**:
    * **Expected Output**: Console messages indicating the number of images augmented per class, e.g., `[FireExtinguisher] Augmenting 300 images...`.
    * **Interpretation**: New images with `_augX` in their filenames will appear in `data/train/images` and corresponding label files in `data/train/labels`. This signifies that the class distribution has been balanced towards the `target_per_class`.

* **`train.py`**:
    * **Expected Output**:
        * Continuous console output displaying training metrics per epoch (loss, mAP, etc.).
        * A detailed log file generated at `logs/train.log`.
        * A `runs/detect/train/` directory containing the `weights/best.pt` (the model with the highest validation performance) and `weights/last.pt` (the model from the final epoch), along with various plots and performance metrics.
    * **Interpretation**:
        * Monitor the `mAP50` and `mAP50-95` values in the console output and `train.log`. These are crucial metrics for object detection, representing mean Average Precision at different Intersection over Union (IoU) thresholds. Higher values indicate better model performance.
        * The generated plots in `runs/detect/train/` provide visual insights into training progress and model performance.

* **`predict.py`**:
    * **Expected Output**:
        * Console messages detailing the prediction process.
        * A prompt to select a training run if multiple exist in `runs/detect/`.
        * Output images with detected bounding boxes saved in `predictions/images/`.
        * Predicted YOLO-format label files saved in `predictions/labels/`.
        * A comprehensive evaluation log appended to `logs/prediction_log.txt` and printed to console.
    * **Interpretation**:
        * The images in `predictions/images/` allow for visual inspection of the model's performance on individual test images.
        * The `logs/prediction_log.txt` provides quantitative metrics (mAP, Precision, Recall) on the test set, indicating how well the model generalizes to unseen data.

* **`visualize.py`**:
    * **Expected Output**: A new OpenCV window displaying images from the dataset with their ground truth bounding box annotations and class labels.
    * **Interpretation**: Allows for visual verification of the dataset's annotations.

* **`app.py` and `index.html`**:
    * **Expected Output**:
        * The terminal will show the Flask server starting up, e.g., `* Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)`.
        * When you visit the URL in your browser, you will see a web page displaying the live webcam feed with real-time bounding boxes and class labels overlaid.
    * **Interpretation**: This demonstrates the model's ability to perform live inference within a web-based, interactive application.
