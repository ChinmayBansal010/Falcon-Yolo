import cv2
import threading
import time
import yaml
import json
import csv
from pathlib import Path
from collections import deque, Counter
from datetime import datetime, timedelta
from ultralytics import YOLO
from flask import Flask, render_template, Response, request, jsonify, stream_with_context

app = Flask(__name__)
lock = threading.Lock()

with open(Path(__file__).parent / "config.yaml", 'r') as f:
    config = yaml.safe_load(f)

video_stream = None
is_streaming = False
detection_confidence = config['initial_confidence']
frame_skip = config['frame_skip']
model = None
model_version = "N/A"
active_tracks = {}
seen_track_ids = set()
detection_history = deque()
log_file_path = Path(__file__).parent / "detection_log.csv"

if config['logging_enabled'] and not log_file_path.exists():
    with open(log_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'track_id', 'class_name', 'confidence'])

def log_detection(track_id, class_name, confidence):
    """Logs a new detection event to the CSV file."""
    if not config['logging_enabled']:
        return
    timestamp = datetime.now().isoformat()
    with open(log_file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, track_id, class_name, confidence])

def get_specific_train_folder(runs_dir):
    """Finds the specific 'train' folder, ignoring others."""
    target_folder = Path(runs_dir) / "train"
    if target_folder.is_dir():
        return target_folder
    return None

def load_model():
    """Loads the YOLO model from the specific 'train' folder."""
    global model, model_version
    train_dir = get_specific_train_folder(config['runs_directory'])
    if not train_dir:
        raise FileNotFoundError("The specific 'train' folder was not found.")
    model_path = train_dir / config['weights_filename']
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found at {model_path}")
    model = YOLO(model_path)
    model_version = train_dir.name
    print(f"Successfully loaded model: {model_path} (Version: {model_version})")

def get_video_stream():
    """Safely initializes and returns the video stream."""
    global video_stream
    with lock:
        if video_stream is None or not video_stream.isOpened():
            video_stream = cv2.VideoCapture(0)
            if not video_stream.isOpened():
                print("Error: Could not open video stream.")
                return None
    return video_stream

def generate_frames():
    """Generates frames for the video stream with object tracking."""
    global active_tracks, seen_track_ids
    frame_count = 0
    cap = get_video_stream()
    if cap is None:
        return

    while is_streaming:
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            continue

        with lock:
            conf = detection_confidence
        
        results = model.track(source=frame, save=False, conf=conf, persist=True, tracker=config['tracker_config'], verbose=False)
        annotated_frame = results[0].plot()

        current_tracks = Counter()
        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
            confs = results[0].boxes.conf.cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()

            for track_id, cls_id, conf_val in zip(track_ids, clss, confs):
                class_name = model.names[int(cls_id)]
                current_tracks[class_name] += 1
                
                if track_id not in seen_track_ids:
                    seen_track_ids.add(track_id)
                    log_detection(track_id, class_name, conf_val)
                    detection_history.append({'timestamp': datetime.now(), 'class': class_name})
        
        with lock:
            active_tracks = dict(current_tracks)

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if ret:
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

def event_stream():
    """Server-Sent Events (SSE) stream for real-time data."""
    initial_load = True
    while True:
        with lock:
            data = {
                "tracks": active_tracks,
                "model_version": model_version
            }
            # On the first yield, also send the class names
            if initial_load and model:
                data["class_names"] = list(model.names.values())
                initial_load = False

        yield f"data: {json.dumps(data)}\n\n"
        time.sleep(1)

def model_updater():
    """Background thread to check for and load new models from the Falcon pipeline."""
    global model, model_version
    drop_path = Path(config['falcon_model_drop_path'])
    version_file = drop_path / "version.json"
    
    while True:
        if version_file.exists():
            try:
                with open(version_file, 'r') as f:
                    new_version_info = yaml.safe_load(f)
                new_model_name = new_version_info.get("model_name")
                new_version_id = new_version_info.get("version_id")
                with lock:
                    is_new = new_version_id != model_version
                if is_new and new_model_name:
                    new_model_path = drop_path / new_model_name
                    if new_model_path.exists():
                        print(f"New model detected: {new_version_id}. Attempting to load...")
                        new_model = YOLO(new_model_path)
                        with lock:
                            model = new_model
                            model_version = new_version_id
                            seen_track_ids.clear()
                        print(f"Successfully hot-swapped to model version: {model_version}")
            except Exception as e:
                print(f"Error during model update: {e}")
        time.sleep(config['updater_check_interval'])

@app.route('/')
def index():
    """Serves the main web page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Streams the webcam feed with detections."""
    global is_streaming
    is_streaming = True
    return Response(stream_with_context(generate_frames()), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/events')
def events():
    """Endpoint for Server-Sent Events."""
    return Response(event_stream(), mimetype='text/event-stream')

@app.route('/chart_data')
def chart_data():
    """Provides data for the live chart."""
    now = datetime.now()
    one_minute_ago = now - timedelta(seconds=60)
    
    while detection_history and detection_history[0]['timestamp'] < one_minute_ago:
        detection_history.popleft()
        
    counts = Counter(d['class'] for d in detection_history)
    labels = list(model.names.values())
    values = [counts.get(label, 0) for label in labels]
    
    return jsonify(labels=labels, values=values)

@app.route('/set_confidence', methods=['POST'])
def set_confidence():
    """Sets the detection confidence threshold."""
    global detection_confidence
    data = request.get_json()
    with lock:
        detection_confidence = float(data.get('confidence', 0.40))
    return jsonify(success=True, confidence=detection_confidence)

if __name__ == '__main__':
    load_model()
    if config['updater_enabled']:
        updater_thread = threading.Thread(target=model_updater, daemon=True)
        updater_thread.start()
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)