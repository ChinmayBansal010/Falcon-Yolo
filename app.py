import cv2
import threading
import time
from ultralytics import YOLO
from flask import Flask, render_template, Response

# Initialize Flask app
app = Flask(__name__)

# Load the YOLOv8 model
# NOTE: Ensure the 'best.pt' file is present in the specified path after training.
model = YOLO("runs/detect/train/weights/best.pt")
class_names = ["FireExtinguisher", "ToolBox", "OxygenTank"]

# Global variables for video streaming control
video_stream = None
is_streaming = False
detection_confidence = 0.25 # Default confidence threshold
lock = threading.Lock() # Lock to ensure thread-safe access to video capture

def get_video_stream():
    """Returns a VideoCapture object, handling potential errors."""
    global video_stream
    if video_stream is None or not video_stream.isOpened():
        # Try to initialize the camera
        video_stream = cv2.VideoCapture(0)
        if not video_stream.isOpened():
            print("Error: Could not open video stream.")
            return None
    return video_stream

def generate_frames():
    """Generates frames for the video stream with object detection."""
    global is_streaming
    cap = get_video_stream()
    if cap is None:
        return

    while is_streaming:
        # Read a frame from the video stream
        success, frame = cap.read()
        if not success:
            break

        # Perform YOLOv8 object detection on the frame
        results = model.predict(source=frame, save=False, conf=detection_confidence)
        annotated_frame = results[0].plot()

        # Encode the frame as a JPEG image
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue

        # Yield the frame in a byte format suitable for web streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        # Control the frame rate to avoid overloading the browser
        time.sleep(0.03)

    # Release the video capture when streaming stops
    with lock:
        if video_stream is not None:
            video_stream.release()
        video_stream = None

@app.route('/')
def index():
    """Serves the main web page by rendering the index.html template."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Streams the webcam feed with detections."""
    global is_streaming
    is_streaming = True
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_stream')
def stop_stream():
    """Stops the video streaming."""
    global is_streaming
    is_streaming = False
    return "Streaming stopped"

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
