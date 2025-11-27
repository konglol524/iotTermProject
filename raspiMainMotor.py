from flask import Flask, Response, render_template_string
import cv2
import time
import datetime
import numpy as np
from picamera2 import Picamera2
import smbus
import os
from gpiozero import AngularServo
import threading

app = Flask(__name__)

# --- Servo Setup ---
SERVO_PIN = 12
servo = AngularServo(SERVO_PIN, min_angle=-90, max_angle=90, 
                     min_pulse_width=0.0005, max_pulse_width=0.0025)

# Servo state tracking
servo_lock = threading.Lock()
lid_open = False
lid_close_time = 0

def open_lid():
    """Open the trash can lid (non-blocking)"""
    global lid_open, lid_close_time
    
    with servo_lock:
        if not lid_open:
            print("[SERVO] Opening lid to 90 degrees")
            servo.angle = 90
            time.sleep(0.5)  # Wait for servo to reach position
            servo.detach()  # Stop sending PWM signal to prevent jitter
            lid_open = True
            lid_close_time = time.time() + 5  # Close after 5 seconds
            print("[SERVO] Lid will close in 5 seconds")

def check_and_close_lid():
    """Check if it's time to close the lid"""
    global lid_open, lid_close_time
    
    if lid_open and time.time() >= lid_close_time:
        with servo_lock:
            print("[SERVO] Closing lid to 0 degrees")
            servo.angle = 0
            time.sleep(0.5)  # Wait for servo to reach position
            servo.detach()  # Stop sending PWM signal to prevent jitter
            lid_open = False

# --- BH1750 Light Sensor Setup ---
DEVICE_ADDRESS = 0x23 
CONTINUOUS_HIGH_RES_MODE = 0x10

try:
    bus = smbus.SMBus(1)
except Exception as e:
    print(f"Warning: I2C Bus not detected. {e}")
    bus = None

font = cv2.FONT_HERSHEY_SIMPLEX

# --- Load MobileNet SSD (Deep Learning Model) ---
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Define paths to the model files (Must be in same directory)
PROTO_PATH = "MobileNetSSD_deploy.prototxt"
MODEL_PATH = "MobileNetSSD_deploy.caffemodel"
NET = None

# Attempt to load the model once on startup
if os.path.exists(PROTO_PATH) and os.path.exists(MODEL_PATH):
    print("Loading MobileNet SSD model...")
    NET = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_PATH)
    # Optimization for RPi:
    NET.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    NET.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
else:
    print("WARNING: Model files not found. Object detection will be SKIPPED.")
    print(f"Please download '{PROTO_PATH}' and '{MODEL_PATH}'")

# Initialize servo to closed position
print("Initializing servo to closed position (0 degrees)")
servo.angle = 0
time.sleep(0.5)  # Wait for servo to reach position
servo.detach()  # Detach to prevent jitter when idle
print("Servo detached - no jitter mode enabled")

# --- Sensor Functions ---
def convert_to_number(data):
    return ((data[1] + (256 * data[0])) / 1.2)

def read_light():
    if bus is None: return None
    try:
        data = bus.read_i2c_block_data(DEVICE_ADDRESS, CONTINUOUS_HIGH_RES_MODE, 2)
        return convert_to_number(data)
    except OSError:
        return None

# --- Image Processing (Object Detection) ---
def detect_objects(img):
    """
    Returns a list of detected objects: [(label, startX, startY, endX, endY), ...]
    """
    detected_results = []
    
    if NET is None:
        return detected_results

    (h, w) = img.shape[:2]
    
    # PERFORMANCE: Resize image smaller before processing (reduces AI load by 75%)
    small_img = cv2.resize(img, (320, 240))  # Smaller = faster
    
    # 1. Prepare the image for the Deep Learning model
    blob = cv2.dnn.blobFromImage(small_img, 0.007843, (300, 300), 127.5)
    
    # 2. Pass blob through the network
    NET.setInput(blob)
    detections = NET.forward()

    # 3. Loop over detections
    found_anything = False
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        idx = int(detections[0, 0, i, 1])
        
        # Only process detections with reasonable confidence
        if confidence > 0.6:
            label_name = CLASSES[idx]
            found_anything = True
            
            # Check if the detected object is a bottle (Index 5)
            if label_name == "bottle":
                # Compute bounding box coordinates (scale back to original size)
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Ensure bounding box is within frame
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)
                
                # Dynamic Label
                label_text = f"Bottle: {confidence * 100:.1f}%"
                detected_results.append((label_text, startX, startY, endX, endY))
                print(f"   >>> CONFIRMED bottle at ({startX}, {startY}) to ({endX}, {endY})")

    return detected_results

def generate_frames():
    prev_time = time.time()
    last_sensor_time = 0
    sensor_interval = 1.0  # Reduced sensor reading frequency
    lux_display = "Lux: Init..."
    lux_color = (0, 255, 255)

    # PERFORMANCE SETTINGS
    frame_count = 0
    SKIP_FRAMES = 15  # Run AI only every 15 frames (~once per second at 15fps)
    last_detections = []

    try:
        picam2 = Picamera2()
        # PERFORMANCE: Reduced resolution for smoother streaming
        config = picam2.create_preview_configuration(
            main={"format": 'XRGB8888', "size": (640, 480)},
            controls={"FrameRate": 15}  # Limit to 15 FPS
        )
        picam2.configure(config)
        picam2.start()
        time.sleep(2)  # Let camera warm up
    except Exception as e:
        print(f"Camera Error: {e}")
        return

    while True:
        try:
            frame = picam2.capture_array()
            
            # Convert 4-channel to 3-channel
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            height, width, _ = frame.shape
            
            current_time = time.time()
            dt = current_time - prev_time
            if dt > 0: fps = 1 / dt
            else: fps = 0
            prev_time = current_time

            # --- Detect bottle (With Frame Skipping) ---
            if frame_count % SKIP_FRAMES == 0:
                last_detections = detect_objects(frame)
                if last_detections:
                    print(f"[ALERT] {len(last_detections)} bottle(s) detected!")
                    # Open the lid when bottle is detected
                    open_lid()
            
            # Check if lid should be closed
            check_and_close_lid()
            
            # Reduced logging frequency
            if frame_count % 60 == 0:
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Stream Active - FPS: {int(fps)}")

            frame_count += 1

            # --- Draw Detections (Every Frame) ---
            for (label, sX, sY, eX, eY) in last_detections:
                cv2.rectangle(frame, (sX, sY), (eX, eY), (0, 255, 0), 2)
                cv2.rectangle(frame, (sX, sY - 15), (sX + 130, sY), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, label, (sX, sY - 5), font, 0.5, (0, 0, 0), 1)

            # --- Read Sensor (Throttled) ---
            if current_time - last_sensor_time > sensor_interval:
                lux = read_light()
                if lux is not None:
                    lux_display = f"Light: {lux:.1f} Lux"
                    if lux < 10: lux_color = (0, 0, 255)
                    elif lux > 500: lux_color = (255, 255, 255)
                    else: lux_color = (0, 255, 255)
                last_sensor_time = current_time

            # --- Overlays ---
            cv2.putText(frame, f"{width}x{height} | FPS:{int(fps)}", (7, 40), font, 1, (100, 255, 0), 2)
            cv2.putText(frame, lux_display, (7, 80), font, 1, lux_color, 2)
            
            # Show lid status
            lid_status = "LID: OPEN" if lid_open else "LID: CLOSED"
            lid_status_color = (0, 255, 0) if lid_open else (100, 100, 100)
            cv2.putText(frame, lid_status, (7, 120), font, 1, lid_status_color, 2)

            # PERFORMANCE: Reduced JPEG quality for faster encoding
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except Exception as e:
            print(f"Stream Loop Error: {e}")
            break

@app.route('/')
def index():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>RPi Smart Trash Can</title>
        <style>
            body { background: #222; color: #ddd; font-family: sans-serif; text-align: center; }
            img { border: 4px solid #555; border-radius: 8px; margin-top: 20px;}
            .info { color: #888; font-size: 14px; margin-top: 10px; }
            .status { color: #0f0; font-size: 18px; font-weight: bold; margin-top: 15px; }
        </style>
    </head>
    <body>
        <h1>🗑️ Smart Trash Can - Bottle Detector</h1>
        <img src="{{ url_for('video_feed') }}" width="640" height="480">
        <p class="info">AI Detection: ~1x per second | Camera: 15 FPS</p>
        <p class="status">Lid opens automatically when bottle detected!</p>
        <p>Ensure <b>MobileNetSSD_deploy</b> files are in the script folder.</p>
    </body>
    </html>
    """)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        print("="*50)
        print("Smart Trash Can System Starting...")
        print("- Servo initialized at 0° (closed)")
        print("- Bottle detection active")
        print("- Lid will open to 90° when bottle detected")
        print("- Lid will close after 5 seconds")
        print("="*50)
        app.run(host='0.0.0.0', port=5000, threaded=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
        servo.angle = 0  # Ensure lid is closed on exit
        time.sleep(0.5)
        servo.detach()  # Stop PWM signal
        print("Lid closed. Goodbye!")