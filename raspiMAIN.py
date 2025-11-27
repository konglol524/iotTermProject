from flask import Flask, Response, render_template_string
import cv2
import time
import numpy as np
from picamera2 import Picamera2
import smbus

app = Flask(__name__)

# --- BH1750 Light Sensor Setup (From File 2) ---
DEVICE_ADDRESS = 0x23 
POWER_ON = 0x01
RESET = 0x07
CONTINUOUS_HIGH_RES_MODE = 0x10

# Initialize I2C (Bus 1 is default for RPi)
try:
    bus = smbus.SMBus(1)
except Exception as e:
    print(f"Warning: I2C Bus not detected. Sensor data will be empty. Error: {e}")
    bus = None

font = cv2.FONT_HERSHEY_SIMPLEX

def convert_to_number(data):
    """Simple function to convert 2 bytes of data into a decimal number"""
    return ((data[1] + (256 * data[0])) / 1.2)

def read_light():
    """Reads light data from the I2C bus"""
    if bus is None:
        return None
    try:
        # Request 2 bytes. This command also triggers a new measurement (resetting the cycle).
        # We must ensure we don't call this faster than the measurement time (~180ms).
        data = bus.read_i2c_block_data(DEVICE_ADDRESS, CONTINUOUS_HIGH_RES_MODE, 2)
        return convert_to_number(data)
    except OSError as e:
        # Prevent crashing the stream if sensor disconnects
        return None

# --- Face Detection & Image Processing (From File 1) ---
def process_img(img):
    # Load face detection model
    # Ensure the xml file is in the same directory or provide full path
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        # Draw rectangle
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Apply box filter to blur the face region
        box_kernel = np.ones((25, 25), np.float32) / (25 * 25)
        # Ensure we don't go out of bounds
        face_roi = img[y:y + h, x:x + w]
        if face_roi.size > 0:
            img[y:y + h, x:x + w, :] = cv2.filter2D(face_roi, -1, box_kernel)

    return img

def generate_frames():
    prev_time = time.time()
    fps = 0

    # Sensor timing variables
    last_sensor_time = 0
    sensor_interval = 0.5  # Read sensor every 0.5 seconds (Sensor takes ~180ms to measure)
    
    # Cache sensor values to display between updates
    lux_display = "Lux: Initializing..."
    lux_color = (0, 255, 255)

    # Initialize Raspberry Pi camera
    try:
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)})
        picam2.configure(config)
        picam2.start()
    except Exception as e:
        print(f"Camera failed to start: {e}")
        return

    while True:
        try:
            frame = picam2.capture_array()
            
            # Get image dimensions
            height, width, channels = frame.shape

            # --- FPS Calculation ---
            current_time = time.time()
            dt = current_time - prev_time
            if dt > 0:
                fps = 1 / dt
            prev_time = current_time
            fps_text = str(int(fps))

            # --- Face Detection ---
            frame = process_img(frame)

            # --- Read Light Sensor (Throttled) ---
            # We only read every 'sensor_interval' seconds to allow the BH1750 to complete its measurement cycle.
            if current_time - last_sensor_time > sensor_interval:
                lux = read_light()
                if lux is not None:
                    lux_display = f"Light: {lux:.1f} Lux"
                    # Visual feedback based on light levels
                    if lux < 10:
                        lux_display += " (Dark)"
                        lux_color = (0, 0, 255) # Red
                    elif lux > 500:
                        lux_display += " (Bright)"
                        lux_color = (255, 255, 255) # White
                    else:
                        lux_color = (0, 255, 255) # Yellow/Cyan
                last_sensor_time = current_time

            # --- Overlay Text ---
            # 1. FPS and Resolution (Top Left)
            info_text = f"{width}x{height} | FPS:{fps_text}"
            cv2.putText(frame, info_text, (7, 40), font, 1, (100, 255, 0), 2, cv2.LINE_AA)

            # 2. Light Level (Below FPS) - Uses cached value
            cv2.putText(frame, lux_display, (7, 80), font, 1, lux_color, 2, cv2.LINE_AA)

            # --- Encode & Output ---
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"Error in loop: {e}")
            break

@app.route('/')
def index():
    html_code = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Raspberry Pi Smart Stream</title>
        <style>
            body { background-color: #222; color: white; font-family: sans-serif; text-align: center; }
            h1 { margin-top: 20px; }
            img { border: 5px solid #444; border-radius: 10px; }
        </style>
    </head>
    <body>
        <h1>Live Camera & Sensor Stream</h1>
        <img src="{{ url_for('video_feed') }}" width="640" height="480">
        <p>Face Detection + BH1750 Light Sensor</p>
    </body>
    </html>
    """
    return render_template_string(html_code)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Ensure I2C pins are correct in your wiring
    # SDA -> GPIO2, SCL -> GPIO3
    print("Starting server...")
    app.run(host='0.0.0.0', port=5000)