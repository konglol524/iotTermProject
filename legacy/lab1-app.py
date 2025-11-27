from flask import Flask, Response, render_template, render_template_string
import cv2
import time
import numpy as np
from picamera2 import Picamera2

app = Flask(__name__)

font = cv2.FONT_HERSHEY_SIMPLEX

def process_img(img):
    # Load face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
        'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        # Apply box filter to blur the face region
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        box_kernel = np.ones((25, 25), np.float32) / (25 * 25)
        img[y:y + h, x:x + w, :] = cv2.filter2D(img[y:y + h, x:x + w, :], -1, box_kernel)

    return img

def generate_frames():
    prev_time = time.time()
    fps = 0

    # Initialize Raspberry Pi camera
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
    picam2.start()

    while True:
        frame = picam2.capture_array()
        # Get image dimensions from the frame
        height, width, channels = frame.shape

        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        fps = str(int(fps))
        
        # calling the function that detect and blur faces.
        frame = process_img(frame)

        text = f"{width}x{height} | fps:{fps}"
        cv2.putText(frame, text, (7, 70), font, 1, (100, 255, 0), 3, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    html_code = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Raspberry Pi Camera Stream</title>
    </head>
    <body>
        <h1>Live Camera Stream</h1>
        <img src="{{ url_for('video_feed') }}" width="640" height="480">
    </body>
    </html>
    """
    return render_template_string(html_code)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
