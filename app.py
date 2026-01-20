from flask import Flask, render_template, Response
import cv2
import time
import os
import threading

app = Flask(__name__)

# Folders for media
IMAGES_FOLDER = "images"
VIDEOS_FOLDER = "videos"
os.makedirs(IMAGES_FOLDER, exist_ok=True)
os.makedirs(VIDEOS_FOLDER, exist_ok=True)

# Camera setup
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Motion detection
motion_threshold = 50000
prev_frame = None
lock = threading.Lock()

# Video recording parameters
RECORD_DURATION = 15  # seconds
FPS = 20

def detect_motion(frame):
    global prev_frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if prev_frame is None:
        prev_frame = gray
        return False

    diff = cv2.absdiff(prev_frame, gray)
    prev_frame = gray
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    motion_level = cv2.countNonZero(thresh)

    return motion_level > motion_threshold

def record_video(frames, width, height):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    video_file = os.path.join(VIDEOS_FOLDER, f"{timestamp}.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_file, fourcc, FPS, (width, height))

    for frame in frames:
        out.write(frame)
    out.release()
    print(f"Saved video: {video_file}")

def gen_frames():
    frame_buffer = []
    recording = False
    frames_needed = RECORD_DURATION * FPS

    while True:
        success, frame = camera.read()
        if not success:
            continue

        if detect_motion(frame):
            # Save image
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            image_file = os.path.join(IMAGES_FOLDER, f"{timestamp}.jpg")
            with lock:
                cv2.imwrite(image_file, frame)
                print(f"Motion detected! Saved image: {image_file}")

            # Start video recording
            recording = True
            frame_buffer.append(frame)
        elif recording:
            frame_buffer.append(frame)
            if len(frame_buffer) >= frames_needed:
                with lock:
                    record_video(frame_buffer, 640, 480)
                frame_buffer = []
                recording = False

        # Stream video
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)
