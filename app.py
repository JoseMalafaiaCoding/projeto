from flask import Flask, Response, render_template
import cv2
from ultralytics import YOLO
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)
# Carregar modelo YOLOv8
model = YOLO("best1.pt")

# Abrir webcam
cap = cv2.VideoCapture(0)

def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Rodar detecção no frame
        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()

        # Codificar frame para JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()

        # Montar resposta no formato de streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)