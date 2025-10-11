from flask import Flask, Response, render_template
import cv2
from ultralytics import YOLO
from flask_cors import CORS
from yolo_model_train import *
import os

class YOLOfeed:
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)
        # Carregar modelo YOLOv8
        self.model = YOLO(YOLOModelTrain().get_latest_model())
        self.cap = cv2.VideoCapture(0)
        self._register_routes()
        self._register_feed()
        self.allowed_classes = []

    def gen_frames(self):
        while True:
            success, frame = self.cap.read()
            if not success:
                break
            
            # Rodar detecção no frame
            results = self.model(frame, verbose=False)
            result = results[0]
            boxes = result.boxes
            names = self.model.names
            # # Substitui as caixas do resultado apenas pelas filtradas
            mask = [self.model.names[int(b.cls[0])] in self.allowed_classes for b in boxes]
            result.boxes = boxes[mask]
            annotated_frame = result.plot() #results[0].plot()

            # Codificar frame para JPEG
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()

            # Montar resposta no formato de streaming
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    def _register_routes(self):
        @self.app.route('/')
        def index():
            return render_template('index.html')

    def _register_feed(self):
        @self.app.route('/video_feed')
        def video_feed():
            return Response(self.gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



# if __name__ == "__main__":
#     YOLOfeed(YOLOModelTrain()).app.run(host="0.0.0.0", port=5000, debug=False)