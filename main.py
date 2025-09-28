from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from ultralytics.utils import SETTINGS
from yolo_interface import *

current_path = os.path.dirname(os.path.abspath(__file__))
SETTINGS.update({
    "datasets_dir": current_path
})
# 1. Carregar modelo YOLOv8
# model_path = get_latest_model()
if __name__ == "__main__":
    io = YOLOInterface("", "").root.mainloop()

# # 3. Rodar a detecção
# results = model("amostra.mov", save=True)

# # 4. Mostrar resultados no terminal
# for r in results:
#     for box in r.boxes:
#         cls = int(box.cls[0])  # classe detectada
#         label = model.names[cls]  # nome da classe
#         conf = float(box.conf[0])  # confiança
#         print(f"Detectado: {label} - Confiança: {conf:.2f}")
