from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from model_train import *
from ultralytics.utils import SETTINGS

current_path = os.path.dirname(os.path.abspath(__file__))
SETTINGS.update({
    "datasets_dir": current_path
})
# 1. Carregar modelo YOLOv8
model_path = get_latest_model()
if __name__ == "__main__":
    print(f"Carregando modelo: {model_path}")
    train_model()
# model = YOLO("best1.pt")
# # 2. Ler a imagem (JPEG)
# image_path = "trabalhadores.jpg"
# img = cv2.imread(image_path)

# # 3. Rodar a detecção
# results = model("amostra.mov", save=True)

# # 4. Mostrar resultados no terminal
# for r in results:
#     for box in r.boxes:
#         cls = int(box.cls[0])  # classe detectada
#         label = model.names[cls]  # nome da classe
#         conf = float(box.conf[0])  # confiança
#         print(f"Detectado: {label} - Confiança: {conf:.2f}")

# # 5. Exibir imagem com bounding boxes
# annotated_img = results[0].plot()

# # Converter BGR (cv2) para RGB (matplotlib)
# annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

# plt.figure(figsize=(10, 8))
# plt.imshow(annotated_img)
# plt.axis("off")
# plt.show()