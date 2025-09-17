from flask import Flask, request
from flask_cors import CORS
import base64
import cv2
import numpy as np
from datetime import datetime
import pytz
from pathlib import Path

app = Flask(__name__)
CORS(app)
@app.route('/processar', methods=['POST'])
def processar():
    data = request.json
    if not data or 'image' not in data:
        return "Nenhuma imagem recebida", 400

    # imagem vem em formato base64 "data:image/png;base64,...."
    img_data = data['image'].split(",")[1]
    img_bytes = base64.b64decode(img_data)

    # converter para array numpy
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    Path("capturas").mkdir(parents=True, exist_ok=True)
    filepath = "C:\\Users\\junin\\OneDrive\\Desktop\\capturas\\"
    filename = datetime.now(pytz.timezone('America/Sao_Paulo')).strftime("captura_%Y%m%d_%H%M%S.jpg")
    # mostrar a imagem em uma janela
    cv2.imwrite(filepath+filename, img)

    return "Imagem recebida e exibida no servidor!", 200

if __name__ == "__main__":
    app.run(debug=True)