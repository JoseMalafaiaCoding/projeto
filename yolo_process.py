import os
from ultralytics import YOLO
from datetime import datetime
import pytz
import cv2

class YOLOProcess:
    def __init__(self, model_path, input_dir, output_dir, imgsz=640, device=0):
        """
        Classe para processar imagens com YOLOv8.

        :param model_path: Caminho para o arquivo .pt do modelo treinado
        :param input_dir: Pasta com as imagens de entrada
        :param output_dir: Pasta onde salvar as imagens processadas
        :param imgsz: Tamanho da imagem para a inferência
        :param device: Dispositivo (0=GPU, "cpu"=CPU)
        """
        self.model = YOLO(model_path)
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.imgsz = imgsz
        self.device = device
        self.allowed_classes = [
            "earplugs"
            ,"half_mask"
            ,"hard_hat"
            ,"no_earplugs"
            ,"no_half_mask"
            ,"no_hard_hat"
            ,"no_safety_boots"
            ,"no_safety_glasses"
            ,"no_safety_gloves"
            ,"no_safety_vest"
            ,"safety_boots"
            ,"safety_glasses"
            ,"safety_gloves"
            ,"safety_vest"]

        # Criar a pasta de saída se não existir
        os.makedirs(self.output_dir, exist_ok=True)

    def process_images(self):
        """
        Processa todas as imagens da pasta de entrada com YOLOv8
        e salva os resultados na pasta de saída.
        """
        # lista imagens válidas
        valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
        images = [f for f in os.listdir(self.input_dir) if f.lower().endswith(valid_ext)]
        if not images:
            return f"[INFO] Nenhuma imagem encontrada em {self.input_dir}"
        current_timestamp = datetime.now(tz=pytz.timezone("America/Sao_Paulo")).strftime("%Y_%m_%d_%H%M%S")
        results_path = f"results_{current_timestamp}"
        os.mkdir(f"datasets/processed/{results_path}")
        for img_name in images:
            img_path = os.path.join(self.input_dir, img_name)
            print(f"[INFO] Processando: {img_path}")
            # rodar a inferência

            results = self.model.predict(
                source=img_path,
                imgsz=self.imgsz,
                device=self.device,
                save=False,             # salva resultados
                save_txt=False,        # salvar labels em txt? (False = só imagem)
                project=self.output_dir,
                name=results_path,        # subpasta para resultados
                exist_ok=True          # sobrescreve se já existir
            )

            result = results[0]
            boxes = result.boxes
            mask = [self.model.names[int(b.cls[0])] in self.allowed_classes for b in boxes]
            result.boxes = boxes[mask]
            annotated_img = result.plot()
            ret, buffer = cv2.imencode('.jpg', annotated_img)
            print(cv2.imwrite(os.path.join("datasets/processed", results_path, img_name), annotated_img))
            print(os.path.join(".", results_path, img_name))

        return f"[INFO] Processamento concluído! Resultados em: {os.path.join(self.output_dir, results_path)}"