from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
import os
import torch
import shutil

def get_latest_model(models_dir="model_history"):
    subdirs = [os.path.join(models_dir, d) for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
    if not subdirs:
        return "best.pt"
    latest = max(subdirs, key=os.path.getmtime)
    file_path = os.path.join(latest, "weights", "best.pt")
    return file_path

def clean_old_trainings(models_dir="model_history", max_trainings=50):
    """
    Mantém apenas os últimos `max_trainings` diretórios dentro de models_dir.
    Remove os mais antigos até restarem exatamente `max_trainings`.
    """
    subdirs = [os.path.join(models_dir, d) for d in os.listdir(models_dir)
               if os.path.isdir(os.path.join(models_dir, d))]

    if len(subdirs) <= max_trainings:
        return

    subdirs.sort(key=os.path.getmtime)

    excess = len(subdirs) - max_trainings

    for i in range(excess):
        old_dir = subdirs[i]
        print(f"[INFO] Apagando treino antigo: {old_dir}")
        shutil.rmtree(old_dir, ignore_errors=True)

def train_model():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    project_dir = ".\\model_history"
    exp_name = f"epi_yolov8_{timestamp}"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    device_str = ""
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if torch.cuda.is_available():
        device_str = "cuda"
    else:
        device_str = "cpu"
    device = torch.device(device=device_str)
    model = YOLO(get_latest_model())
    model.to(device)
    model.train(
        data=f"{script_dir}\\data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        name=exp_name,
        project=project_dir,
        device=0
    )



