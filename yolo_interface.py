import tkinter as tk
from tkinter import filedialog, messagebox
import os
import subprocess
from yolo_process import *
from yolo_model_train import *
from yolo_feed import *
from yolo_selector import *

class YOLOInterface:
    def __init__(self, feed_func, train_func, output_dir="./datasets/processed"):
        """
        Classe da interface gráfica para YOLOv8
        :param root: janela Tkinter principal
        :param feed_func: função para abrir feed de vídeo
        :param train_func: função para treinar modelo
        :param output_dir: diretório de saída das imagens processadas
        """
        self.root = tk.Tk()
        self.feed_func = feed_func
        self.train_func = train_func
        self.output_dir = output_dir
        self.feed_obj = YOLOfeed()
        self.model_obj = YOLOModelTrain()
        self.root.title("Interface YOLOv8")
        largura_tela = self.root.winfo_screenwidth()
        altura_tela = self.root.winfo_screenheight()
        x = (largura_tela // 2) - (400 // 2)
        y = (altura_tela // 2) - (200 // 2)
        self.root.geometry(f"400x200+{x}+{y}")

        # Botões principais
        btn_process = tk.Button(self.root, text="Processar Imagens", command=self.open_process_window, width=30, height=2)
        btn_process.pack(pady=10)

        btn_feed = tk.Button(self.root, text="Abrir Feed de Vídeo", command=self.open_feed_window, width=30, height=2)
        btn_feed.pack(pady=10)

        btn_train = tk.Button(self.root, text="Treinar Modelo", command=self.open_train_window, width=30, height=2)
        btn_train.pack(pady=10)

    def open_process_window(self):
        input_dir = filedialog.askdirectory(title="Selecione a pasta com imagens")
        model_path = self.model_obj.get_latest_model()
        if not input_dir:
            return
        try:
            process_result = YOLOProcess(model_path=model_path,input_dir=input_dir,output_dir=".\\datasets\\processed").process_images()
            messagebox.showinfo("Sucesso", process_result)

            # Abre a pasta de saída no Explorer
            subprocess.Popen(f'explorer "{os.path.abspath(self.output_dir)}"')
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao processar imagens:\n{str(e)}")

    def open_feed_window(self):
        self.root.destroy()
        root2 = tk.Tk()
        selector = YOLOSelector(root2)
        root2.mainloop()
        print(selector.selected_classes)
        self.feed_obj.allowed_classes = selector.selected_classes
        self.feed_obj.app.run(host="0.0.0.0", port=5000, debug=False)

    def open_train_window(self):
        self.model_obj.run_training()