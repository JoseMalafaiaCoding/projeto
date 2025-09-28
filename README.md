# Reconhecimento de EPIs com Visão Computacional

Este projeto propõe o desenvolvimento de um sistema baseado em **visão computacional** para o reconhecimento de **Equipamentos de Proteção Individual (EPIs)** em ambientes industriais e de risco.  
O objetivo é reduzir os acidentes de trabalho relacionados à ausência ou uso incorreto desses equipamentos, utilizando técnicas modernas de **Inteligência Artificial** e **Detecção de Objetos**.

---

## 📌 Resumo

A solução utiliza algoritmos de **IA** como **YOLOv8** (Ultralytics) e **OpenCV** para identificar, em tempo real, a presença de EPIs obrigatórios (capacetes, luvas, óculos de proteção, etc.) em imagens e vídeos capturados por câmeras.  
Quando situações de não conformidade são detectadas, o sistema emite **alertas automáticos** para os responsáveis, promovendo ações corretivas imediatas.

Espera-se que a implementação dessa ferramenta contribua para:
- **Reduzir riscos de acidentes**;
- **Aumentar a segurança ocupacional**;
- **Auxiliar empregadores** no cumprimento das normas de segurança.

---

## 🎯 Objetivos

- Desenvolver um sistema automatizado para monitoramento do uso de EPIs em tempo real.
- Implementar algoritmos de detecção de objetos para identificar EPIs obrigatórios.
- Integrar o sistema a câmeras industriais ou feeds de vídeo.
- Emitir alertas automáticos em casos de não conformidade.
- Avaliar a eficácia do sistema com base em métricas de desempenho.

---

## 🧪 Metodologia

- **Tipo de pesquisa**: aplicada, exploratória e experimental.  
- **Amostragem**: imagens e vídeos representativos obtidos de bancos de dados públicos.  
- **Ferramentas principais**:  
  - Python  
  - [YOLOv8](https://ultralytics.com)  
  - OpenCV  
  - Pandas, Matplotlib, Scikit-learn (para análise de resultados)  

- **Métricas avaliadas**: Acurácia, Precisão, Recall e F1-score.  

---

## 💻 Tecnologias

- **Python 3.10+**
- **YOLOv8 (Ultralytics)**
- **OpenCV**
- **Tkinter** (para interface gráfica)
- **Flask** (para servir feeds de vídeo processados em tempo real)
- **Pandas / Matplotlib / Scikit-learn**

---

## 🖼️ Interface Gráfica

O sistema conta com uma **interface gráfica** desenvolvida em Python (Tkinter), que oferece três funcionalidades principais:

1. **Processar Imagens**  
   - Abre o explorador de arquivos para selecionar uma pasta de imagens.  
   - Processa todas as imagens com o modelo YOLOv8.  
   - Salva os resultados (imagens anotadas com *bounding boxes*) em um diretório de saída.  

2. **Abrir Feed de Vídeo**  
   - Permite abrir a webcam ou um arquivo de vídeo.  
   - O feed é processado em tempo real com o modelo YOLOv8.  
   - Exibe as detecções diretamente na janela.  

3. **Treinar Modelo**  
   - Gera uma janela com botão **"Treinar"**.  
   - Executa o processo de treinamento do modelo com base nos datasets configurados.  
   - Suporta versionamento automático dos arquivos `.pt` treinados.

---

## 📊 Cenário Brasileiro e Motivação

- O Brasil registrou mais de **724 mil acidentes de trabalho em 2024** (dados do MTE e Previdência).  
- Pesquisas mostram que **grande parte desses acidentes está relacionada à falta ou uso incorreto de EPIs**.  
- Sistemas automatizados de monitoramento podem reduzir esses números de forma significativa.  