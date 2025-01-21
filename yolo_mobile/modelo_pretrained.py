import torch
from PIL import Image
import matplotlib.pyplot as plt

# Carregar o modelo pré-treinado
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Carregar a imagem de exemplo
image_path = 'imgs/recimg.jpg'  # Altere para o caminho da sua imagem
image = Image.open(image_path)

# Realizar a detecção
results = model(image_path)

# Exibir os resultados
results.print()  # Imprimir no console
results.show()   # Mostrar com matplotlib ou interface gráfica
