import os
from flask import Flask, request, render_template, send_file
import torch
from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Carregar o modelo YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "Nenhum arquivo selecionado.", 400

    file = request.files['file']
    if file.filename == '':
        return "Nenhum arquivo selecionado.", 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Processar a imagem com YOLOv5
    results = model(file_path)
    results.save(save_dir=UPLOAD_FOLDER)  # Salvar resultados no diretório de uploads

    # Enviar o arquivo processado de volta
    output_image = os.path.join(UPLOAD_FOLDER, f"labels/{file.filename}")
    return send_file(output_image, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)

