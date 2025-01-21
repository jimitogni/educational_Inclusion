import cv2
import tensorflow as tf

# Carregar o modelo
interpreter = tf.lite.Interpreter(model_path="yolov5s.tflite")
interpreter.allocate_tensors()

# Pré-processar imagem
image = cv2.imread("imagem.jpg")
input_data = preprocess_image(image)

# Executar a inferência
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

from gtts import gTTS
import os

texto = "Livro detectado."
tts = gTTS(text=texto, lang='pt')
tts.save("audio.mp3")
os.system("mpg321 audio.mp3")


