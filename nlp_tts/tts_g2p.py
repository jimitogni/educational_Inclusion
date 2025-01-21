from gtts import gTTS
import os

texto = "Texto ou objeto detectado."
tts = gTTS(text=texto, lang='pt')
tts.save("saida.mp3")
os.system("mpg321 saida.mp3")


