import whisper

# Carregar o modelo
model = whisper.load_model("base")

# Transcrever áudio
result = model.transcribe("audio.wav", language="pt")
print("Texto reconhecido:", result["text"])

comandos = {
    "iniciar leitura": "start_reading",
    "traduzir para libras": "translate_to_libras",
    "finalizar": "stop_action"
}

texto = result["text"].lower()
if texto in comandos:
    executar_funcao(comandos[texto])
else:
    print("Entrada de texto:", texto)

import sounddevice as sd
import numpy as np

def capturar_audio():
    samplerate = 16000
    duration = 5  # Segundos
    audio = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    return audio

audio = capturar_audio()
result = model.transcribe(audio, fp16=False)  # Transcrição direta

palavras_ativacao = ["iniciar", "assistente", "traduzir", "finalizar"]

texto = result["text"].lower()
if any(palavra in texto for palavra in palavras_ativacao):
    tipo = "comando"
else:
    tipo = "texto"

from transformers import pipeline

# Carregar modelo de classificação
classificador = pipeline("text-classification", model="bert-base-uncased")

entrada = result["text"]
classificacao = classificador(entrada)

if classificacao[0]["label"] == "comando":
    tipo = "comando"
else:
    tipo = "texto"

def identificar_tipo_entrada(texto):
    palavras_ativacao = ["iniciar", "assistente", "traduzir", "finalizar"]
    
    if any(palavra in texto.lower() for palavra in palavras_ativacao):
        return "comando"
    else:
        classificacao = classificador(texto)
        return classificacao[0]["label"]

texto = result["text"]
tipo_entrada = identificar_tipo_entrada(texto)

if tipo_entrada == "comando":
    executar_comando(texto)
else:
    inserir_texto(texto)

palavras_ativacao = ["iniciar", "assistente", "traduzir", "finalizar"]

def verificar_palavras_chave(texto):
    return any(palavra in texto.lower() for palavra in palavras_ativacao)

from transformers import pipeline

classificador = pipeline("text-classification", model="bert-base-uncased")

def classificar_texto(texto):
    resultado = classificador(texto)
    return resultado[0]["label"]

def identificar_tipo_entrada(texto):
    if verificar_palavras_chave(texto):
        return "comando"
    else:
        return classificar_texto(texto)

entrada = "iniciar tradução para Libras"
tipo_entrada = identificar_tipo_entrada(entrada)

if tipo_entrada == "comando":
    executar_comando(entrada)
else:
    processar_texto(entrada)



