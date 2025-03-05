# Development of an Inclusive Educational Platform Using Open Technologies and Machine Learning: A Case Study on Accessibility Enhancement.

New features:
AI Personal Assistant with:
- Mistral 7B v0.3 pre-trained
- Quantization implemented (bitsandbytes → Reduces memory footprint)
- FastAPI working

Next steps: 
- Fine-Tuning (datasets → Loads and processes fine-tuning data)
- PEFT Enables QLoRA fine-tuning
- datasets → Loads and processes fine-tuning data
- trl → Implements reward models like PPO/DPO

This work proposed and developed technological solutions for educational inclusion, integrating machine learning, natural language processing, and cross-platform interfaces. 

The main contributions include:

nlp_speech_red: Speech recognition functionality to support voice commands and text creation through voice input; 

trained_model: Real-time object recognition using the YOLOv5 model, adapted for educational environments; 

nlp_tts: Grapheme-to-Phoneme (G2P) conversion for Text-to-Speech systems using seq2seq models with attention, ensuring natural voice reading; 

yolo_mobile: A cross-platform mobile application in Flutter with local inference execution using TensorFlow Lite;

accessibility: Web tool that groups all applications developed using html, python, flask;

This project is justified by the need to make learning more equitable and accessible to all students, contributing to inclusive and high-quality education. Furthermore, it seeks to harness the potential of open technologies to create sustainable, replicable solutions that can be adapted to various educational contexts.

This project is part of my final doctoral thesis.
