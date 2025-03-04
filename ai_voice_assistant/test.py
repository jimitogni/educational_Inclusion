from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model
#MODEL_NAME = "mistralai/Mistral-7B-v0.3"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
#MODEL_NAME = "microsoft/phi-2"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

# Test a simple prompt
prompt = "Explain the importance of AI in education."
inputs = tokenizer(prompt, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_length=200)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Generated Text:", response)
