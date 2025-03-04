from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# Load model and tokenizer
MODEL_NAME = "mistralai/Mistral-7B-v0.3"
#MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
#MODEL_NAME = "microsoft/phi-2"

app = FastAPI()

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Load model and tokenizer with reduced memory usage
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)

print("Model loaded successfully!")

# Define a request model for proper JSON parsing
class PromptRequest(BaseModel):
    prompt: str

# 🚀 Test the model once when the server starts
prompt = "Explain the importance of AI in education for people with special needs."
# inputs = tokenizer(prompt, return_tensors="pt").to(device)
# outputs = model.generate(**inputs, max_length=200)
# response = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print("Test Output:", response)  # ✅ This prints the model output

# Home page route
@app.get("/")
def home():
    return {"message": "Welcome to the Mistral-7B API! Use /generate to get responses."}

# Generate response route
# Corrected POST route
@app.post("/generate")
async def generate_text(request: PromptRequest):
    inputs = tokenizer(request.prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": response}


# @app.get("/generate")
# async def generate_get(prompt: str = "Explain the importance of AI in education for people with special needs."):
#     inputs = tokenizer(prompt, return_tensors="pt").to(device)
#     outputs = model.generate(**inputs, max_length=200)
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return {"response": response}













    
# Run with: uvicorn app:app --host 0.0.0.0 --port 8000