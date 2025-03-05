from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
from datasets import load_dataset

# ✅ Load dataset
dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")

# ✅ Load tokenizer
MODEL_NAME = "mistralai/Mistral-7B-v0.3"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # Fix padding issue

# ✅ Define quantization configuration (fix deprecated method)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # ✅ 4-bit quantization
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# ✅ Load model with quantization (Fix: use `bitsandbytes_config`, not `quantization_config`)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16,
    offload_folder="./offload",
    quantization_config=bnb_config  # ✅ Corrected argument
)

# ✅ Prepare model for training (Fix: Ensures it can be fine-tuned)
model = prepare_model_for_kbit_training(model)

# ✅ Apply LoRA adapters
lora_config = LoraConfig(
    r=16, 
    lora_alpha=32, 
    lora_dropout=0.1, 
    target_modules=["q_proj", "v_proj"]  # ✅ Fine-tune only key layers
)
model = get_peft_model(model, lora_config)

# ✅ Ensure only LoRA layers require gradients
for name, param in model.named_parameters():
    if "lora" in name.lower():  
        param.requires_grad = True

# ✅ Preprocess dataset
def preprocess_function(example):
    # Extract question and choices
    choices_text = example["choices"]["text"]
    choices_labels = example["choices"]["label"]

    # Ensure labels are stored correctly
    answer_index = choices_labels.index(example["answerKey"]) if example["answerKey"] in choices_labels else -1

    # Format input prompt
    formatted_prompt = f"Question: {example['question']}\nOptions:\n"
    formatted_prompt += "\n".join([f"({label}) {text}" for label, text in zip(choices_labels, choices_text)])
    formatted_prompt += "\nAnswer:"

    # Tokenize input
    inputs = tokenizer(
        formatted_prompt,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    # Ensure labels match input IDs and are properly formatted
    labels = inputs["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100  # Ignore padding tokens in loss calculation

    # ✅ Convert tensors to lists to avoid dict errors
    inputs = {k: v.squeeze(0).tolist() for k, v in inputs.items()}
    inputs["labels"] = labels.squeeze(0).tolist()  # Ensure labels are also in list format

    return inputs  # ✅ Now, Trainer will process it correctly


# ✅ Apply preprocessing
formatted_dataset = dataset.map(preprocess_function)

# ✅ Enable gradient checkpointing (Fix: allows memory-efficient training)
model.gradient_checkpointing_enable()

# ✅ Training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=1,  # ✅ Reduce memory usage
    gradient_accumulation_steps=2,  # ✅ Accumulate gradients to fit training into memory
    gradient_checkpointing=True,  # ✅ Memory-efficient training
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    output_dir="./mistral-finetuned",
    remove_unused_columns=False,  # ✅ Ensure dataset fields aren't removed
    fp16=True  # ✅ Enable mixed precision training
)
formatted_dataset = dataset.map(preprocess_function, remove_columns=["question", "choices", "id", "answerKey"])
from transformers import DataCollatorForSeq2Seq

# ✅ Use a data collator that works with sequence-to-sequence models
data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True)

trainer = Trainer(
    model=model,
    train_dataset=formatted_dataset,
    args=training_args,
    data_collator=data_collator  # ✅ Fixes dtype issue
)




# ✅ Start fine-tuning
trainer.train()


# #print("Model loaded successfully!")

# # Define a request model for proper JSON parsing
# class PromptRequest(BaseModel):
#     prompt: str

# # 🚀 Test the model once when the server starts
# prompt = "Explain the importance of AI in education for people with special needs."
# # inputs = tokenizer(prompt, return_tensors="pt").to(device)
# # outputs = model.generate(**inputs, max_length=200)
# # response = tokenizer.decode(outputs[0], skip_special_tokens=True)
# # print("Test Output:", response)  # This prints the model output

# # Home page route
# @app.get("/")
# def home():
#     return {"message": "Welcome to the Mistral-7B API! Use /generate to get responses."}

# # Generate response route
# # Corrected POST route
# @app.post("/generate")
# async def generate_text(request: PromptRequest):
#     inputs = tokenizer(request.prompt, return_tensors="pt").to(device)
#     outputs = model.generate(**inputs, max_length=200)
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return {"response": response}


# @app.get("/generate")
# async def generate_get(prompt: str = "Explain the importance of AI in education for people with special needs."):
#     inputs = tokenizer(prompt, return_tensors="pt").to(device)
#     outputs = model.generate(**inputs, max_length=200)
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return {"response": response}

    


    
# Run with: uvicorn app:app --host 0.0.0.0 --port 8000