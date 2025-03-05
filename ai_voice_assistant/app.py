import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

# Initialize DDP
def setup():
    dist.init_process_group(backend="nccl")  # ✅ Use NCCL for fast GPU communication
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))  # ✅ Assign GPU to each process

def cleanup():
    dist.destroy_process_group()

# ✅ Run setup for distributed training
setup()

# ✅ Load model and tokenizer
MODEL_NAME = "mistralai/Mistral-7B-v0.3"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # Fix padding issue

# ✅ Define quantization config
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# ✅ Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map={"": int(os.environ["LOCAL_RANK"])},  # Assign GPU based on LOCAL_RANK
    bitsandbytes_config=bnb_config
)

# ✅ Prepare model for LoRA fine-tuning
model = prepare_model_for_kbit_training(model)

# ✅ Apply LoRA adapters
lora_config = LoraConfig(
    r=16, 
    lora_alpha=32, 
    lora_dropout=0.1, 
    target_modules=["q_proj", "v_proj"]
)
model = get_peft_model(model, lora_config)

# ✅ Wrap model in DDP
model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])], output_device=int(os.environ["LOCAL_RANK"]))

# ✅ Load dataset
dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")

# ✅ Preprocess dataset
def preprocess_function(example):
    choices_text = example["choices"]["text"]
    choices_labels = example["choices"]["label"]
    
    formatted_prompt = f"Question: {example['question']}\nOptions:\n"
    formatted_prompt += "\n".join([f"({label}) {text}" for label, text in zip(choices_labels, choices_text)])
    formatted_prompt += "\nAnswer:"

    inputs = tokenizer(
        formatted_prompt,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    labels = inputs["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    inputs["labels"] = labels

    return {key: value.squeeze(0).tolist() for key, value in inputs.items()}

# ✅ Apply preprocessing
formatted_dataset = dataset.map(preprocess_function)

# ✅ Training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    output_dir="./mistral-finetuned",
    remove_unused_columns=False,
    fp16=True
)

# ✅ Trainer
trainer = Trainer(
    model=model,
    train_dataset=formatted_dataset,
    args=training_args
)

# ✅ Start training
trainer.train()

# ✅ Clean up DDP
cleanup()
