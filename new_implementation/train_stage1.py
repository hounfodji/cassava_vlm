#!/usr/bin/env python3
"""
Stage 1: Disease Recognition Foundation Training Script
Fine-tune Qwen3-VL-8B-Instruct sur la reconnaissance de maladies des cultures
"""

import os
import json
import re
import torch
from pathlib import Path
from PIL import Image
from dataclasses import dataclass
from typing import Dict, List, Any
from datasets import Dataset
from transformers import (
    AutoProcessor,
    TrainingArguments,
    Trainer
)
from tqdm import tqdm

# IMPORTANT: Import unsloth BEFORE transformers pour les optimisations
from unsloth import FastVisionModel

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = Path("/home/hounfodjidagba/learning/agrovista")
DATA_DIR = BASE_DIR / "data" / "stage1"
CHECKPOINT_DIR = BASE_DIR / "checkpoints" / "stage1_disease_foundation"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Modèle
MODEL_NAME = "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit"

# LoRA Configuration
LORA_CONFIG = {
    "r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}

# Training Configuration
TRAINING_CONFIG = {
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "num_train_epochs": 3,
    "learning_rate": 2e-5,
    "warmup_steps": 100,
    "logging_steps": 25,
    "save_steps": 500,
    "save_total_limit": 3,
    "fp16": not torch.cuda.is_bf16_supported(),
    "bf16": torch.cuda.is_bf16_supported(),
    "optim": "adamw_8bit",
    "weight_decay": 0.01,
    "lr_scheduler_type": "cosine",
    "max_grad_norm": 1.0,
    "seed": 42
}

print(f"\nBase Directory: {BASE_DIR}")
print(f"Data Directory: {DATA_DIR}")
print(f"Checkpoint Directory: {CHECKPOINT_DIR}")

# ============================================================================
# Chargement du Modèle
# ============================================================================

print("\n" + "="*60)
print("LOADING MODEL WITH UNSLOTH")
print("="*60)

model, tokenizer = FastVisionModel.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)

model = FastVisionModel.get_peft_model(
    model,
    r=LORA_CONFIG["r"],
    lora_alpha=LORA_CONFIG["lora_alpha"],
    lora_dropout=LORA_CONFIG["lora_dropout"],
    target_modules=LORA_CONFIG["target_modules"],
    use_rslora=True,
    use_gradient_checkpointing="unsloth"
)

processor = AutoProcessor.from_pretrained(MODEL_NAME)

print(f"Model: {MODEL_NAME}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Trainable %: {100 * sum(p.numel() for p in model.parameters() if p.requires_grad) / sum(p.numel() for p in model.parameters()):.2f}%")
print("="*60)

# ============================================================================
# Chargement et Formatage des Données
# ============================================================================

print("\n" + "="*60)
print("LOADING DATASETS")
print("="*60)

def load_json_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

plantvillage_data = load_json_dataset(DATA_DIR / "plantvillage_vqa.json")
disease_diagnosis_data = load_json_dataset(DATA_DIR / "disease_diagnosis.json")
disease_knowledge_data = load_json_dataset(DATA_DIR / "disease_knowledge.json")

print(f"PlantVillage VQA: {len(plantvillage_data):,} samples")
print(f"Disease Diagnosis: {len(disease_diagnosis_data):,} samples")
print(f"Disease Knowledge: {len(disease_knowledge_data):,} samples")

all_data = plantvillage_data + disease_diagnosis_data + disease_knowledge_data
print(f"Total Stage 1: {len(all_data):,} samples")

def extract_image_path(text):
    """Extrait le chemin de l'image du tag <img>"""
    match = re.search(r'<img>(.*?)</img>', text)
    return match.group(1) if match else None

def format_sample_for_qwen3vl(sample):
    """Prépare un échantillon pour Qwen3-VL"""
    conversations = sample['conversations']

    image_path = None
    formatted_conversations = []

    for turn in conversations:
        role = turn['from']
        content = turn['value']

        if '<img>' in content:
            img_path = extract_image_path(content)
            if img_path:
                image_path = img_path
            # Remplacer <img>path</img> par <image> (marqueur Qwen-VL)
            content = re.sub(r'<img>.*?</img>', '<image>', content)

        formatted_conversations.append({
            "role": role,
            "content": content
        })

    return {
        "conversations": formatted_conversations,
        "image_path": image_path,
        "id": sample['id']
    }

print("\nFormatting data...")
formatted_samples = []
for sample in tqdm(all_data, desc="Formatting"):
    formatted = format_sample_for_qwen3vl(sample)
    formatted_samples.append(formatted)

dataset = Dataset.from_list(formatted_samples)
dataset = dataset.train_test_split(test_size=0.05, seed=42)
train_dataset = dataset['train']
eval_dataset = dataset['test']

print(f"Train: {len(train_dataset):,} samples")
print(f"Validation: {len(eval_dataset):,} samples")

# ============================================================================
# Data Collator
# ============================================================================

@dataclass
class Qwen3VLDataCollator:
    """Data collator pour Qwen3-VL"""
    processor: Any
    base_dir: str = str(BASE_DIR)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Pour Qwen3-VL, utiliser le format de messages avec images
        batch_messages = []
        batch_images_all = []

        for feature in features:
            image_path = feature['image_path']
            image = None

            if image_path:
                # Normaliser le chemin
                if image_path.startswith('/'):
                    full_path = None
                elif image_path.startswith('PlantVillage/'):
                    full_path = f"{self.base_dir}/data/raw/plantvillage/{image_path}"
                elif image_path.startswith('data/'):
                    full_path = f"{self.base_dir}/{image_path}"
                else:
                    full_path = f"{self.base_dir}/{image_path}"

                if full_path:
                    try:
                        image = Image.open(full_path).convert('RGB')
                    except Exception:
                        image = Image.new('RGB', (224, 224), color='white')
                else:
                    image = Image.new('RGB', (224, 224), color='white')

            # Construire les messages en format chat
            conversations = feature['conversations']
            messages = []
            for turn in conversations:
                role = turn['role']
                content = turn['content']

                # Le premier message utilisateur contient l'image
                if role == 'user' and image is not None and len(messages) == 0:
                    messages.append({
                        "role": role,
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": content.replace('<image>', '').strip()}
                        ]
                    })
                else:
                    messages.append({
                        "role": role,
                        "content": content.replace('<image>', '').strip()
                    })

            batch_messages.append(messages)

        # Appliquer le chat template et tokenizer
        batch_texts = []
        for messages in batch_messages:
            text = self.processor.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            batch_texts.append(text)

        # Extraire toutes les images
        all_images = []
        for messages in batch_messages:
            for msg in messages:
                if isinstance(msg.get("content"), list):
                    for item in msg["content"]:
                        if item.get("type") == "image":
                            all_images.append(item["image"])

        # Tokenizer avec le processor
        batch = self.processor(
            text=batch_texts,
            images=all_images if all_images else None,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )

        batch["labels"] = batch["input_ids"].clone()

        return batch

data_collator = Qwen3VLDataCollator(processor=processor, base_dir=str(BASE_DIR))

print("Data collator created")

# ============================================================================
# Training
# ============================================================================

print("\n" + "="*60)
print("CONFIGURING TRAINER")
print("="*60)

training_args = TrainingArguments(
    output_dir=str(CHECKPOINT_DIR),
    per_device_train_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
    per_device_eval_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
    gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
    num_train_epochs=TRAINING_CONFIG["num_train_epochs"],
    learning_rate=TRAINING_CONFIG["learning_rate"],
    warmup_steps=TRAINING_CONFIG["warmup_steps"],
    logging_steps=TRAINING_CONFIG["logging_steps"],
    save_steps=TRAINING_CONFIG["save_steps"],
    save_total_limit=TRAINING_CONFIG["save_total_limit"],
    eval_strategy="steps",
    eval_steps=TRAINING_CONFIG["save_steps"],
    fp16=TRAINING_CONFIG["fp16"],
    bf16=TRAINING_CONFIG["bf16"],
    optim=TRAINING_CONFIG["optim"],
    weight_decay=TRAINING_CONFIG["weight_decay"],
    lr_scheduler_type=TRAINING_CONFIG["lr_scheduler_type"],
    max_grad_norm=TRAINING_CONFIG["max_grad_norm"],
    seed=TRAINING_CONFIG["seed"],
    report_to="none",
    remove_unused_columns=False,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    dataloader_pin_memory=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

print(f"Effective batch size: {TRAINING_CONFIG['per_device_train_batch_size'] * TRAINING_CONFIG['gradient_accumulation_steps']}")
print(f"Total training steps: {len(train_dataset) // (TRAINING_CONFIG['per_device_train_batch_size'] * TRAINING_CONFIG['gradient_accumulation_steps']) * TRAINING_CONFIG['num_train_epochs']}")
print("="*60)

print("\n" + "="*60)
print("STARTING TRAINING - STAGE 1")
print("="*60)
print("Estimated time: 6-8 hours on V100 16GB")
print("Target: >90% disease accuracy")
print("="*60)

trainer.train()

print("\n" + "="*60)
print("TRAINING COMPLETED")
print("="*60)

# ============================================================================
# Sauvegarde
# ============================================================================

final_model_path = CHECKPOINT_DIR / "final_model"
trainer.save_model(str(final_model_path))
tokenizer.save_pretrained(str(final_model_path))

lora_adapters_path = CHECKPOINT_DIR / "lora_adapters"
model.save_pretrained(str(lora_adapters_path))

metrics = {
    "stage": "stage1_disease_foundation",
    "model": MODEL_NAME,
    "train_samples": len(train_dataset),
    "eval_samples": len(eval_dataset),
    "epochs": TRAINING_CONFIG["num_train_epochs"],
    "lora_config": LORA_CONFIG,
    "training_config": TRAINING_CONFIG
}

with open(CHECKPOINT_DIR / "training_metrics.json", 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"\nModèle sauvegardé dans: {final_model_path}")
print(f"Adapters LoRA: {lora_adapters_path}")
print("\nSTAGE 1 COMPLETED SUCCESSFULLY")
