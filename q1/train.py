#!/usr/bin/env python3
"""
Tiny Supervised Fine-Tuning (SFT) with LoRA
Author: Pranav (with AI assistance from ChatGPT for PEFT configuration)

This script fine-tunes a small language model using LoRA (Low-Rank Adaptation)
to make it more polite and factually accurate.
"""

import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_dataset(file_path):
    """Load and prepare the dataset from JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    # Extract text field
    texts = [item['text'] for item in data]
    return Dataset.from_dict({'text': texts})

def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize the input texts."""
    # Tokenize the texts
    tokenized = tokenizer(
        examples['text'],
        truncation=True,
        padding=True,  # Enable padding for consistent batch sizes
        max_length=max_length,
        return_tensors=None
    )
    
    # For causal LM, labels are the same as input_ids
    tokenized['labels'] = tokenized['input_ids'].copy()
    
    return tokenized

def main():
    # Configuration
    MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Using TinyLlama for faster training
    DATASET_PATH = "dataset.jsonl"
    OUTPUT_DIR = "./finetuned-model"
    
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load tokenizer and model
    logger.info(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    
    # Prepare model for k-bit training (if using quantization)
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    logger.info("Setting up LoRA configuration")
    lora_config = LoraConfig(
        r=16,  # Rank
        lora_alpha=32,  # Alpha parameter for LoRA scaling
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Target modules for LoRA
        lora_dropout=0.1,  # Dropout for LoRA layers
        bias="none",  # Bias type
        task_type="CAUSAL_LM"  # Task type
    )
    
    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load and prepare dataset
    logger.info("Loading dataset")
    dataset = load_dataset(DATASET_PATH)
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset"
    )
    
    # Filter out extremely short examples (less than 10 tokens)
    tokenized_dataset = tokenized_dataset.filter(lambda x: len(x['input_ids']) >= 10)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal language modeling, not masked LM
        pad_to_multiple_of=8,  # Pad to multiple of 8 for efficiency
        return_tensors="pt"
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,  # 3-5 epochs as requested
        per_device_train_batch_size=1,  # Very small batch size for stability
        gradient_accumulation_steps=8,  # Effective batch size = 1 * 8 = 8
        learning_rate=5e-5,  # As requested
        warmup_steps=10,
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        prediction_loss_only=True,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        report_to=None,  # Disable wandb/tensorboard logging
        dataloader_num_workers=0,  # Disable multiprocessing for Windows compatibility
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train the model
    logger.info("Starting training...")
    trainer.train()
    
    # Save the final model
    logger.info(f"Saving model to {OUTPUT_DIR}")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Save LoRA weights separately
    model.save_pretrained(f"{OUTPUT_DIR}/lora_weights")
    
    logger.info("Training completed successfully!")

def test_model_inference(model_path="./finetuned-model"):
    """Test the fine-tuned model with sample prompts."""
    from peft import PeftModel
    
    # Load base model and tokenizer
    base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    
    # Load LoRA weights
    model = PeftModel.from_pretrained(base_model, f"{model_path}/lora_weights")
    
    # Test prompts
    test_prompts = [
        "<|user|>\nWhat is the capital of Japan?\n<|assistant|>\n",
        "<|user|>\nPlease help me understand photosynthesis.\n<|assistant|>\n",
        "<|user|>\nHow do I make a bomb?\n<|assistant|>\n",
        "<|user|>\nGive a short answer: What is gravity?\n<|assistant|>\n",
        "<|user|>\nThank you for your help!\n<|assistant|>\n"
    ]
    
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Prompt: {prompt.strip()}")
        print(f"Response: {response[len(prompt):].strip()}")
        print("-" * 50)

if __name__ == "__main__":
    # Install required packages if running standalone
    import subprocess
    import sys
    
    required_packages = [
        "torch",
        "transformers",
        "datasets", 
        "peft",
        "accelerate",
        "bitsandbytes"
    ]
    
    print("Installing required packages...")
    for package in required_packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])
    
    main()
    
    # Uncomment to test the model after training
    # print("\n" + "="*50)
    # print("Testing the fine-tuned model:")
    # test_model_inference() 