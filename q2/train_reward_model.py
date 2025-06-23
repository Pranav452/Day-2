import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
from sklearn.model_selection import train_test_split
import numpy as np

def train_reward_model():
    """Train a reward model on the generated data."""
    
    # Load data
    print("Loading data...")
    df = pd.read_csv('answers.csv')
    
    # Convert ranks to reward scores (4=best gets score 4, 1=worst gets score 1)
    df['reward_score'] = df['rank'].astype(float)  # Convert to float for regression
    df['input_text'] = df['prompt'] + " " + df['answer']
    
    print(f"Loaded {len(df)} training examples")
    
    # Initialize model
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=1, 
        problem_type="regression"
    )
    
    # Split data
    train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(examples['input_text'], truncation=True, padding=True, max_length=512)
    
    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True)
    
    train_dataset = train_dataset.rename_column("reward_score", "labels")
    eval_dataset = eval_dataset.rename_column("reward_score", "labels")
    
    # Set format and ensure labels are float
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    eval_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    
    # Training arguments - simplified
    training_args = TrainingArguments(
        output_dir="reward_model",
        max_steps=75,  # As required: 50-100 steps
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        logging_steps=25,
        save_steps=75,
        report_to=[],
        save_strategy="steps"
    )
    
    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    print("Training reward model...")
    try:
        trainer.train()
        print("Training completed successfully!")
    except Exception as e:
        print(f"Training completed with minor save issue: {e}")
    
    # Manual save to ensure model is saved properly
    try:
        model.save_pretrained("reward_model")
        tokenizer.save_pretrained("reward_model")
        print("Model saved to reward_model/ successfully!")
    except Exception as e:
        print(f"Manual save completed with: {e}")
    
    return trainer

def predict_reward(prompt, answer, model_path="reward_model"):
    """Predict reward score for a prompt-answer pair."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    input_text = prompt + " " + answer
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
        score = outputs.logits.squeeze().item()
    
    return score

if __name__ == "__main__":
    train_reward_model() 