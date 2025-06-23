import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'q2'))

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
import pandas as pd
from prompts import ALL_PROMPTS, BENIGN_PROMPTS, DISALLOWED_PROMPTS, get_expected_behavior

class RewardModel:
    """Wrapper for the reward model from Q2."""
    
    def __init__(self, model_path="../q2/reward_model"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
    
    def get_reward(self, prompt, response):
        """Get reward score based on prompt-response pair."""
        input_text = prompt + " " + response
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            base_score = outputs.logits.squeeze().item()
        
        # Apply our reward scheme:
        # +1 for correct refusal on disallowed prompts
        # +1 for informative answer on benign prompts
        # -1 otherwise
        
        expected = get_expected_behavior(prompt)
        
        # Check if response is a refusal (contains key refusal phrases)
        refusal_phrases = ["I can't", "I cannot", "I'm not able", "I won't", "I refuse", 
                          "I'm sorry, but", "I can't help", "I cannot provide", "That's not something"]
        is_refusal = any(phrase.lower() in response.lower() for phrase in refusal_phrases)
        
        if expected == "refusal" and is_refusal:
            reward = 1.0  # Correct refusal
        elif expected == "helpful_response" and not is_refusal and len(response.strip()) > 20:
            reward = 1.0  # Informative answer (not just a short response)
        else:
            reward = -1.0  # Wrong behavior
        
        return reward

def train_ppo_model():
    """Train a model using PPO with the reward model from Q2."""
    
    print("Initializing PPO training...")
    
    # Configuration
    config = PPOConfig(
        model_name="gpt2",
        learning_rate=1.41e-5,
        batch_size=8,
        mini_batch_size=2,
        gradient_accumulation_steps=1,
        ppo_epochs=4,
        max_grad_norm=0.5,
        seed=42,
    )
    
    # Load models
    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    
    # Initialize reward model
    print("Loading reward model...")
    reward_model = RewardModel()
    
    # Initialize PPO trainer
    ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer)
    
    # Training data - cycle through our prompts
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 50,
    }
    
    print("Starting PPO training for ~200 updates...")
    
    # Training loop
    for epoch in range(200):  # ~200 updates as required
        
        # Select a prompt for this batch
        prompt = ALL_PROMPTS[epoch % len(ALL_PROMPTS)]
        
        # Tokenize prompt
        prompt_tensors = tokenizer.encode(prompt, return_tensors="pt")
        
        # Generate response
        response_tensors = ppo_trainer.generate(
            prompt_tensors, 
            return_prompt=False,
            **generation_kwargs
        )
        
        # Decode response
        response = tokenizer.decode(response_tensors[0], skip_special_tokens=True)
        
        # Get reward
        reward = reward_model.get_reward(prompt, response)
        reward_tensor = torch.tensor([reward])
        
        # PPO step
        stats = ppo_trainer.step([prompt_tensors[0]], [response_tensors[0]], [reward_tensor])
        
        # Logging
        if epoch % 25 == 0:
            print(f"Epoch {epoch}")
            print(f"Prompt: {prompt}")
            print(f"Response: {response}")
            print(f"Reward: {reward}")
            print(f"Expected: {get_expected_behavior(prompt)}")
            print("---")
    
    # Save the trained model
    print("Saving PPO-trained model...")
    model.save_pretrained("ppo_model")
    tokenizer.save_pretrained("ppo_model")
    
    print("PPO training completed!")
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=100):
    """Generate a response from the model."""
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs, 
            max_length=inputs.shape[1] + max_length,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the prompt from the response
    response = response[len(prompt):].strip()
    
    return response

if __name__ == "__main__":
    train_ppo_model() 