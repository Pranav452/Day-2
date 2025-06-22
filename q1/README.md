# Q1: Tiny Supervised Fine-Tuning (SFT) with LoRA

**Author**: Pranav  
**AI Assistance**: ChatGPT helped with PEFT configuration and LoRA setup

To reproduce all results from scratch:
```bash
cd q1/
pip install -r requirements.txt
python train.py  # Trains model (~7.5 min on GPU)
```
All evaluation results are already documented in `comparison_before_after.md`

## Project Overview

This project demonstrates how to fine-tune a small language model (TinyLlama-1.1B) using LoRA (Low-Rank Adaptation) to make it more polite, helpful, and factually accurate. The goal is to transform a raw base model into a well-behaved assistant through supervised fine-tuning on a custom dataset.

## 🎯 Objectives

- Create a small but diverse training dataset (25 prompt/response pairs)
- Fine-tune using LoRA for memory efficiency
- Improve model politeness, factual accuracy, and safety
- Demonstrate measurable improvements through before/after comparisons

## 📁 Project Structure

```
q1/
├── dataset.jsonl                     # Training dataset (26 examples)
├── train.py                          # Fine-tuning script with LoRA
├── comparison_before_after.md        # Actual evaluation results
├── requirements.txt                  # Python dependencies
└── README.md                        # This file

# Generated after training:
└── finetuned-model/                 # Created automatically by train.py
    ├── lora_weights/                # LoRA adapter weights
    └── ...                          # Tokenizer and config files
```

## 📊 Dataset Composition

Our training dataset includes exactly what was requested:

| Category | Count | Examples |
|----------|-------|----------|
| **Factual Q&A** | 3 | "What is the capital of France?" → "Paris" |
| **Polite-tone** | 3 | "Please translate 'Bonjour'" → "Certainly! 'Bonjour' means 'Hello'..." |
| **Short vs Long** | 2 | Brief AI explanation vs detailed AI explanation |
| **Refusals** | 2 | Bomb making → "I can't help with that..." |
| **Additional** | 15 | Various helpful responses, math, science, gratitude |

## 🚀 Setup Instructions

### Prerequisites
- Python 3.10+
- 8GB+ RAM (16GB recommended)
- GPU with 6GB+ VRAM (optional but recommended)

### Installation

1. **Clone/Download the project**
   ```bash
   cd q1/
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the training script**
   ```bash
   python train.py
   ```

## ⚙️ Technical Configuration

### Model Details
- **Base Model**: TinyLlama/TinyLlama-1.1B-Chat-v1.0
- **Method**: LoRA (Low-Rank Adaptation) 
- **Parameters Updated**: ~0.2% of total model parameters

### LoRA Configuration
```python
LoraConfig(
    r=16,                    # Rank
    lora_alpha=32,          # Alpha scaling
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.1,       # Dropout
    task_type="CAUSAL_LM"
)
```

### Training Settings
- **Epochs**: 3
- **Learning Rate**: 5e-5
- **Batch Size**: 2 (with gradient accumulation = 8 effective)
- **Max Length**: 512 tokens
- **Mixed Precision**: FP16 (if GPU available)

## 📈 Results Summary

The fine-tuning successfully improved the model across all target areas:

### Key Improvements
1. **Factual Accuracy**: More confident, direct answers
2. **Politeness**: Consistent use of courteous language
3. **Safety**: Proper refusal of harmful requests
4. **Instruction Following**: Better adherence to length constraints
5. **Engagement**: Warmer, more helpful responses

### Before vs After Examples

| Aspect | Before | After |
|--------|--------|-------|
| **Factual** | "I think Tokyo, but not sure..." | "The capital of Japan is Tokyo." |
| **Politeness** | Basic responses | "Certainly! I'd be happy to help..." |
| **Safety** | Potential harmful content | "I can't help with that. Here are safe alternatives..." |

See `comparison_before_after.md` for detailed evaluation results.

## 🔧 Usage

### Training a New Model
```bash
python train.py
```

### Testing the Fine-tuned Model
Uncomment the test lines in `train.py`:
```python
print("\n" + "="*50)
print("Testing the fine-tuned model:")
test_model_inference()
```

### Custom Dataset
To use your own data, modify `dataset.jsonl` with this format:
```json
{"text": "<|user|>\nYour question here\n<|assistant|>\nYour response here"}
```

## 📋 Requirements Met

✅ **Dataset**: 25 prompt/response pairs with all 4 required categories  
✅ **Model**: TinyLlama-1.1B (efficient for laptops/small GPUs)  
✅ **Method**: LoRA with PEFT for efficient fine-tuning  
✅ **Training**: 3 epochs, 5e-5 learning rate as specified  
✅ **Evaluation**: 5 test prompts with before/after comparisons  
✅ **Deliverables**: Complete project with documentation  

## 💡 Key Learnings

1. **LoRA Efficiency**: Only updating ~0.2% of parameters still yields significant improvements
2. **Data Quality**: Even 25 well-designed examples can teach important behaviors
3. **Safety Training**: Including refusal examples helps create responsible AI behavior
4. **Politeness Transfer**: Training on polite examples generalizes to new interactions

## 🎓 Academic Integrity

This project represents my individual work with AI assistance from ChatGPT for:
- PEFT/LoRA configuration details
- Hugging Face Transformers API usage
- Training optimization techniques

All code implementation, dataset creation, evaluation design, and analysis represents my personal learning and understanding of supervised fine-tuning concepts.

## 📝 Future Improvements

- Expand dataset with more diverse examples
- Experiment with different LoRA ranks and alphas  
- Add quantitative evaluation metrics
- Try instruction-following datasets like Alpaca
- Compare with other efficient fine-tuning methods

---
