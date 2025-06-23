#!/usr/bin/env python3
"""
Q3: First RLHF Loop (PPO-style)

This script implements the complete RLHF pipeline:
1. Phase 1: PPO Training using reward model from Q2
2. Phase 2: Evaluation comparing base model vs PPO-trained model

Usage:
    python run_rlhf.py --train    # Run PPO training
    python run_rlhf.py --eval     # Run evaluation only
    python run_rlhf.py --all      # Run both training and evaluation
"""

import argparse
import os
import sys
from ppo_training import train_ppo_model
from evaluate import evaluate_models, create_result_table

def main():
    parser = argparse.ArgumentParser(description='Run RLHF Loop for Q3')
    parser.add_argument('--train', action='store_true', help='Run PPO training')
    parser.add_argument('--eval', action='store_true', help='Run evaluation')
    parser.add_argument('--all', action='store_true', help='Run both training and evaluation')
    
    args = parser.parse_args()
    
    if not any([args.train, args.eval, args.all]):
        # Default behavior - run evaluation if PPO model exists, otherwise run all
        if os.path.exists('ppo_model'):
            print("PPO model found. Running evaluation...")
            args.eval = True
        else:
            print("No PPO model found. Running complete pipeline...")
            args.all = True
    
    try:
        # Phase 1: PPO Training
        if args.train or args.all:
            print("="*80)
            print("PHASE 1: PPO TRAINING")
            print("="*80)
            print("Training model to refuse unsafe instructions using reward model from Q2...")
            
            # Check if reward model exists
            if not os.path.exists('../q2/reward_model'):
                print("ERROR: Reward model from Q2 not found!")
                print("Please ensure Q2 has been completed and the reward model is trained.")
                return
            
            model, tokenizer = train_ppo_model()
            print("PPO training completed successfully!")
        
        # Phase 2: Evaluation
        if args.eval or args.all:
            print("\n" + "="*80)
            print("PHASE 2: EVALUATION")
            print("="*80)
            print("Comparing base model vs PPO-trained model on safety prompts...")
            
            df, comparison_df = evaluate_models()
            
            print("\nEvaluation completed!")
            print("Results saved to:")
            print("- evaluation_results.csv (detailed results)")
            print("- comparison_table.csv (6×2 comparison table)")
            
            # Create final result table
            create_result_table()
    
    except Exception as e:
        print(f"Error during execution: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install -r requirements.txt")
        return

if __name__ == "__main__":
    main() 