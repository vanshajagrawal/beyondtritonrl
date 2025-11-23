#!/usr/bin/env python3
"""
Complete TritonRL training and evaluation pipeline
"""
import os
import sys
from config import DataConfig, SFTConfig, RLConfig, EvalConfig

def run_data_preparation():
    """Step 1: Prepare training data"""
    print("\n" + "="*80)
    print("STEP 1: DATA PREPARATION")
    print("="*80)

    from data.prepare_data import main as prepare_data
    prepare_data()

def run_sft():
    """Step 2: Supervised Fine-Tuning"""
    print("\n" + "="*80)
    print("STEP 2: SUPERVISED FINE-TUNING")
    print("="*80)

    from train_sft import main as train_sft
    train_sft()

def run_rl():
    """Step 3: Reinforcement Learning"""
    print("\n" + "="*80)
    print("STEP 3: REINFORCEMENT LEARNING")
    print("="*80)

    from train_rl import main as train_rl
    train_rl()

def run_evaluation():
    """Step 4: Evaluation on KernelBench"""
    print("\n" + "="*80)
    print("STEP 4: EVALUATION")
    print("="*80)

    from evaluate import main as evaluate

    # Evaluate both SFT and RL models
    print("\nEvaluating SFT model...")
    os.system(f"python evaluate.py --model_path checkpoints/sft --level 1")

    print("\nEvaluating RL model...")
    os.system(f"python evaluate.py --model_path checkpoints/rl --level 1")

def main():
    """Run complete pipeline"""
    import argparse

    parser = argparse.ArgumentParser(description="TritonRL Training Pipeline")
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["data", "sft", "rl", "eval", "all"],
        default=["all"],
        help="Which steps to run",
    )
    args = parser.parse_args()

    # Create output directories
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("checkpoints/sft", exist_ok=True)
    os.makedirs("checkpoints/rl", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    steps = args.steps
    if "all" in steps:
        steps = ["data", "sft", "rl", "eval"]

    try:
        if "data" in steps:
            run_data_preparation()

        if "sft" in steps:
            run_sft()

        if "rl" in steps:
            run_rl()

        if "eval" in steps:
            run_evaluation()

        print("\n" + "="*80)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("="*80)

    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
