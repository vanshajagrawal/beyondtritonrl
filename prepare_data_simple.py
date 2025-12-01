#!/usr/bin/env python3
"""
Simplified data preparation - uses KernelBook as-is (no DeepSeek generation)
This saves 2-3 hours in the 10-hour budget
"""

import json
import os
from datasets import load_dataset
from tqdm import tqdm

def prepare_kernelbook_data(max_samples=1000):
    """
    Load and prepare KernelBook data for training
    Uses existing Triton code (no generation needed)
    """
    print("="*80)
    print("LOADING KERNELBOOK DATASET")
    print("="*80)

    # Create output directory
    os.makedirs("data/processed", exist_ok=True)

    # Load KernelBook from HuggingFace
    print("\nLoading KernelBook from HuggingFace...")
    dataset = load_dataset("GPUMODE/KernelBook", split="train")

    print(f"Total samples in KernelBook: {len(dataset)}")
    print(f"Using first {max_samples} samples for 10-hour budget")

    # Take subset for faster training
    dataset = dataset.select(range(min(max_samples, len(dataset))))

    # Prepare SFT training data
    sft_data = []
    difficulty_labels = []

    print("\nProcessing samples...")
    for idx, item in enumerate(tqdm(dataset)):
        # Extract fields
        pytorch_code = item.get("python_code", "")
        triton_code = item.get("triton_code", "")

        # Skip if missing required fields
        if not pytorch_code or not triton_code:
            continue

        # Create instruction prompt
        instruction = f"""Your task is to write a custom Triton kernel to optimize the following PyTorch code:

```python
{pytorch_code}
```

Write an optimized Triton kernel. Output the new code in codeblocks."""

        # Create SFT training sample
        sft_data.append({
            "id": f"kernelbook_{idx}",
            "instruction": instruction,
            "output": triton_code,
            "pytorch_code": pytorch_code,
        })

        # Label difficulty (heuristic: based on code complexity)
        difficulty = estimate_difficulty(pytorch_code, triton_code)

        difficulty_labels.append({
            "id": f"kernelbook_{idx}",
            "pytorch_code": pytorch_code,
            "triton_code": triton_code,
            "difficulty": difficulty,
        })

    print(f"\nPrepared {len(sft_data)} training samples")

    # Count by difficulty
    l1_count = sum(1 for d in difficulty_labels if d["difficulty"] == 1)
    l2_count = sum(1 for d in difficulty_labels if d["difficulty"] == 2)
    l3_count = sum(1 for d in difficulty_labels if d["difficulty"] == 3)

    print(f"  Level 1 (single ops): {l1_count}")
    print(f"  Level 2 (fusion): {l2_count}")
    print(f"  Level 3 (architectures): {l3_count}")

    # Save datasets
    print("\nSaving datasets...")

    with open("data/processed/sft_train.jsonl", "w") as f:
        for item in sft_data:
            f.write(json.dumps(item) + "\n")

    with open("data/processed/difficulty_labels.jsonl", "w") as f:
        for item in difficulty_labels:
            f.write(json.dumps(item) + "\n")

    print(f"✓ Saved to data/processed/sft_train.jsonl")
    print(f"✓ Saved to data/processed/difficulty_labels.jsonl")

    return sft_data, difficulty_labels


def estimate_difficulty(pytorch_code, triton_code):
    """
    Heuristic difficulty estimation:
    - Level 1: Single operation (one nn module or simple function)
    - Level 2: Multiple operations (fusion patterns)
    - Level 3: Full architectures (many modules)
    """
    # Count nn modules as proxy for complexity
    nn_count = pytorch_code.count("nn.")

    # Count lines as another proxy
    lines = len([l for l in pytorch_code.split('\n') if l.strip()])

    if nn_count <= 1 and lines < 30:
        return 1  # Single op
    elif nn_count <= 3 and lines < 80:
        return 2  # Fusion
    else:
        return 3  # Architecture


def main():
    print("\n" + "="*80)
    print("SIMPLIFIED DATA PREPARATION (10-Hour Budget)")
    print("="*80)
    print("\nStrategy: Use KernelBook as-is (no generation)")
    print("Saves: 2-3 hours of DeepSeek API calls")
    print("Trade-off: Lower quality labels, but faster iteration\n")

    sft_data, difficulty_labels = prepare_kernelbook_data(max_samples=1000)

    print("\n" + "="*80)
    print("DATA PREPARATION COMPLETE")
    print("="*80)
    print(f"\nNext steps:")
    print("  1. Run: python train_integrated.py --stage sft")
    print("  2. Run: python train_integrated.py --stage rl")
    print("  3. Run: python evaluate.py --model_path checkpoints/rl_final")


if __name__ == "__main__":
    main()
