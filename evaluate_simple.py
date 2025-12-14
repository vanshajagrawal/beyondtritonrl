#!/usr/bin/env python3
"""
Simplified evaluation script for 10-hour budget
Tests on small subset of KernelBench
"""

import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from extensions import HardenedVerifier, StagedEvaluator
from extensions.config import ExtensionConfig
from tqdm import tqdm

def evaluate_model(model_path, num_test_tasks=20):
    """
    Evaluate model on test tasks

    Args:
        model_path: Path to trained model
        num_test_tasks: Number of tasks to test (20 for 10-hour budget)
    """
    print("="*80)
    print(f"EVALUATING MODEL: {model_path}")
    print("="*80)

    # Load model
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model.eval()

    # Setup verifier with extensions
    print("Setting up verifier with all extensions...")
    ext_config = ExtensionConfig(
        enable_multi_input=True,
        enable_staged_eval=True,
        enable_calibrated_timing=False,  # Skip for faster eval
    )
    verifier = HardenedVerifier(ext_config)
    evaluator = StagedEvaluator(verifier, ext_config)

    # Load test tasks (Level 2 fusion tasks)
    print(f"\nLoading {num_test_tasks} test tasks...")
    with open("data/processed/difficulty_labels.jsonl") as f:
        all_data = [json.loads(line) for line in f]

    # Filter Level 2 (fusion) tasks
    fusion_tasks = [d for d in all_data if d["difficulty"] == 2]
    test_tasks = fusion_tasks[:num_test_tasks]

    print(f"Testing on {len(test_tasks)} Level 2 (fusion) tasks")

    # Evaluate
    results = {
        'valid': [],
        'compiled': [],
        'correct': [],
        'tasks': [],
    }

    print("\nGenerating and evaluating...")
    for task in tqdm(test_tasks):
        # Generate code
        instruction = f"""Write a custom Triton kernel to optimize this PyTorch code:

```python
{task['pytorch_code']}
```

Write an optimized Triton kernel."""

        inputs = tokenizer(instruction, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        code = generated[len(instruction):]

        # Evaluate with staged funnel
        result = evaluator.evaluate_with_funnel(
            code,
            task['pytorch_code'],
            []  # Mock test inputs
        )

        # Record results
        passed_stage = result['passed_stage']
        valid = passed_stage != 'none'
        compiled = passed_stage in ['compile', 'tiny_run', 'full_run', 'timing']
        correct = result['code_reward'] > 0.5

        results['valid'].append(valid)
        results['compiled'].append(compiled)
        results['correct'].append(correct)
        results['tasks'].append({
            'id': task['id'],
            'stage': passed_stage,
            'reward': result['code_reward'],
        })

    # Compute metrics
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)

    valid_rate = sum(results['valid']) / len(results['valid']) * 100
    compiled_rate = sum(results['compiled']) / len(results['compiled']) * 100
    correct_rate = sum(results['correct']) / len(results['correct']) * 100

    print(f"\nTested on {len(test_tasks)} Level 2 (fusion) tasks:")
    print(f"  Valid:    {valid_rate:.1f}% ({sum(results['valid'])}/{len(results['valid'])})")
    print(f"  Compiled: {compiled_rate:.1f}% ({sum(results['compiled'])}/{len(results['compiled'])})")
    print(f"  Correct:  {correct_rate:.1f}% ({sum(results['correct'])}/{len(results['correct'])})")

    # Save detailed results
    output_file = f"outputs/eval_{model_path.split('/')[-1]}.json"
    import os
    os.makedirs("outputs", exist_ok=True)

    with open(output_file, "w") as f:
        json.dump({
            'model_path': model_path,
            'num_tasks': len(test_tasks),
            'metrics': {
                'valid_rate': valid_rate,
                'compiled_rate': compiled_rate,
                'correct_rate': correct_rate,
            },
            'detailed_results': results['tasks'],
        }, f, indent=2)

    print(f"\n✓ Detailed results saved to {output_file}")

    return {
        'valid': valid_rate,
        'compiled': compiled_rate,
        'correct': correct_rate,
    }


def compare_models():
    """Compare SFT baseline vs RL model"""
    print("="*80)
    print("COMPARING MODELS")
    print("="*80)

    # Evaluate both
    print("\n[1/2] Evaluating SFT baseline...")
    sft_results = evaluate_model("checkpoints/sft_final", num_test_tasks=20)

    print("\n[2/2] Evaluating RL model...")
    rl_results = evaluate_model("checkpoints/rl_final", num_test_tasks=20)

    # Compare
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)

    print(f"\n{'Metric':<15} {'SFT':<10} {'RL':<10} {'Improvement':<12}")
    print("-"*50)

    for metric in ['valid', 'compiled', 'correct']:
        sft_val = sft_results[metric]
        rl_val = rl_results[metric]
        improvement = rl_val - sft_val

        print(f"{metric.capitalize():<15} {sft_val:>6.1f}%   {rl_val:>6.1f}%   {improvement:>+6.1f}%")

    print("\n" + "="*80)
    if rl_results['correct'] > sft_results['correct']:
        print("✓ RL IMPROVED OVER SFT BASELINE")
        print(f"  Correctness: {sft_results['correct']:.1f}% → {rl_results['correct']:.1f}% (+{rl_results['correct'] - sft_results['correct']:.1f}%)")
    else:
        print("⚠ No improvement (may need more training)")
    print("="*80)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="checkpoints/rl_final")
    parser.add_argument("--compare", action="store_true", help="Compare SFT vs RL")
    parser.add_argument("--num_tasks", type=int, default=20)
    args = parser.parse_args()

    if args.compare:
        compare_models()
    else:
        evaluate_model(args.model_path, args.num_tasks)


if __name__ == "__main__":
    main()
# Enhanced testing
