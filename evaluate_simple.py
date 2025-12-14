#!/usr/bin/env python3
"""
Enhanced Evaluation Script with Comprehensive Metrics

This module provides robust evaluation capabilities for TritonRL models with:
- Multiple evaluation metrics (validity, compilation, correctness, performance)
- Statistical analysis and confidence intervals
- Error handling and recovery
- Detailed logging and progress tracking
- Support for different evaluation modes and configurations
"""

import logging
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from statistics import mean, stdev
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from extensions import HardenedVerifier, StagedEvaluator
from extensions.config import ExtensionConfig
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics with statistical analysis."""
    valid_rate: float
    compiled_rate: float
    correct_rate: float
    performance_score: float
    confidence_interval: Tuple[float, float]
    sample_size: int

@dataclass
class EvaluationResult:
    """Detailed result for a single evaluation."""
    task_id: str
    passed_stage: str
    code_reward: float
    execution_time: float
    generated_code: str
    error_message: Optional[str] = None

def evaluate_model(model_path: str, num_test_tasks: int = 20) -> EvaluationMetrics:
    """
    Evaluate model on test tasks with comprehensive metrics.

    Args:
        model_path: Path to trained model directory
        num_test_tasks: Number of tasks to test (default: 20 for budget constraints)

    Returns:
        EvaluationMetrics: Comprehensive evaluation results with statistics

    Raises:
        FileNotFoundError: If model or test data files don't exist
        RuntimeError: If evaluation fails
    """
    logger.info("="*80)
    logger.info(f"EVALUATING MODEL: {model_path}")
    logger.info("="*80)

    start_time = time.time()

    # Load model with error handling
    try:
        logger.info("Loading model and tokenizer...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        # Handle missing pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model.eval()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Model loading failed: {e}")

    # Setup verifier with extensions
    try:
        logger.info("Setting up verifier with extensions...")
        ext_config = ExtensionConfig(
            enable_multi_input=True,
            enable_staged_eval=True,
            enable_calibrated_timing=False,  # Skip for faster evaluation
        )
        verifier = HardenedVerifier(ext_config)
        evaluator = StagedEvaluator(verifier, ext_config)
        logger.info("Verifier setup complete")
    except Exception as e:
        logger.error(f"Failed to setup verifier: {e}")
        raise RuntimeError(f"Verifier setup failed: {e}")

    # Load and filter test tasks
    try:
        logger.info(f"Loading test tasks from dataset...")
        with open("data/processed/difficulty_labels.jsonl", 'r') as f:
            all_data = [json.loads(line) for line in f]

        # Filter Level 2 (fusion) tasks with validation
        fusion_tasks = [d for d in all_data if d.get("difficulty") == 2]
        if len(fusion_tasks) < num_test_tasks:
            logger.warning(f"Only {len(fusion_tasks)} fusion tasks available, requested {num_test_tasks}")
            num_test_tasks = len(fusion_tasks)

        test_tasks = fusion_tasks[:num_test_tasks]
        logger.info(f"Testing on {len(test_tasks)} Level 2 (fusion) tasks")
    except FileNotFoundError:
        logger.error("Test data file not found")
        raise FileNotFoundError("data/processed/difficulty_labels.jsonl not found")
    except Exception as e:
        logger.error(f"Failed to load test data: {e}")
        raise RuntimeError(f"Test data loading failed: {e}")

    # Initialize results tracking
    evaluation_results: List[EvaluationResult] = []
    generation_times: List[float] = []

    logger.info("Starting evaluation of {} tasks...".format(len(test_tasks)))

    for task in tqdm(test_tasks, desc="Evaluating tasks"):
        task_start_time = time.time()

        try:
            # Generate optimized prompt
            instruction = f"""Write a custom Triton kernel to optimize this PyTorch code:

```python
{task['pytorch_code']}
```

Requirements:
- Use Triton language constructs efficiently
- Optimize for memory access patterns
- Minimize shared memory usage
- Ensure correctness of the optimization

Write an optimized Triton kernel."""

            # Tokenize with error handling
            inputs = tokenizer(
                instruction,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            ).to(model.device)

            # Generate code with controlled parameters
            generation_start = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    temperature=0.7,  # Slightly lower for more consistent results
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1,  # Reduce repetition
                )

            generation_time = time.time() - generation_start
            generation_times.append(generation_time)

            # Extract generated code
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the instruction part to get just the generated code
            code_start = generated_text.find("```python")  # Look for code block start
            if code_start != -1:
                code = generated_text[code_start:]
                # Extract just the Triton kernel code
                code_lines = code.split('\n')
                triton_code = []
                in_code_block = False
                for line in code_lines:
                    if '```' in line:
                        in_code_block = not in_code_block
                        continue
                    if in_code_block:
                        triton_code.append(line)
                code = '\n'.join(triton_code)
            else:
                # Fallback: extract everything after the instruction
                code = generated_text[len(instruction):].strip()

            # Evaluate with comprehensive error handling
            try:
                result = evaluator.evaluate_with_funnel(
                    code,
                    task['pytorch_code'],
                    []  # Mock test inputs for basic evaluation
                )

                passed_stage = result.get('passed_stage', 'none')
                code_reward = result.get('code_reward', 0.0)

                # Enhanced validation logic
                valid = passed_stage != 'none'
                compiled = passed_stage in ['compile', 'tiny_run', 'full_run', 'timing']
                correct = code_reward > 0.5

                eval_result = EvaluationResult(
                    task_id=task.get('id', 'unknown'),
                    passed_stage=passed_stage,
                    code_reward=code_reward,
                    execution_time=time.time() - task_start_time,
                    generated_code=code[:500] + "..." if len(code) > 500 else code  # Truncate for storage
                )

            except Exception as eval_error:
                logger.warning(f"Evaluation failed for task {task.get('id', 'unknown')}: {eval_error}")
                eval_result = EvaluationResult(
                    task_id=task.get('id', 'unknown'),
                    passed_stage='error',
                    code_reward=0.0,
                    execution_time=time.time() - task_start_time,
                    generated_code=code[:500] + "..." if len(code) > 500 else code,
                    error_message=str(eval_error)
                )
                valid = compiled = correct = False

            evaluation_results.append(eval_result)

        except Exception as task_error:
            logger.error(f"Task evaluation failed: {task_error}")
            evaluation_results.append(EvaluationResult(
                task_id=task.get('id', 'unknown'),
                passed_stage='error',
                code_reward=0.0,
                execution_time=time.time() - task_start_time,
                generated_code="",
                error_message=str(task_error)
            ))

    # Calculate comprehensive metrics
    logger.info("Calculating evaluation metrics...")

    # Extract boolean results
    valid_results = [r.passed_stage != 'none' and r.passed_stage != 'error' for r in evaluation_results]
    compiled_results = [r.passed_stage in ['compile', 'tiny_run', 'full_run', 'timing'] for r in evaluation_results]
    correct_results = [r.code_reward > 0.5 for r in evaluation_results]

    # Calculate rates
    valid_rate = sum(valid_results) / len(valid_results) * 100
    compiled_rate = sum(compiled_results) / len(compiled_results) * 100
    correct_rate = sum(correct_results) / len(correct_results) * 100

    # Calculate performance score (weighted combination)
    performance_score = (valid_rate * 0.3 + compiled_rate * 0.3 + correct_rate * 0.4)

    # Calculate confidence interval for correctness (most important metric)
    if len(correct_results) > 1:
        correct_mean = mean(correct_results)
        if len(correct_results) > 1:
            correct_std = stdev(correct_results)
            # 95% confidence interval
            margin = 1.96 * (correct_std / (len(correct_results) ** 0.5))
            confidence_interval = (max(0, correct_mean - margin), min(1, correct_mean + margin))
        else:
            confidence_interval = (correct_mean, correct_mean)
    else:
        confidence_interval = (correct_rate / 100, correct_rate / 100)

    # Convert back to percentages
    confidence_interval = (confidence_interval[0] * 100, confidence_interval[1] * 100)

    # Display results
    logger.info("\n" + "="*80)
    logger.info("ENHANCED EVALUATION RESULTS")
    logger.info("="*80)

    logger.info(f"\nTested on {len(test_tasks)} Level 2 (fusion) tasks:")
    logger.info(f"  Valid:       {valid_rate:.1f}% ({sum(valid_results)}/{len(valid_results)})")
    logger.info(f"  Compiled:    {compiled_rate:.1f}% ({sum(compiled_results)}/{len(compiled_results)})")
    logger.info(f"  Correct:     {correct_rate:.1f}% ({sum(correct_results)}/{len(correct_results)})")
    logger.info(f"  Performance: {performance_score:.1f}/100 (weighted score)")

    if len(correct_results) > 1:
        logger.info(f"  Confidence:  [{confidence_interval[0]:.1f}%, {confidence_interval[1]:.1f}%] (95% CI)")

    # Calculate timing statistics
    total_time = time.time() - start_time
    avg_generation_time = mean(generation_times) if generation_times else 0
    logger.info(".2f")
    logger.info(".3f")

    # Save comprehensive results
    import os
    os.makedirs("outputs", exist_ok=True)
    output_file = f"outputs/eval_{model_path.split('/')[-1]}_{int(time.time())}.json"

    results_data = {
        'model_path': model_path,
        'evaluation_timestamp': time.time(),
        'num_tasks': len(test_tasks),
        'metrics': {
            'valid_rate': valid_rate,
            'compiled_rate': compiled_rate,
            'correct_rate': correct_rate,
            'performance_score': performance_score,
            'confidence_interval_95': confidence_interval,
        },
        'timing': {
            'total_evaluation_time': total_time,
            'average_generation_time': avg_generation_time,
            'tasks_per_second': len(test_tasks) / total_time if total_time > 0 else 0,
        },
        'detailed_results': [
            {
                'task_id': r.task_id,
                'passed_stage': r.passed_stage,
                'code_reward': r.code_reward,
                'execution_time': r.execution_time,
                'error_message': r.error_message,
            }
            for r in evaluation_results
        ],
    }

    with open(output_file, "w") as f:
        json.dump(results_data, f, indent=2)

    logger.info(f"\n✓ Comprehensive results saved to {output_file}")

    return EvaluationMetrics(
        valid_rate=valid_rate,
        compiled_rate=compiled_rate,
        correct_rate=correct_rate,
        performance_score=performance_score,
        confidence_interval=confidence_interval,
        sample_size=len(test_tasks)
    )


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
