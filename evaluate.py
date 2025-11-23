import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from config import EvalConfig
from verifiers import TritonVerifier
from data.prompts import (
    EVAL_INSTRUCTION_TEMPLATE,
    EVAL_ONE_SHOT_EXAMPLE_PYTORCH,
    EVAL_ONE_SHOT_EXAMPLE_TRITON,
)

class KernelBenchEvaluator:
    def __init__(self, model_path: str, config: EvalConfig):
        self.config = config
        self.verifier = TritonVerifier()

        print(f"Loading model: {model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        self.model.eval()

    def load_kernelbench(self, level: int = 1):
        """Load KernelBench evaluation tasks"""
        print(f"Loading KernelBench Level {level}...")
        dataset = load_dataset(self.config.kernelbench_path, split="train")

        # Filter by level
        tasks = [item for item in dataset if item.get("level") == level]
        print(f"Loaded {len(tasks)} tasks")
        return tasks

    def generate_solutions(self, task: dict, num_samples: int = 10):
        """Generate multiple Triton solutions for a task"""
        prompt = EVAL_INSTRUCTION_TEMPLATE.format(
            example_pytorch=EVAL_ONE_SHOT_EXAMPLE_PYTORCH,
            example_triton=EVAL_ONE_SHOT_EXAMPLE_TRITON,
            target_pytorch=task["pytorch_code"],
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        solutions = []
        for _ in range(num_samples):
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=16384,
                    temperature=self.config.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            solution = generated[len(prompt):]
            solutions.append(solution)

        return solutions

    def evaluate_task(self, task: dict, solutions: list):
        """Evaluate solutions for a single task"""
        results = {
            "task_id": task.get("id"),
            "valid": [],
            "compiled": [],
            "correct": [],
            "speedup": [],
        }

        for solution in solutions:
            # Syntax check
            syntax = self.verifier.syntax_check(solution)

            # Functionality check
            func = self.verifier.functionality_check(solution)

            results["valid"].append(syntax and func)

            if not (syntax and func):
                results["compiled"].append(False)
                results["correct"].append(False)
                results["speedup"].append(0.0)
                continue

            # Compilation check
            compiled = self.verifier.compilation_check(solution)
            results["compiled"].append(compiled)

            if not compiled:
                results["correct"].append(False)
                results["speedup"].append(0.0)
                continue

            # Get test inputs
            namespace = {}
            exec(task["pytorch_code"], namespace)
            test_inputs = namespace["get_inputs"]()

            # Correctness check
            correct = self.verifier.correctness_check(
                solution,
                task["pytorch_code"],
                test_inputs,
            )
            results["correct"].append(correct)

            # Speedup measurement
            if correct:
                speedup = self.verifier.speedup_metric(
                    solution,
                    task["pytorch_code"],
                    test_inputs,
                )
                results["speedup"].append(speedup)
            else:
                results["speedup"].append(0.0)

        return results

    def compute_metrics(self, all_results: list, k: int = 10):
        """Compute pass@k metrics"""
        def pass_at_k(results_list, key):
            passed = sum(1 for r in results_list if any(r[key][:k]))
            return passed / len(results_list) * 100

        metrics = {
            f"pass@{k}_valid": pass_at_k(all_results, "valid"),
            f"pass@{k}_compiled": pass_at_k(all_results, "compiled"),
            f"pass@{k}_correct": pass_at_k(all_results, "correct"),
            f"pass@{k}_fast1": sum(1 for r in all_results if any(s > 1.0 for s in r["speedup"][:k])) / len(all_results) * 100,
            f"pass@{k}_fast2": sum(1 for r in all_results if any(s > 2.0 for s in r["speedup"][:k])) / len(all_results) * 100,
            "mean_speedup": sum(max(r["speedup"][:k]) for r in all_results) / len(all_results),
        }

        return metrics

    def evaluate(self, level: int = 1):
        """Run full evaluation on KernelBench"""
        tasks = self.load_kernelbench(level)

        all_results = []
        for task in tqdm(tasks, desc=f"Evaluating Level {level}"):
            solutions = self.generate_solutions(task, num_samples=self.config.num_samples)
            results = self.evaluate_task(task, solutions)
            all_results.append(results)

        # Compute metrics
        metrics = self.compute_metrics(all_results, k=self.config.num_samples)

        print(f"\n=== KernelBench Level {level} Results ===")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.2f}")

        # Save results
        output_file = f"outputs/results_level{level}.json"
        with open(output_file, "w") as f:
            json.dump({
                "metrics": metrics,
                "detailed_results": all_results,
            }, f, indent=2)

        print(f"\nResults saved to {output_file}")
        return metrics

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--level", type=int, default=1, choices=[1, 2, 3])
    args = parser.parse_args()

    config = EvalConfig()
    evaluator = KernelBenchEvaluator(args.model_path, config)
    evaluator.evaluate(level=args.level)

if __name__ == "__main__":
    main()
