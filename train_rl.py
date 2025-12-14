import torch
import json
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from verl import RewardManager, GRPOTrainer
from config import RLConfig
from verifiers import TritonVerifier

class HierarchicalRewardManager(RewardManager):
    """Hierarchical reward decomposition for plan and code tokens"""

    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.verifier = TritonVerifier()

    def compute_rewards(self, prompts, outputs, refs):
        """
        Compute hierarchical rewards for plan and code tokens

        Args:
            prompts: Task instructions with PyTorch reference
            outputs: Generated outputs (plan + code)
            refs: PyTorch reference codes

        Returns:
            plan_rewards, code_rewards: Token-level rewards
        """
        batch_rewards = []

        for prompt, output, ref in zip(prompts, outputs, refs):
            # Split output into plan and code
            plan_tokens, code_tokens = self._split_plan_code(output)

            # Extract code for verification
            generated_code = self._extract_code(output)

            # Get test inputs
            test_inputs = self._get_test_inputs(ref)

            # Compute rewards
            r_plan = self.verifier.compute_reward(
                generated_code, ref, test_inputs, reward_type="plan"
            )

            r_code = self.verifier.compute_reward(
                generated_code, ref, test_inputs, reward_type="code"
            )

            # Assign rewards to tokens
            plan_reward_tensor = torch.full((len(plan_tokens),), r_plan)
            code_reward_tensor = torch.full((len(code_tokens),), r_code)

            batch_rewards.append({
                "plan_rewards": plan_reward_tensor,
                "code_rewards": code_reward_tensor,
                "plan_indices": plan_tokens,
                "code_indices": code_tokens,
            })

        return batch_rewards

    def _split_plan_code(self, output):
        """Split output into plan and code token indices"""
        # Find <think> tags for plan tokens
        think_start = output.find("<think>")
        think_end = output.find("</think>")

        if think_start != -1 and think_end != -1:
            plan_tokens = list(range(think_start, think_end))
            code_tokens = list(range(think_end, len(output)))
        else:
            # If no think tags, split at code block
            code_start = output.find("```python")
            if code_start != -1:
                plan_tokens = list(range(0, code_start))
                code_tokens = list(range(code_start, len(output)))
            else:
                # Default: first half is plan, second half is code
                mid = len(output) // 2
                plan_tokens = list(range(0, mid))
                code_tokens = list(range(mid, len(output)))

        return plan_tokens, code_tokens

    def _extract_code(self, output):
        """Extract Python code from output"""
        import re
        code_blocks = re.findall(r'```python\n(.*?)```', output, re.DOTALL)
        return code_blocks[0] if code_blocks else output

    def _get_test_inputs(self, pytorch_ref):
        """Extract test inputs from PyTorch reference"""
        namespace = {}
        exec(pytorch_ref, namespace)
        if 'get_inputs' in namespace:
            return namespace['get_inputs']()
        return []

class TritonRLTrainer:
    def __init__(self, config: RLConfig):
        self.config = config

        print(f"Loading SFT model: {config.sft_model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            config.sft_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.sft_model_path,
            trust_remote_code=True,
        )

        self.reward_manager = HierarchicalRewardManager(alpha=config.alpha)

    def load_data(self):
        """Load RL training data with difficulty labels"""
        print("Loading RL training data...")

        # Load difficulty labels
        with open("data/processed/difficulty_labels.jsonl") as f:
            difficulty_data = [json.loads(line) for line in f]

        # Filter by data mixing strategy
        p1, p2 = self.config.data_mix

        level1_data = [d for d in difficulty_data if d["difficulty"] == 1]
        level2_data = [d for d in difficulty_data if d["difficulty"] == 2]

        # Sample according to mixing probabilities
        import random
        sampled_data = []

        num_samples = len(difficulty_data)
        num_level1 = int(num_samples * p1)
        num_level2 = int(num_samples * p2)

        sampled_data.extend(random.sample(level1_data, min(num_level1, len(level1_data))))
        sampled_data.extend(random.sample(level2_data, min(num_level2, len(level2_data))))

        print(f"Sampled {len(sampled_data)} tasks (Level 1: {num_level1}, Level 2: {num_level2})")
        return sampled_data

    def train(self):
        """Run RL training with GRPO and hierarchical rewards"""
        dataset = self.load_data()

        # Initialize GRPO trainer
        trainer = GRPOTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            reward_manager=self.reward_manager,
            config={
                "output_dir": self.config.output_dir,
                "num_epochs": self.config.num_epochs,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "group_size": self.config.group_size,
                "alpha": self.config.alpha,
                "epsilon": self.config.epsilon,
                "max_prompt_length": self.config.max_prompt_length,
                "max_response_length": self.config.max_response_length,
            }
        )

        print("Starting RL training with hierarchical rewards...")
        trainer.train(dataset)

        print(f"Saving final model to {self.config.output_dir}")
        self.model.save_pretrained(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)

def main():
    config = RLConfig()
    trainer = TritonRLTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
# Performance improvements
