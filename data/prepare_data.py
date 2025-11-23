import json
import os
from typing import List, Dict
from datasets import load_dataset
from tqdm import tqdm
import openai
from prompts import SFT_INSTRUCTION_TEMPLATE, DIFFICULTY_LABELING_TEMPLATE

class DataPreparer:
    def __init__(self, config):
        self.config = config
        self.client = openai.OpenAI()

    def load_kernelbook(self) -> List[Dict]:
        """Load PyTorch reference codes from KernelBook"""
        print("Loading KernelBook dataset...")
        dataset = load_dataset(self.config.kernelbook_path, split="train")

        pytorch_codes = []
        for item in dataset:
            if self._is_executable(item.get("pytorch_code", "")):
                pytorch_codes.append({
                    "id": item.get("id", len(pytorch_codes)),
                    "pytorch_code": item["pytorch_code"],
                })

                if len(pytorch_codes) >= self.config.max_tasks:
                    break

        print(f"Loaded {len(pytorch_codes)} valid PyTorch codes")
        return pytorch_codes

    def _is_executable(self, code: str) -> bool:
        """Basic validation of PyTorch code"""
        required = ["class Model", "def forward", "def get_inputs"]
        return all(req in code for req in required)

    def generate_triton_solutions(self, pytorch_codes: List[Dict]) -> List[Dict]:
        """Generate Triton solutions using DeepSeek-R1"""
        print("Generating Triton solutions with DeepSeek-R1...")

        sft_data = []
        for item in tqdm(pytorch_codes):
            instruction = SFT_INSTRUCTION_TEMPLATE.format(
                pytorch_code=item["pytorch_code"]
            )

            # Generate multiple variations
            for i in range(self.config.num_variations):
                try:
                    response = self.client.chat.completions.create(
                        model=self.config.deepseek_model,
                        messages=[{"role": "user", "content": instruction}],
                        temperature=self.config.deepseek_temperature,
                        top_p=self.config.deepseek_top_p,
                        max_tokens=16384,
                    )

                    output = response.choices[0].message.content

                    sft_data.append({
                        "id": f"{item['id']}_{i}",
                        "instruction": instruction,
                        "output": output,
                        "pytorch_code": item["pytorch_code"],
                    })

                except Exception as e:
                    print(f"Error generating solution for {item['id']}_{i}: {e}")
                    continue

        print(f"Generated {len(sft_data)} training samples")
        return sft_data

    def label_difficulty(self, pytorch_codes: List[Dict]) -> List[Dict]:
        """Label task difficulty using Qwen3-235B"""
        print("Labeling task difficulty...")

        labeled_data = []
        for item in tqdm(pytorch_codes):
            prompt = DIFFICULTY_LABELING_TEMPLATE.format(
                pytorch_code=item["pytorch_code"]
            )

            try:
                response = self.client.chat.completions.create(
                    model=self.config.labeler_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.labeler_temperature,
                    top_p=self.config.labeler_top_p,
                    max_tokens=10,
                )

                level = int(response.choices[0].message.content.strip())

                labeled_data.append({
                    "id": item["id"],
                    "pytorch_code": item["pytorch_code"],
                    "difficulty": level,
                })

            except Exception as e:
                print(f"Error labeling {item['id']}: {e}")
                labeled_data.append({
                    "id": item["id"],
                    "pytorch_code": item["pytorch_code"],
                    "difficulty": 1,  # Default to Level 1
                })

        print(f"Labeled {len(labeled_data)} tasks")
        return labeled_data

    def save_datasets(self, sft_data: List[Dict], labeled_data: List[Dict]):
        """Save prepared datasets"""
        os.makedirs("data/processed", exist_ok=True)

        with open("data/processed/sft_train.jsonl", "w") as f:
            for item in sft_data:
                f.write(json.dumps(item) + "\n")

        with open("data/processed/difficulty_labels.jsonl", "w") as f:
            for item in labeled_data:
                f.write(json.dumps(item) + "\n")

        print("Saved datasets to data/processed/")

def main():
    from config import DataConfig

    config = DataConfig()
    preparer = DataPreparer(config)

    # Load PyTorch codes
    pytorch_codes = preparer.load_kernelbook()

    # Generate SFT training data
    sft_data = preparer.generate_triton_solutions(pytorch_codes)

    # Label difficulty for RL training
    labeled_data = preparer.label_difficulty(pytorch_codes)

    # Save datasets
    preparer.save_datasets(sft_data, labeled_data)

if __name__ == "__main__":
    main()
