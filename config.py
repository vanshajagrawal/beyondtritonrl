from dataclasses import dataclass
from typing import Optional

@dataclass
class DataConfig:
    kernelbook_path: str = "GPUMODE/KernelBook"
    num_variations: int = 5
    max_tasks: int = 11621
    deepseek_model: str = "deepseek-reasoner"
    deepseek_temperature: float = 0.6
    deepseek_top_p: float = 0.95
    labeler_model: str = "Qwen/Qwen2.5-72B-Instruct"
    labeler_temperature: float = 0.7
    labeler_top_p: float = 0.8

@dataclass
class SFTConfig:
    base_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    output_dir: str = "checkpoints/sft"
    num_epochs: int = 2
    batch_size: int = 16
    learning_rate: float = 1e-5
    max_seq_length: int = 12288
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 500

@dataclass
class RLConfig:
    sft_model_path: str = "checkpoints/sft"
    output_dir: str = "checkpoints/rl"
    num_epochs: int = 2
    batch_size: int = 32
    learning_rate: float = 1e-6
    max_prompt_length: int = 2048
    max_response_length: int = 16384
    group_size: int = 4
    alpha: float = 0.1
    epsilon: float = 0.2
    data_mix: tuple = (1.0, 0.0)  # (p1, p2) for Level 1 and Level 2
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 50
    logging_steps: int = 5

@dataclass
class EvalConfig:
    kernelbench_path: str = "ScalingIntelligence/KernelBench"
    num_samples: int = 10
    temperature: float = 0.8
    device: str = "cuda"
    timeout: int = 30
# Config updates
