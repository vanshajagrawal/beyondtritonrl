"""
Comprehensive Configuration System for TritonRL

This module provides validated, environment-aware configuration classes
with automatic parameter validation, type checking, and optimal defaults.
Supports different training phases (SFT, RL, Evaluation) with hardware-specific tuning.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class DataConfig:
    """
    Configuration for data processing and generation.

    Handles dataset creation, difficulty labeling, and synthetic data generation
    with optimized parameters for different model capabilities.
    """
    # Data paths
    kernelbook_path: str = field(default="GPUMODE/KernelBook")
    processed_data_dir: str = field(default="data/processed")

    # Data generation parameters
    num_variations: int = field(default=5, metadata={"help": "Number of variations per base task"})
    max_tasks: int = field(default=11621, metadata={"help": "Maximum tasks to process"})

    # DeepSeek generation parameters
    deepseek_model: str = field(default="deepseek-reasoner", metadata={"help": "Model for task generation"})
    deepseek_temperature: float = field(default=0.6, metadata={"help": "Creativity for generation"})
    deepseek_top_p: float = field(default=0.95, metadata={"help": "Nucleus sampling parameter"})

    # Labeling parameters
    labeler_model: str = field(default="Qwen/Qwen2.5-72B-Instruct", metadata={"help": "Model for difficulty labeling"})
    labeler_temperature: float = field(default=0.7, metadata={"help": "Consistency for labeling"})
    labeler_top_p: float = field(default=0.8, metadata={"help": "Focus for labeling"})

    # Validation
    seed: int = field(default=42, metadata={"help": "Random seed for reproducibility"})

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.num_variations < 1:
            raise ValueError("num_variations must be >= 1")
        if self.max_tasks < 1:
            raise ValueError("max_tasks must be >= 1")
        if not (0 < self.deepseek_temperature <= 2.0):
            raise ValueError("deepseek_temperature must be in (0, 2.0]")
        if not (0 < self.labeler_temperature <= 2.0):
            raise ValueError("labeler_temperature must be in (0, 2.0]")

        # Create directories
        Path(self.processed_data_dir).mkdir(parents=True, exist_ok=True)

@dataclass
class SFTConfig:
    """
    Supervised Fine-Tuning configuration with hardware-aware optimization.

    Automatically adjusts parameters based on available hardware and provides
    validation for training stability and performance.
    """
    # Model configuration
    base_model: str = field(default="Qwen/Qwen2.5-Coder-7B-Instruct",
                           metadata={"help": "Base model for fine-tuning"})
    output_dir: str = field(default="checkpoints/sft",
                           metadata={"help": "Directory to save checkpoints"})

    # Training parameters
    num_epochs: int = field(default=2, metadata={"help": "Number of training epochs"})
    batch_size: int = field(default_factory=lambda: SFTConfig._get_optimal_batch_size(),
                           metadata={"help": "Training batch size (auto-adjusted)"})
    learning_rate: float = field(default=1e-5, metadata={"help": "Learning rate"})
    max_seq_length: int = field(default=12288, metadata={"help": "Maximum sequence length"})

    # Optimization parameters
    gradient_accumulation_steps: int = field(default=1, metadata={"help": "Gradient accumulation steps"})
    warmup_steps: int = field(default=100, metadata={"help": "Learning rate warmup steps"})
    weight_decay: float = field(default=0.01, metadata={"help": "Weight decay for regularization"})

    # Monitoring and saving
    logging_steps: int = field(default=10, metadata={"help": "Logging frequency"})
    save_steps: int = field(default=500, metadata={"help": "Checkpoint saving frequency"})
    save_total_limit: int = field(default=3, metadata={"help": "Maximum checkpoints to keep"})

    # Advanced options
    use_fp16: bool = field(default=False, metadata={"help": "Use FP16 training"})
    use_bf16: bool = field(default=True, metadata={"help": "Use BF16 training"})
    dataloader_num_workers: int = field(default=4, metadata={"help": "DataLoader workers"})

    @staticmethod
    def _get_optimal_batch_size() -> int:
        """Determine optimal batch size based on available GPU memory."""
        if not hasattr(SFTConfig, '_cached_batch_size'):
            try:
                import torch
                if torch.cuda.is_available():
                    # Estimate based on GPU memory
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                    if gpu_memory > 40:  # High-end GPUs
                        SFTConfig._cached_batch_size = 16
                    elif gpu_memory > 20:  # Mid-range GPUs
                        SFTConfig._cached_batch_size = 8
                    else:  # Low-end GPUs
                        SFTConfig._cached_batch_size = 4
                else:
                    SFTConfig._cached_batch_size = 2  # CPU fallback
            except:
                SFTConfig._cached_batch_size = 8  # Default fallback

        return SFTConfig._cached_batch_size

    def __post_init__(self):
        """Validate SFT configuration."""
        if self.num_epochs < 1:
            raise ValueError("num_epochs must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.max_seq_length < 1:
            raise ValueError("max_seq_length must be >= 1")
        if self.use_fp16 and self.use_bf16:
            raise ValueError("Cannot use both FP16 and BF16 simultaneously")

        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

@dataclass
class RLConfig:
    """
    Reinforcement Learning configuration with advanced optimization.

    Provides comprehensive parameter tuning for GRPO training with
    hierarchical rewards, exploration strategies, and stability controls.
    """
    # Model paths
    sft_model_path: str = field(default="checkpoints/sft",
                               metadata={"help": "Path to SFT model checkpoint"})
    output_dir: str = field(default="checkpoints/rl",
                           metadata={"help": "Directory to save RL checkpoints"})

    # Core training parameters
    num_epochs: int = field(default=2, metadata={"help": "Number of training epochs"})
    batch_size: int = field(default=8, metadata={"help": "Training batch size per GPU"})
    learning_rate: float = field(default=1e-6, metadata={"help": "Learning rate for policy updates"})

    # Sequence lengths
    max_prompt_length: int = field(default=2048, metadata={"help": "Maximum prompt length"})
    max_response_length: int = field(default=16384, metadata={"help": "Maximum response length"})

    # GRPO-specific parameters
    group_size: int = field(default=4, metadata={"help": "Group size for GRPO"})
    alpha: float = field(default=0.1, metadata={"help": "KL penalty coefficient"})
    epsilon: float = field(default=0.2, metadata={"help": "Clipping parameter for PPO-style updates"})

    # Data composition
    data_mix: Tuple[float, float] = field(default=(0.7, 0.3),
                                         metadata={"help": "Mix of (easy, hard) tasks"})

    # Optimization parameters
    gradient_accumulation_steps: int = field(default=4, metadata={"help": "Gradient accumulation steps"})
    warmup_steps: int = field(default=50, metadata={"help": "Learning rate warmup steps"})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Maximum gradient norm"})

    # Advanced RL parameters
    entropy_coef: float = field(default=0.01, metadata={"help": "Entropy bonus coefficient"})
    value_loss_coef: float = field(default=0.5, metadata={"help": "Value function loss coefficient"})
    gamma: float = field(default=1.0, metadata={"help": "Discount factor"})
    gae_lambda: float = field(default=0.95, metadata={"help": "GAE lambda parameter"})

    # Exploration and stability
    exploration_noise: float = field(default=0.1, metadata={"help": "Exploration noise for generation"})
    reward_clip: float = field(default=10.0, metadata={"help": "Reward clipping threshold"})

    # Monitoring
    logging_steps: int = field(default=5, metadata={"help": "Logging frequency"})
    eval_steps: int = field(default=50, metadata={"help": "Evaluation frequency"})
    save_steps: int = field(default=100, metadata={"help": "Checkpoint saving frequency"})

    # Reproducibility
    seed: int = field(default=42, metadata={"help": "Random seed for reproducibility"})

    def __post_init__(self):
        """Validate RL configuration and set derived parameters."""
        # Basic validation
        if self.num_epochs < 1:
            raise ValueError("num_epochs must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if not (0 <= self.alpha <= 1):
            raise ValueError("alpha must be in [0, 1]")
        if not (0 <= self.epsilon <= 1):
            raise ValueError("epsilon must be in [0, 1]")

        # Data mix validation
        p1, p2 = self.data_mix
        if abs(p1 + p2 - 1.0) > 1e-6:
            raise ValueError(f"data_mix probabilities must sum to 1, got {p1 + p2}")

        # Sequence length validation
        if self.max_response_length <= self.max_prompt_length:
            raise ValueError("max_response_length must be > max_prompt_length")

        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Log configuration summary
        logger.info(f"RL Config: epochs={self.num_epochs}, batch_size={self.batch_size}, "
                   f"lr={self.learning_rate}, data_mix={self.data_mix}")

@dataclass
class EvalConfig:
    """
    Evaluation configuration with performance and reliability optimizations.

    Configures evaluation parameters for different testing scenarios and
    hardware environments with automatic optimization.
    """
    # Data and model paths
    kernelbench_path: str = field(default="ScalingIntelligence/KernelBench",
                                 metadata={"help": "Path to KernelBench dataset"})
    model_path: str = field(default="checkpoints/rl_final",
                           metadata={"help": "Path to model for evaluation"})

    # Evaluation parameters
    num_samples: int = field(default=20, metadata={"help": "Number of samples to evaluate"})
    temperature: float = field(default=0.7, metadata={"help": "Generation temperature"})
    timeout: int = field(default=60, metadata={"help": "Evaluation timeout in seconds"})

    # Hardware configuration
    device: str = field(default_factory=lambda: "cuda" if EvalConfig._cuda_available() else "cpu")
    num_workers: int = field(default=4, metadata={"help": "Number of evaluation workers"})

    # Advanced options
    use_fast_tokenizer: bool = field(default=True, metadata={"help": "Use fast tokenizer"})
    compile_model: bool = field(default=True, metadata={"help": "Compile model for faster inference"})
    enable_progress_bar: bool = field(default=True, metadata={"help": "Show progress bar"})

    @staticmethod
    def _cuda_available() -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def __post_init__(self):
        """Validate evaluation configuration."""
        if self.num_samples < 1:
            raise ValueError("num_samples must be >= 1")
        if not (0 < self.temperature <= 2.0):
            raise ValueError("temperature must be in (0, 2.0]")
        if self.timeout < 1:
            raise ValueError("timeout must be >= 1")

        # Adjust parameters based on hardware
        if self.device == "cpu":
            self.num_workers = 1  # Reduce workers for CPU
            self.compile_model = False  # Compilation not beneficial on CPU


@dataclass
class ConfigManager:
    """
    Centralized configuration management with environment detection.

    Automatically optimizes configurations based on available hardware
    and provides validation across all training phases.
    """

    # Environment detection
    cuda_available: bool = field(default_factory=lambda: ConfigManager._detect_cuda())
    gpu_memory_gb: float = field(default_factory=lambda: ConfigManager._get_gpu_memory())
    num_cpus: int = field(default_factory=lambda: os.cpu_count() or 4)

    # Configuration instances
    data: DataConfig = field(default_factory=DataConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    @staticmethod
    def _detect_cuda() -> bool:
        """Detect CUDA availability."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    @staticmethod
    def _get_gpu_memory() -> float:
        """Get available GPU memory in GB."""
        if not ConfigManager._detect_cuda():
            return 0.0
        try:
            import torch
            return torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except:
            return 0.0

    def optimize_for_hardware(self):
        """
        Optimize all configurations for the detected hardware.

        Automatically adjusts batch sizes, workers, and other parameters
        based on available CPU/GPU resources.
        """
        logger.info(f"Optimizing configurations for hardware: "
                   f"CUDA={self.cuda_available}, GPU_MEM={self.gpu_memory_gb:.1f}GB, "
                   f"CPUs={self.num_cpus}")

        # Optimize based on GPU memory
        if self.gpu_memory_gb > 40:
            # High-end GPU
            self.sft.batch_size = 16
            self.rl.batch_size = 8
            self.sft.dataloader_num_workers = 8
        elif self.gpu_memory_gb > 20:
            # Mid-range GPU
            self.sft.batch_size = 8
            self.rl.batch_size = 4
            self.sft.dataloader_num_workers = 4
        else:
            # Low-end GPU or CPU
            self.sft.batch_size = 4
            self.rl.batch_size = 2
            self.sft.dataloader_num_workers = 2

        # CPU optimizations
        if self.num_cpus <= 4:
            self.sft.dataloader_num_workers = 2
            self.eval.num_workers = 2

        logger.info("Hardware optimization complete")

    def validate_all(self):
        """Validate all configuration instances."""
        configs = [self.data, self.sft, self.rl, self.eval]
        for config in configs:
            if hasattr(config, '__post_init__'):
                # Re-run validation
                config.__post_init__()

        logger.info("All configurations validated successfully")

    @classmethod
    def create_optimized(cls) -> 'ConfigManager':
        """Create and return an optimized configuration manager."""
        manager = cls()
        manager.optimize_for_hardware()
        manager.validate_all()
        return manager
