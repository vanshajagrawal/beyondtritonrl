"""
Optimized RL Training for TritonRL with Performance Enhancements

This module implements reinforcement learning training with:
- Memory-efficient training with gradient checkpointing
- Mixed precision training for faster convergence
- Optimized batch processing and data loading
- Advanced reward shaping and exploration strategies
- Comprehensive monitoring and logging
"""

import os
import logging
import torch
import json
from typing import Dict, List, Optional, Any
from contextlib import nullcontext
from transformers import AutoModelForCausalLM, AutoTokenizer
from verl import RewardManager, GRPOTrainer
from config import RLConfig
from verifiers import TritonVerifier

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Performance optimizations
torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 for faster matmul
torch.backends.cudnn.allow_tf32 = True  # Allow TF32 for faster convolutions

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
    """
    Optimized RL Trainer with performance enhancements.

    Features:
    - Memory-efficient model loading with device mapping
    - Gradient checkpointing for reduced memory usage
    - Mixed precision training support
    - Advanced data preprocessing and batching
    - Comprehensive error handling and logging
    """

    def __init__(self, config: RLConfig):
        """
        Initialize the RL trainer with performance optimizations.

        Args:
            config: RLConfig object with training parameters
        """
        self.config = config

        # Setup device and precision
        self.device = self._setup_device()
        self.use_amp = self._should_use_amp()

        # Load model with optimizations
        self.model, self.tokenizer = self._load_model_optimized()

        # Initialize reward manager
        self.reward_manager = HierarchicalRewardManager(alpha=config.alpha)

        # Setup training optimizations
        self._apply_model_optimizations()

        logger.info("RL Trainer initialized with performance optimizations")

    def _setup_device(self) -> torch.device:
        """Setup optimal device configuration."""
        if torch.cuda.is_available():
            # Use CUDA with optimizations
            device = torch.device("cuda")
            # Enable CUDA optimizations
            torch.cuda.set_device(0)
            torch.cuda.empty_cache()
            logger.info("Using CUDA device with optimizations")
        else:
            device = torch.device("cpu")
            logger.warning("CUDA not available, using CPU (performance will be limited)")
        return device

    def _should_use_amp(self) -> bool:
        """Determine if automatic mixed precision should be used."""
        return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7

    def _load_model_optimized(self) -> tuple:
        """Load model with memory and performance optimizations."""
        logger.info(f"Loading SFT model: {self.config.sft_model_path}")

        try:
            # Load model with optimizations
            model = AutoModelForCausalLM.from_pretrained(
                self.config.sft_model_path,
                torch_dtype=torch.bfloat16 if self.use_amp else torch.float32,
                device_map="auto",  # Automatic device placement
                trust_remote_code=True,
                low_cpu_mem_usage=True,  # Optimize CPU memory during loading
                load_in_8bit=False,  # Use full precision for training
                use_cache=False,  # Disable KV cache for training
            )

            tokenizer = AutoTokenizer.from_pretrained(
                self.config.sft_model_path,
                trust_remote_code=True,
            )

            # Handle missing tokens
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            logger.info("Model loaded successfully with optimizations")
            return model, tokenizer

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")

    def _apply_model_optimizations(self):
        """Apply performance optimizations to the model."""
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

        # Disable unnecessary features for training
        self.model.config.use_cache = False

        # Enable better transformer optimizations
        if hasattr(self.model.config, 'attn_implementation'):
            # Use flash attention if available
            try:
                self.model.config.attn_implementation = "flash_attention_2"
                logger.info("Flash attention enabled")
            except:
                logger.info("Using standard attention implementation")

    def load_data(self) -> List[Dict[str, Any]]:
        """
        Load and preprocess RL training data with optimized sampling.

        Returns:
            List of training samples with difficulty balancing
        """
        logger.info("Loading RL training data with optimized sampling...")

        try:
            # Load difficulty labels with error handling
            data_path = "data/processed/difficulty_labels.jsonl"
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Training data not found: {data_path}")

            with open(data_path, 'r') as f:
                difficulty_data = [json.loads(line.strip()) for line in f if line.strip()]

            logger.info(f"Loaded {len(difficulty_data)} total tasks")
        except Exception as e:
            logger.error(f"Failed to load training data: {e}")
            raise

        # Advanced data mixing with validation
        p1, p2 = self.config.data_mix
        if abs(p1 + p2 - 1.0) > 1e-6:
            logger.warning(f"Data mix probabilities don't sum to 1: {p1} + {p2} = {p1 + p2}")

        # Separate by difficulty with validation
        level1_data = [d for d in difficulty_data if d.get("difficulty") == 1]
        level2_data = [d for d in difficulty_data if d.get("difficulty") == 2]

        logger.info(f"Available tasks - Level 1: {len(level1_data)}, Level 2: {len(level2_data)}")

        # Optimized sampling with reproducibility
        import random
        random.seed(self.config.seed)  # For reproducible sampling

        target_total = len(difficulty_data)
        target_level1 = int(target_total * p1)
        target_level2 = int(target_total * p2)

        # Sample with bounds checking
        actual_level1 = min(target_level1, len(level1_data))
        actual_level2 = min(target_level2, len(level2_data))

        sampled_data = []
        sampled_data.extend(random.sample(level1_data, actual_level1))
        sampled_data.extend(random.sample(level2_data, actual_level2))

        # Shuffle final dataset for better training
        random.shuffle(sampled_data)

        logger.info(f"Sampled {len(sampled_data)} tasks "
                   f"(Level 1: {actual_level1}/{target_level1}, "
                   f"Level 2: {actual_level2}/{target_level2})")

        # Validate sampled data
        if len(sampled_data) == 0:
            raise ValueError("No training data available after sampling")

        return sampled_data

    def train(self):
        """
        Run optimized RL training with comprehensive monitoring.

        Features:
        - Memory-efficient training with gradient accumulation
        - Advanced learning rate scheduling
        - Comprehensive logging and checkpointing
        - Error handling and recovery
        """
        logger.info("Starting optimized RL training pipeline")

        # Load and validate dataset
        try:
            dataset = self.load_data()
            if len(dataset) == 0:
                raise ValueError("Empty dataset - no training data available")
        except Exception as e:
            logger.error(f"Dataset loading failed: {e}")
            raise

        # Enhanced training configuration
        training_config = self._create_optimized_config()

        # Initialize GRPO trainer with optimizations
        try:
            trainer = GRPOTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                reward_manager=self.reward_manager,
                config=training_config
            )
            logger.info("GRPO trainer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize trainer: {e}")
            raise

        # Training execution with monitoring
        try:
            logger.info("Starting RL training with hierarchical rewards...")
            logger.info(f"Training on {len(dataset)} samples for {self.config.num_epochs} epochs")

            trainer.train(dataset)

            logger.info("Training completed successfully")

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            # Ensure model is saved even if training fails
            self._save_model_safely()

    def _create_optimized_config(self) -> Dict[str, Any]:
        """Create optimized training configuration."""
        base_config = {
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

        # Add performance optimizations
        optimized_config = {
            **base_config,
            # Memory optimizations
            "gradient_checkpointing": True,
            "gradient_accumulation_steps": max(1, 8 // self.config.batch_size),  # Adaptive accumulation

            # Precision settings
            "bf16": self.use_amp,
            "fp16": False,  # Prefer bf16 over fp16

            # Learning optimizations
            "warmup_ratio": 0.1,
            "lr_scheduler_type": "cosine_with_min_lr",
            "min_lr_ratio": 0.1,

            # Stability improvements
            "max_grad_norm": 1.0,
            "adam_beta1": 0.9,
            "adam_beta2": 0.95,

            # Monitoring
            "logging_steps": 10,
            "save_steps": 100,
            "evaluation_strategy": "steps",
            "eval_steps": 50,

            # Data efficiency
            "dataloader_num_workers": 4,
            "dataloader_pin_memory": torch.cuda.is_available(),
        }

        return optimized_config

    def _save_model_safely(self):
        """Save model with error handling and validation."""
        try:
            output_dir = self.config.output_dir
            os.makedirs(output_dir, exist_ok=True)

            logger.info(f"Saving final model to {output_dir}")

            # Save model
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)

            # Validate save
            if not os.path.exists(os.path.join(output_dir, "config.json")):
                raise FileNotFoundError("Model config not saved properly")

            logger.info("Model saved successfully")

        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            # Attempt emergency save
            emergency_dir = f"{self.config.output_dir}_emergency"
            try:
                self.model.save_pretrained(emergency_dir)
                logger.info(f"Emergency save successful: {emergency_dir}")
            except:
                logger.critical("Emergency save also failed")

def main():
    config = RLConfig()
    trainer = TritonRLTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
# Performance improvements
