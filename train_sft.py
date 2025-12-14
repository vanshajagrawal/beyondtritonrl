"""
Supervised Fine-Tuning Trainer for TritonRL Models

This module provides functionality for fine-tuning language models on instruction-response pairs
using supervised learning techniques. It handles data loading, tokenization, and training
orchestration with proper error handling and logging.
"""

import logging
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from config import SFTConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SFTTrainer:
    """
    Supervised Fine-Tuning Trainer for language models.

    This class handles the complete SFT pipeline including model loading,
    data preprocessing, training, and model saving.

    Args:
        config: SFTConfig object containing training hyperparameters
    """

    def __init__(self, config: SFTConfig):
        """
        Initialize the SFT trainer with configuration.

        Args:
            config: Training configuration object

        Raises:
            ValueError: If config is invalid
            RuntimeError: If model loading fails
        """
        if not isinstance(config, SFTConfig):
            raise ValueError("config must be an instance of SFTConfig")

        self.config = config
        self.model = None
        self.tokenizer = None

        try:
            logger.info(f"Loading base model: {config.base_model}")
            self.model = AutoModelForCausalLM.from_pretrained(
                config.base_model,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,  # Optimize memory usage
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.base_model,
                trust_remote_code=True,
            )

            # Handle missing pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("Set pad_token to eos_token")

        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise RuntimeError(f"Tokenizer loading failed: {e}")

    def load_data(self):
        """
        Load and preprocess SFT training data.

        Returns:
            Dataset: Tokenized dataset ready for training

        Raises:
            FileNotFoundError: If training data file doesn't exist
            ValueError: If data format is invalid
        """
        data_path = "data/processed/sft_train.jsonl"
        logger.info(f"Loading SFT training data from {data_path}")

        try:
            dataset = load_dataset("json", data_files=data_path, split="train")
            logger.info(f"Loaded {len(dataset)} training examples")
        except FileNotFoundError:
            logger.error(f"Training data file not found: {data_path}")
            raise FileNotFoundError(f"SFT training data not found at {data_path}")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise RuntimeError(f"Dataset loading failed: {e}")

        # Validate data format
        required_columns = ["instruction", "output"]
        if not all(col in dataset.column_names for col in required_columns):
            raise ValueError(f"Dataset missing required columns: {required_columns}")

        def format_prompt(example):
            """Format instruction-response pairs into training text."""
            try:
                instruction = example.get("instruction", "").strip()
                output = example.get("output", "").strip()

                if not instruction or not output:
                    logger.warning("Empty instruction or output found")
                    return {"text": ""}

                text = f"{instruction}\n\n{output}"
                return {"text": text}
            except Exception as e:
                logger.error(f"Error formatting prompt: {e}")
                return {"text": ""}

        logger.info("Formatting prompts...")
        dataset = dataset.map(format_prompt, remove_columns=dataset.column_names)

        # Filter out empty examples
        dataset = dataset.filter(lambda x: len(x["text"]) > 0)
        logger.info(f"After filtering: {len(dataset)} valid examples")

        def tokenize_function(examples):
            """Tokenize text examples with proper error handling."""
            try:
                return self.tokenizer(
                    examples["text"],
                    truncation=True,
                    max_length=self.config.max_seq_length,
                    padding="max_length",
                )
            except Exception as e:
                logger.error(f"Tokenization error: {e}")
                # Return empty tokens as fallback
                return {
                    "input_ids": [[self.tokenizer.pad_token_id] * self.config.max_seq_length],
                    "attention_mask": [[0] * self.config.max_seq_length]
                }

        logger.info("Tokenizing dataset...")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
        )

        logger.info(f"Tokenization complete. Dataset size: {len(tokenized_dataset)}")
        return tokenized_dataset

    def train(self):
        """Run SFT training"""
        dataset = self.load_data()

        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            save_total_limit=3,
            bf16=True,
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            report_to="wandb",
            remove_unused_columns=False,
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )

        print("Starting SFT training...")
        trainer.train()

        print(f"Saving final model to {self.config.output_dir}")
        trainer.save_model(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)

def main():
    config = SFTConfig()
    trainer = SFTTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
# Minor code cleanup
