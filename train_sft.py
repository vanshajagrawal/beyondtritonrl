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

class SFTTrainer:
    def __init__(self, config: SFTConfig):
        self.config = config

        print(f"Loading base model: {config.base_model}")
        self.model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.base_model,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_data(self):
        """Load preprocessed SFT training data"""
        print("Loading SFT training data...")
        dataset = load_dataset("json", data_files="data/processed/sft_train.jsonl", split="train")

        def format_prompt(example):
            text = f"{example['instruction']}\n\n{example['output']}"
            return {"text": text}

        dataset = dataset.map(format_prompt, remove_columns=dataset.column_names)

        def tokenize(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.config.max_seq_length,
                padding="max_length",
            )

        tokenized_dataset = dataset.map(
            tokenize,
            batched=True,
            remove_columns=["text"],
        )

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
