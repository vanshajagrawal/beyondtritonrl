#!/usr/bin/env python3
"""
Training script that demonstrates how to use extensions.

Usage:
    # Train with baseline (no extensions)
    python train_with_extensions.py --preset baseline

    # Train with only multi-input testing (easiest)
    python train_with_extensions.py --preset multi_input_only

    # Train with recommended easy combo
    python train_with_extensions.py --preset easy_combo

    # Train with all extensions
    python train_with_extensions.py --preset all_extensions
"""

import argparse
from config_examples import get_config
from extensions import FusionDataGenerator, HardenedVerifier, AdaptiveCurriculum, StagedEvaluator

def train_with_extensions(preset: str = 'baseline', stage: str = 'all'):
    """
    Run training with specified extension preset

    Args:
        preset: Config preset name (see config_examples.py)
        stage: 'data', 'sft', 'rl', or 'all'
    """
    configs = get_config(preset)
    ext_config = configs['extensions']

    print(f"\n{'='*80}")
    print(f"Training with preset: {preset}")
    print(f"Enabled extensions: {ext_config.get_enabled_extensions()}")
    print(f"{'='*80}\n")

    # ========================================================================
    # STAGE 1: Data Preparation (with optional fusion data)
    # ========================================================================
    if stage in ['data', 'all']:
        print("\n[STAGE 1: DATA PREPARATION]")

        from data.prepare_data import DataPreparer

        preparer = DataPreparer(configs['data'])
        pytorch_codes = preparer.load_kernelbook()

        # Generate fusion tasks if enabled
        if ext_config.enable_fusion_data:
            print("\n→ Extension: Generating fusion-centric tasks...")
            fusion_gen = FusionDataGenerator(ext_config)
            fusion_tasks = fusion_gen.generate_fusion_tasks()

            # Merge with existing data
            print(f"  Added {len(fusion_tasks)} fusion tasks")
            # Would merge here in actual implementation

        # Generate SFT data
        sft_data = preparer.generate_triton_solutions(pytorch_codes)
        labeled_data = preparer.label_difficulty(pytorch_codes)
        preparer.save_datasets(sft_data, labeled_data)

    # ========================================================================
    # STAGE 2: Supervised Fine-Tuning (same as baseline)
    # ========================================================================
    if stage in ['sft', 'all']:
        print("\n[STAGE 2: SUPERVISED FINE-TUNING]")
        from train_sft import SFTTrainer

        trainer = SFTTrainer(configs['sft'])
        trainer.train()

    # ========================================================================
    # STAGE 3: Reinforcement Learning (with optional extensions)
    # ========================================================================
    if stage in ['rl', 'all']:
        print("\n[STAGE 3: REINFORCEMENT LEARNING]")

        # Initialize verifier (base or hardened)
        if any([
            ext_config.enable_multi_input,
            ext_config.enable_strict_sandbox,
            ext_config.enable_calibrated_timing,
        ]):
            print("\n→ Extension: Using HardenedVerifier")
            verifier = HardenedVerifier(ext_config)
        else:
            from verifiers import TritonVerifier
            verifier = TritonVerifier()

        # Initialize staged evaluator if enabled
        if ext_config.enable_staged_eval:
            print("\n→ Extension: Using StagedEvaluator")
            evaluator = StagedEvaluator(verifier, ext_config)
        else:
            evaluator = None

        # Initialize curriculum if enabled
        if ext_config.enable_adaptive_curriculum:
            print("\n→ Extension: Using AdaptiveCurriculum")
            curriculum = AdaptiveCurriculum(ext_config)
        else:
            curriculum = None

        # Run RL training (would integrate extensions here)
        print("\nStarting RL training with extensions...")
        print("  → Verifier:", "Hardened" if isinstance(verifier, HardenedVerifier) else "Base")
        print("  → Evaluator:", "Staged" if evaluator else "Single-stage")
        print("  → Curriculum:", "Adaptive" if curriculum else "Static")

        # Actual training would happen here
        # from train_rl import TritonRLTrainer
        # trainer = TritonRLTrainer(configs['rl'], verifier, evaluator, curriculum)
        # trainer.train()

        print("\nRL training would run here with configured extensions")

    print(f"\n{'='*80}")
    print("Training pipeline completed!")
    print(f"{'='*80}\n")

def main():
    parser = argparse.ArgumentParser(description="Train TritonRL with modular extensions")
    parser.add_argument(
        '--preset',
        type=str,
        default='baseline',
        choices=[
            'baseline',
            'multi_input_only',
            'fusion_data_only',
            'curriculum_only',
            'calibrated_timing_only',
            'staged_eval_only',
            'sandbox_only',
            'easy_combo',
            'all_extensions',
        ],
        help='Configuration preset'
    )
    parser.add_argument(
        '--stage',
        type=str,
        default='all',
        choices=['data', 'sft', 'rl', 'all'],
        help='Which training stage to run'
    )

    args = parser.parse_args()

    train_with_extensions(args.preset, args.stage)

if __name__ == "__main__":
    main()
