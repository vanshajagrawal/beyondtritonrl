#!/usr/bin/env python3
"""Resume training after spot instance interruption"""

import os
import sys
from checkpoint_manager import CheckpointManager, get_latest_checkpoint

def resume_training():
    s3_bucket = os.environ.get("BUCKET_NAME")

    if not s3_bucket:
        print("Error: BUCKET_NAME not set!")
        print("Set it with: export BUCKET_NAME=your-bucket-name")
        sys.exit(1)

    print("="*80)
    print("RESUMING TRAINING AFTER INTERRUPTION")
    print("="*80)

    # Initialize checkpoint manager
    manager = CheckpointManager(s3_bucket)

    # Restore checkpoints from S3
    print("\nRestoring checkpoints from S3...")
    if not manager.restore_from_s3():
        print("Error: Failed to restore checkpoints!")
        sys.exit(1)

    # Check what stage we were in
    sft_checkpoint = get_latest_checkpoint("checkpoints/sft")
    rl_checkpoint = get_latest_checkpoint("checkpoints/rl_best_of_n")

    # Load training state
    state = manager.load_training_state()

    if state:
        print(f"\nLast training state:")
        print(f"  Stage: {state.get('stage', 'unknown')}")
        print(f"  Epoch: {state.get('epoch', 'unknown')}")
        print(f"  Step: {state.get('step', 'unknown')}")

    # Determine what to resume
    if rl_checkpoint:
        print("\n✓ Found RL checkpoint - resuming RL training...")
        stage = "rl"
    elif sft_checkpoint:
        print("\n✓ Found SFT checkpoint - resuming SFT training...")
        stage = "sft"
    else:
        print("\n⚠️  No checkpoints found - starting from scratch...")
        stage = "all"

    # Resume training
    print(f"\nResuming with: python train_integrated.py --stage {stage}")
    os.system(f"python train_integrated.py --stage {stage}")

if __name__ == "__main__":
    resume_training()
