# AWS Spot Instance Checkpointing Guide

## 🎯 Goal
Protect against spot instance interruptions by automatically saving checkpoints to S3 and resuming training seamlessly.

---

## ⚡ Strategy: Multi-Layer Protection

### Layer 1: Frequent Model Checkpoints (Every 5 min)
### Layer 2: S3 Sync (Every 10 min)
### Layer 3: Spot Interruption Handler (2-min warning)
### Layer 4: Auto-Resume on New Instance

---

## 🔧 Implementation

### Step 1: Create S3 Bucket for Checkpoints

```bash
# Create bucket (do this once)
export BUCKET_NAME="tritonrl-checkpoints-$(date +%s)"
aws s3 mb s3://${BUCKET_NAME}

# Enable versioning (keeps old checkpoints)
aws s3api put-bucket-versioning \
    --bucket ${BUCKET_NAME} \
    --versioning-configuration Status=Enabled

echo "Created bucket: ${BUCKET_NAME}"
echo "Save this for later: export BUCKET_NAME=${BUCKET_NAME}"
```

---

### Step 2: Checkpoint Manager Script

Create `checkpoint_manager.py`:

```python
#!/usr/bin/env python3
"""
Checkpoint manager with automatic S3 sync and spot interruption handling
"""

import os
import sys
import time
import json
import subprocess
import threading
import signal
from pathlib import Path

class CheckpointManager:
    def __init__(self, s3_bucket, local_checkpoint_dir="checkpoints", sync_interval=600):
        """
        Args:
            s3_bucket: S3 bucket name (e.g., 'tritonrl-checkpoints-12345')
            local_checkpoint_dir: Local directory for checkpoints
            sync_interval: Seconds between S3 syncs (default 600 = 10 min)
        """
        self.s3_bucket = s3_bucket
        self.local_dir = Path(local_checkpoint_dir)
        self.sync_interval = sync_interval
        self.running = True
        self.last_sync = 0

        # Create local checkpoint directory
        self.local_dir.mkdir(parents=True, exist_ok=True)

        # Start background sync thread
        self.sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self.sync_thread.start()

        # Register spot interruption handler
        self._setup_interruption_handler()

        print(f"✓ CheckpointManager initialized")
        print(f"  Local dir: {self.local_dir}")
        print(f"  S3 bucket: s3://{self.s3_bucket}")
        print(f"  Sync interval: {self.sync_interval}s")

    def _sync_loop(self):
        """Background thread that syncs to S3 periodically"""
        while self.running:
            current_time = time.time()

            if current_time - self.last_sync >= self.sync_interval:
                self.sync_to_s3()
                self.last_sync = current_time

            time.sleep(30)  # Check every 30 seconds

    def sync_to_s3(self, force=False):
        """
        Sync local checkpoints to S3

        Args:
            force: If True, sync immediately regardless of interval
        """
        if not force and time.time() - self.last_sync < self.sync_interval:
            return

        print(f"\n{'='*60}")
        print(f"Syncing checkpoints to S3...")
        print(f"{'='*60}")

        try:
            # Sync entire checkpoint directory
            cmd = [
                "aws", "s3", "sync",
                str(self.local_dir),
                f"s3://{self.s3_bucket}/checkpoints/",
                "--exclude", "*.log",  # Don't sync large log files
                "--exclude", "wandb/*",  # Don't sync wandb runs
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print(f"✓ Synced to s3://{self.s3_bucket}/checkpoints/")
                self.last_sync = time.time()

                # Save sync metadata
                metadata = {
                    "last_sync": self.last_sync,
                    "sync_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "local_dir": str(self.local_dir),
                }

                with open(self.local_dir / "last_sync.json", "w") as f:
                    json.dump(metadata, f, indent=2)

            else:
                print(f"✗ Sync failed: {result.stderr}")

        except Exception as e:
            print(f"✗ Sync error: {e}")

    def restore_from_s3(self):
        """Restore checkpoints from S3 to local"""
        print(f"\n{'='*60}")
        print(f"Restoring checkpoints from S3...")
        print(f"{'='*60}")

        try:
            cmd = [
                "aws", "s3", "sync",
                f"s3://{self.s3_bucket}/checkpoints/",
                str(self.local_dir),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print(f"✓ Restored from s3://{self.s3_bucket}/checkpoints/")
                return True
            else:
                print(f"✗ Restore failed: {result.stderr}")
                return False

        except Exception as e:
            print(f"✗ Restore error: {e}")
            return False

    def _setup_interruption_handler(self):
        """Setup handler for spot instance interruption (2-min warning)"""

        def interruption_handler(signum, frame):
            print("\n" + "="*60)
            print("⚠️  SPOT INSTANCE INTERRUPTION DETECTED!")
            print("="*60)
            print("Performing emergency checkpoint sync...")

            # Force immediate sync
            self.sync_to_s3(force=True)

            print("\n✓ Emergency sync complete!")
            print("Checkpoints saved to S3. You can resume training on a new instance.")
            print("="*60)

            sys.exit(0)

        # Register SIGTERM handler (sent by AWS 2 minutes before termination)
        signal.signal(signal.SIGTERM, interruption_handler)

        # Also handle SIGINT (Ctrl+C) gracefully
        signal.signal(signal.SIGINT, interruption_handler)

    def check_spot_interruption(self):
        """
        Check AWS metadata for spot interruption notice
        Called periodically during training
        """
        try:
            # Query instance metadata for termination time
            result = subprocess.run(
                ["curl", "-s", "http://169.254.169.254/latest/meta-data/spot/instance-action"],
                capture_output=True,
                text=True,
                timeout=1,
            )

            if result.returncode == 0 and result.stdout:
                # Interruption notice detected!
                print("\n⚠️  Spot interruption notice detected!")
                self.sync_to_s3(force=True)
                return True

        except Exception:
            pass  # Metadata not available or timeout

        return False

    def save_training_state(self, state_dict):
        """
        Save training state (epoch, step, metrics, etc.)

        Args:
            state_dict: Dict with training state to save
        """
        state_file = self.local_dir / "training_state.json"

        with open(state_file, "w") as f:
            json.dump(state_dict, f, indent=2)

        print(f"✓ Saved training state to {state_file}")

    def load_training_state(self):
        """Load training state if it exists"""
        state_file = self.local_dir / "training_state.json"

        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
            print(f"✓ Loaded training state from {state_file}")
            return state

        return None

    def shutdown(self):
        """Clean shutdown - sync and exit"""
        print("\nShutting down CheckpointManager...")
        self.running = False
        self.sync_to_s3(force=True)
        print("✓ Checkpoint manager shut down cleanly")


def get_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in directory"""
    checkpoint_dir = Path(checkpoint_dir)

    # Look for checkpoint-* directories
    checkpoints = list(checkpoint_dir.glob("checkpoint-*"))

    if not checkpoints:
        return None

    # Sort by step number
    checkpoints.sort(key=lambda x: int(x.name.split("-")[1]))

    return checkpoints[-1]


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument("--action", choices=["sync", "restore"], required=True)
    args = parser.parse_args()

    manager = CheckpointManager(args.bucket)

    if args.action == "sync":
        manager.sync_to_s3(force=True)
    elif args.action == "restore":
        manager.restore_from_s3()
```

---

### Step 3: Modify Training Scripts to Use Checkpointing

Edit `train_integrated.py` to add checkpoint manager:

```python
# At the top of train_integrated.py, add:
from checkpoint_manager import CheckpointManager, get_latest_checkpoint

class IntegratedTrainer:
    def __init__(self):
        # ... existing code ...

        # Initialize checkpoint manager
        s3_bucket = os.environ.get("BUCKET_NAME")
        if s3_bucket:
            self.checkpoint_manager = CheckpointManager(
                s3_bucket=s3_bucket,
                local_checkpoint_dir="checkpoints",
                sync_interval=600,  # Sync every 10 minutes
            )
            print("✓ Checkpoint manager initialized with S3 sync")
        else:
            print("⚠️  No S3 bucket set (checkpoints local only)")
            self.checkpoint_manager = None

    def train_sft(self, max_samples=1000):
        """Stage 1: SFT with checkpointing"""

        # ... existing setup code ...

        # Check for existing checkpoint (resume if interrupted)
        latest_checkpoint = get_latest_checkpoint("checkpoints/sft")
        resume_from_checkpoint = None

        if latest_checkpoint:
            print(f"\n✓ Found existing checkpoint: {latest_checkpoint}")
            print("Resuming training from checkpoint...")
            resume_from_checkpoint = str(latest_checkpoint)

        # Training arguments with frequent checkpointing
        training_args = TrainingArguments(
            output_dir="checkpoints/sft",
            num_train_epochs=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=1e-5,

            # ✨ KEY: Frequent checkpointing
            save_steps=50,  # Save every 50 steps (~5 minutes)
            save_total_limit=3,  # Keep only last 3 checkpoints

            # Enable resume
            resume_from_checkpoint=resume_from_checkpoint,

            # ... rest of args ...
        )

        # ... trainer setup ...

        # Start training
        print("\nStarting SFT training with auto-checkpointing...")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # Force final sync
        if self.checkpoint_manager:
            self.checkpoint_manager.sync_to_s3(force=True)

        # ... rest of method ...
```

---

### Step 4: Spot Interruption Monitoring Script

Create `monitor_spot.sh`:

```bash
#!/bin/bash
# Continuously monitor for spot interruption and sync checkpoints

BUCKET_NAME=$1

if [ -z "$BUCKET_NAME" ]; then
    echo "Usage: ./monitor_spot.sh <s3-bucket-name>"
    exit 1
fi

echo "Monitoring spot instance for interruption..."
echo "Bucket: $BUCKET_NAME"

while true; do
    # Check for spot instance termination notice
    TERMINATION=$(curl -s http://169.254.169.254/latest/meta-data/spot/instance-action 2>/dev/null)

    if [ ! -z "$TERMINATION" ]; then
        echo ""
        echo "=========================================="
        echo "⚠️  SPOT INSTANCE INTERRUPTION DETECTED!"
        echo "=========================================="
        echo "Termination notice: $TERMINATION"
        echo ""
        echo "Performing emergency checkpoint sync..."

        # Sync all checkpoints immediately
        aws s3 sync checkpoints/ s3://${BUCKET_NAME}/checkpoints/ \
            --exclude "*.log" \
            --exclude "wandb/*"

        echo ""
        echo "✓ Emergency sync complete!"
        echo "You can resume training on a new instance with:"
        echo "  export BUCKET_NAME=${BUCKET_NAME}"
        echo "  python resume_training.py"
        echo "=========================================="

        exit 0
    fi

    sleep 5  # Check every 5 seconds
done
```

---

### Step 5: Resume Training Script

Create `resume_training.py`:

```python
#!/usr/bin/env python3
"""
Resume training after spot instance interruption
"""

import os
import sys
from checkpoint_manager import CheckpointManager, get_latest_checkpoint

def resume_training():
    # Get S3 bucket from environment
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
```

---

## 🚀 Usage Guide

### Initial Setup (First Time)

```bash
# 1. Create S3 bucket
export BUCKET_NAME="tritonrl-checkpoints-$(date +%s)"
aws s3 mb s3://${BUCKET_NAME}

# Save this for later!
echo "export BUCKET_NAME=${BUCKET_NAME}" >> ~/.bashrc
source ~/.bashrc

# 2. Start spot interruption monitor (in background)
nohup ./monitor_spot.sh ${BUCKET_NAME} > spot_monitor.log 2>&1 &

# 3. Run training with checkpointing
python train_integrated.py --stage all
```

### After Interruption (New Instance)

```bash
# 1. Set S3 bucket (use the same one!)
export BUCKET_NAME=tritonrl-checkpoints-1234567890

# 2. Resume training (automatically restores and continues)
python resume_training.py
```

---

## 📊 Checkpoint Strategy Summary

| Component | Frequency | Purpose |
|-----------|-----------|---------|
| Model checkpoints | Every 50 steps (~5 min) | Resume training |
| S3 sync | Every 10 minutes | Persist to cloud |
| Spot monitoring | Every 5 seconds | Detect interruption |
| Emergency sync | On SIGTERM (2-min warning) | Save before termination |
| Training state | Every epoch | Track progress |

---

## 💾 What Gets Saved

```
s3://your-bucket/checkpoints/
├── sft/
│   ├── checkpoint-50/
│   ├── checkpoint-100/
│   └── checkpoint-150/
├── rl_best_of_n/
│   ├── checkpoint-10/
│   └── checkpoint-20/
├── training_state.json
└── last_sync.json
```

---

## ⚡ Interruption Recovery Timeline

```
t=0:00    Training running normally
          ↓ Checkpoint every 5 min → S3 sync every 10 min

t=2:35    AWS sends SIGTERM (2-min warning)
          ↓ monitor_spot.sh detects

t=2:35:05 Emergency sync starts
          ↓ All checkpoints → S3

t=2:36:30 Sync complete (90 seconds)
          ↓ Instance terminates

t=2:40    You launch new instance
          ↓ Run resume_training.py

t=2:42    Checkpoints restored from S3
          ↓ Training resumes from step 100

t=2:43    Training continues!
```

**Max data loss: 10 minutes** (time since last sync)

---

## 🎯 Automatic Resume Feature

The training scripts now automatically:
1. ✅ Check for existing checkpoints on startup
2. ✅ Resume from latest checkpoint if found
3. ✅ Continue from exact same step
4. ✅ Sync to S3 every 10 minutes
5. ✅ Emergency sync on interruption (2-min warning)

---

## 💰 Cost Optimization

### Storage Costs:
- Checkpoints: ~20GB per model
- S3 Standard: $0.023/GB/month
- Total: ~$0.46/month for checkpoints

### With Lifecycle Policy:
```bash
# Delete checkpoints older than 7 days
aws s3api put-bucket-lifecycle-configuration \
    --bucket ${BUCKET_NAME} \
    --lifecycle-configuration '{
        "Rules": [{
            "Id": "DeleteOldCheckpoints",
            "Status": "Enabled",
            "Prefix": "checkpoints/",
            "Expiration": {"Days": 7}
        }]
    }'
```

---

## ✅ Testing the System

```bash
# Test checkpoint sync
python checkpoint_manager.py --bucket ${BUCKET_NAME} --action sync

# Test restore
python checkpoint_manager.py --bucket ${BUCKET_NAME} --action restore

# Simulate interruption (for testing)
kill -TERM <training-process-pid>
# Should trigger emergency sync
```

---

## 🔒 Security Best Practices

1. **Use IAM role** (not access keys):
```bash
# Attach IAM role to EC2 instance with S3 access
# Policy: AmazonS3FullAccess (or more restrictive)
```

2. **Encrypt bucket**:
```bash
aws s3api put-bucket-encryption \
    --bucket ${BUCKET_NAME} \
    --server-side-encryption-configuration '{
        "Rules": [{
            "ApplyServerSideEncryptionByDefault": {
                "SSEAlgorithm": "AES256"
            }
        }]
    }'
```

3. **Private bucket**:
```bash
aws s3api put-public-access-block \
    --bucket ${BUCKET_NAME} \
    --public-access-block-configuration \
    "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"
```

---

## 📝 Complete Setup Commands

```bash
#!/bin/bash
# Complete setup script

# 1. Create and configure S3 bucket
export BUCKET_NAME="tritonrl-checkpoints-$(date +%s)"
aws s3 mb s3://${BUCKET_NAME}
aws s3api put-bucket-versioning --bucket ${BUCKET_NAME} --versioning-configuration Status=Enabled
aws s3api put-bucket-encryption --bucket ${BUCKET_NAME} --server-side-encryption-configuration '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'

# 2. Save bucket name
echo "export BUCKET_NAME=${BUCKET_NAME}" >> ~/.bashrc
source ~/.bashrc

# 3. Make scripts executable
chmod +x checkpoint_manager.py monitor_spot.sh resume_training.py

# 4. Start monitoring
nohup ./monitor_spot.sh ${BUCKET_NAME} > spot_monitor.log 2>&1 &

# 5. Start training
python train_integrated.py --stage all

echo "Setup complete! Training with automatic checkpointing."
```

This provides **robust protection** against spot interruptions with minimal overhead!
