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
        """Sync local checkpoints to S3"""
        if not force and time.time() - self.last_sync < self.sync_interval:
            return

        print(f"\n{'='*60}")
        print(f"Syncing checkpoints to S3...")
        print(f"{'='*60}")

        try:
            cmd = [
                "aws", "s3", "sync",
                str(self.local_dir),
                f"s3://{self.s3_bucket}/checkpoints/",
                "--exclude", "*.log",
                "--exclude", "wandb/*",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print(f"✓ Synced to s3://{self.s3_bucket}/checkpoints/")
                self.last_sync = time.time()

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
        """Setup handler for spot instance interruption"""

        def interruption_handler(signum, frame):
            print("\n" + "="*60)
            print("⚠️  SPOT INSTANCE INTERRUPTION DETECTED!")
            print("="*60)
            print("Performing emergency checkpoint sync...")

            self.sync_to_s3(force=True)

            print("\n✓ Emergency sync complete!")
            print("="*60)
            sys.exit(0)

        signal.signal(signal.SIGTERM, interruption_handler)
        signal.signal(signal.SIGINT, interruption_handler)

    def save_training_state(self, state_dict):
        """Save training state"""
        state_file = self.local_dir / "training_state.json"

        with open(state_file, "w") as f:
            json.dump(state_dict, f, indent=2)

        print(f"✓ Saved training state")

    def load_training_state(self):
        """Load training state if it exists"""
        state_file = self.local_dir / "training_state.json"

        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
            print(f"✓ Loaded training state")
            return state

        return None

    def shutdown(self):
        """Clean shutdown"""
        print("\nShutting down CheckpointManager...")
        self.running = False
        self.sync_to_s3(force=True)
        print("✓ Checkpoint manager shut down")


def get_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in directory"""
    checkpoint_dir = Path(checkpoint_dir)

    checkpoints = list(checkpoint_dir.glob("checkpoint-*"))

    if not checkpoints:
        return None

    checkpoints.sort(key=lambda x: int(x.name.split("-")[1]))

    return checkpoints[-1]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--action", choices=["sync", "restore"], required=True)
    args = parser.parse_args()

    manager = CheckpointManager(args.bucket)

    if args.action == "sync":
        manager.sync_to_s3(force=True)
    elif args.action == "restore":
        manager.restore_from_s3()
