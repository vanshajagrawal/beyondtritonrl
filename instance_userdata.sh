#!/bin/bash
# Auto-setup script for AWS instance
set -e

echo "=========================================="
echo "INSTANCE SETUP STARTING"
echo "=========================================="

# Update system
apt-get update -y

# Install dependencies
apt-get install -y git python3-pip tmux htop

# Install Python packages
pip3 install --upgrade pip
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip3 install triton transformers datasets accelerate peft tqdm bitsandbytes

# Clone repo
cd /home/ubuntu
git clone https://github.com/vanshajagrawal/tritonrl.git
cd tritonrl

# Setup S3 checkpointing
export BUCKET_NAME="tritonrl-checkpoints-$(date +%s)"
bash setup_s3_checkpointing.sh

# Start cost monitor in background
nohup bash monitor_costs.sh > cost_monitor.log 2>&1 &

# Start spot interruption monitor
nohup bash monitor_spot.sh $BUCKET_NAME > spot_monitor.log 2>&1 &

# Start training in tmux session
tmux new-session -d -s training 'bash run_budget_pipeline.sh 2>&1 | tee training.log'

echo "=========================================="
echo "✓ SETUP COMPLETE"
echo "=========================================="
echo ""
echo "Training started in tmux session 'training'"
echo "To attach: tmux attach -t training"
echo "Logs: tail -f training.log"
echo "Cost monitor: tail -f cost_monitor.log"
echo "=========================================="
