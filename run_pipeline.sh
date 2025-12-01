#!/bin/bash
# Complete 10-hour pipeline automation script

set -e  # Exit on error

echo "========================================="
echo "TRITONRL + EXTENSIONS: 10-HOUR PIPELINE"
echo "========================================="
echo ""

# Check CUDA
echo "[0/5] Checking CUDA availability..."
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'✓ CUDA available: {torch.cuda.get_device_name(0)}')"
echo ""

# Test extensions
echo "[1/5] Testing all 4 extensions..."
python test_extensions_quick.py
if [ $? -ne 0 ]; then
    echo "✗ Extension tests failed! Fix before continuing."
    exit 1
fi
echo ""

# Prepare data
echo "[2/5] Preparing data..."
python prepare_data_simple.py
if [ $? -ne 0 ]; then
    echo "✗ Data preparation failed!"
    exit 1
fi
echo ""

# SFT training
echo "[3/5] SFT training (1.5 hours)..."
python train_integrated.py --stage sft --max_samples 1000
if [ $? -ne 0 ]; then
    echo "✗ SFT training failed!"
    exit 1
fi
echo ""

# RL training
echo "[4/5] Best-of-N RL training (4 hours)..."
python train_integrated.py --stage rl
if [ $? -ne 0 ]; then
    echo "✗ RL training failed!"
    exit 1
fi
echo ""

# Evaluation
echo "[5/5] Evaluation and comparison..."
python evaluate_simple.py --compare
if [ $? -ne 0 ]; then
    echo "✗ Evaluation failed!"
    exit 1
fi
echo ""

echo "========================================="
echo "PIPELINE COMPLETE!"
echo "========================================="
echo ""
echo "Results saved to:"
echo "  - checkpoints/sft_final/"
echo "  - checkpoints/rl_final/"
echo "  - outputs/eval_*.json"
echo ""
echo "Next: Review outputs/ for detailed metrics"
