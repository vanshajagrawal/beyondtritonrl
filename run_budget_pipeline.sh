#!/bin/bash
# Budget-optimized pipeline for $100 budget
# Reduces dataset size to fit in 3 hours on 8×A100

set -e

echo "=========================================="
echo "BUDGET-OPTIMIZED TRAINING PIPELINE"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  - 200 training samples (vs 1,000 full)"
echo "  - 30 RL fusion tasks (vs 100 full)"
echo "  - 5 samples per task (vs 10 full)"
echo "  - Expected time: ~2.5 hours"
echo "  - Expected cost: ~\$66 on 8×A100 spot"
echo ""
echo "=========================================="
echo ""

# 1. Test extensions (5 min)
echo "Step 1/5: Testing extensions..."
python test_extensions_quick.py
if [ $? -ne 0 ]; then
    echo "✗ Extension test failed!"
    exit 1
fi
echo "✓ Extensions validated"
echo ""

# 2. Prepare data - REDUCED (15 min)
echo "Step 2/5: Preparing data (200 samples)..."
python prepare_data_simple.py --max_samples 200
if [ $? -ne 0 ]; then
    echo "✗ Data preparation failed!"
    exit 1
fi
echo "✓ Data prepared"
echo ""

# 3. SFT training - REDUCED (30 min)
echo "Step 3/5: Running SFT training (200 samples, limited steps)..."
python train_integrated.py \
    --stage sft \
    --max_samples 200 \
    --num_train_epochs 1 \
    --max_steps 100 \
    --save_steps 50
if [ $? -ne 0 ]; then
    echo "✗ SFT training failed!"
    exit 1
fi
echo "✓ SFT training complete"
echo ""

# 4. RL training - REDUCED (1.5 hrs)
echo "Step 4/5: Running Best-of-N RL (30 tasks, 5 samples each)..."
python train_integrated.py \
    --stage rl \
    --num_fusion_tasks 30 \
    --n_samples 5 \
    --top_k 2
if [ $? -ne 0 ]; then
    echo "✗ RL training failed!"
    exit 1
fi
echo "✓ RL training complete"
echo ""

# 5. Evaluation (20 min)
echo "Step 5/5: Evaluating models..."
python evaluate_simple.py --compare --num_tasks 15
if [ $? -ne 0 ]; then
    echo "✗ Evaluation failed!"
    exit 1
fi
echo "✓ Evaluation complete"
echo ""

echo "=========================================="
echo "✓ PIPELINE COMPLETE"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - checkpoints/sft_final/"
echo "  - checkpoints/rl_final/"
echo "  - outputs/eval_*.json"
echo ""
echo "Check results with:"
echo "  cat outputs/eval_comparison.json"
echo "=========================================="
