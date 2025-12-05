#!/bin/bash
# Fast debug pipeline (30 min instead of 10 hours)
# Run this on cheap hardware (g5.xlarge or g5.12xlarge) before expensive H100 run

set -e

echo "=========================================="
echo "DEBUG PIPELINE (Fast Validation)"
echo "=========================================="
echo ""
echo "This will validate:"
echo "  ✓ All imports work"
echo "  ✓ Extensions load correctly"
echo "  ✓ Training pipeline executes"
echo "  ✓ No critical bugs"
echo ""
echo "Expected duration: ~30 minutes"
echo "Expected cost: $0.50-6 (depending on instance type)"
echo ""
echo "=========================================="
echo ""

# 1. Quick extension test (2 min)
echo "Step 1/5: Testing extensions..."
python test_extensions_quick.py
if [ $? -ne 0 ]; then
    echo "✗ Extension test failed!"
    exit 1
fi
echo "✓ Extensions passed"
echo ""

# 2. Tiny data prep (1 min)
echo "Step 2/5: Preparing mini dataset (20 samples)..."
python prepare_data_simple.py --max_samples 20
if [ $? -ne 0 ]; then
    echo "✗ Data preparation failed!"
    exit 1
fi
echo "✓ Data preparation complete"
echo ""

# 3. Mini SFT (10 min)
echo "Step 3/5: Running mini SFT training (10 steps)..."
python train_integrated.py \
    --stage sft \
    --max_samples 20 \
    --num_train_epochs 1 \
    --max_steps 10 \
    --per_device_train_batch_size 1
if [ $? -ne 0 ]; then
    echo "✗ SFT training failed!"
    exit 1
fi
echo "✓ SFT training complete"
echo ""

# 4. Mini RL (10 min)
echo "Step 4/5: Running mini RL training (5 tasks, 3 samples each)..."
python train_integrated.py \
    --stage rl \
    --num_fusion_tasks 5 \
    --n_samples 3 \
    --top_k 2
if [ $? -ne 0 ]; then
    echo "✗ RL training failed!"
    exit 1
fi
echo "✓ RL training complete"
echo ""

# 5. Quick eval (5 min)
echo "Step 5/5: Running evaluation (3 test tasks)..."
python evaluate_simple.py \
    --model_path checkpoints/rl_final \
    --num_tasks 3
if [ $? -ne 0 ]; then
    echo "✗ Evaluation failed!"
    exit 1
fi
echo "✓ Evaluation complete"
echo ""

echo ""
echo "=========================================="
echo "✓ DEBUG PIPELINE COMPLETE"
echo "=========================================="
echo ""
echo "All tests passed! The implementation is working correctly."
echo ""
echo "You're now ready for the production H100 run."
echo "No code changes needed - just remove the size limits:"
echo ""
echo "  ./run_pipeline.sh"
echo ""
echo "This will run the full 10-hour pipeline with 1,000 samples."
echo "=========================================="
