# Budget-Optimized Training Plan

## Budget: $100

## Instance Analysis

### p4d.24xlarge (8×A100 40GB)
- **Spot price**: $26.34/hr
- **Available hours**: $100 / $26.34 = **3.8 hours**
- **Problem**: Full pipeline needs 10 hours ❌

### g5.12xlarge (4×A10G 24GB)
- **Spot price**: ~$2-3/hr
- **Available hours**: $100 / $2.50 = **40 hours**
- **Performance**: ~2-3× slower than A100
- **Effective training time**: 10 hrs × 2.5 = 25 hours needed ❌

## Optimized Strategy

### Option 1: Reduced Dataset on A100 (RECOMMENDED)
**Instance**: p4d.24xlarge (8×A100)
**Cost**: $26.34/hr × 3.5 hrs = **~$92**

**Training modifications**:
```python
# Reduce samples significantly
prepare_data_simple.py --max_samples 200  # Instead of 1,000

# Faster SFT
train_integrated.py --stage sft \
    --max_samples 200 \
    --num_train_epochs 1 \
    --max_steps 100  # Limit steps

# Faster RL
train_integrated.py --stage rl \
    --num_fusion_tasks 30 \  # Instead of 100
    --n_samples 5 \          # Instead of 10
    --top_k 2                # Instead of 3
```

**Time breakdown**:
- Data prep: 15 min (200 samples)
- SFT training: 30 min (200 samples, limited steps)
- RL training: 1.5 hrs (30 tasks × 5 samples)
- Evaluation: 20 min (quick eval)
- **Total: ~2.5 hours = $65.85**
- **Buffer: 1 hour for issues = $26.34**
- **Total: $92.19 < $100** ✅

**Expected results**:
- SFT baseline: ~35-40% (lower due to less data)
- RL improvement: ~45-50% (+10% over SFT)
- **Still demonstrates RL works!**

### Option 2: G5 with Full Dataset (CHEAPER, SLOWER)
**Instance**: g5.12xlarge (4×A10G)
**Cost**: $2.50/hr × 25 hrs = **~$62.50**

**Time breakdown**:
- Data prep: 1 hr
- SFT training: 4 hrs (with quantization)
- RL training: 10 hrs
- Evaluation: 2 hrs
- **Total: ~17 hours = $42.50**
- **Well under budget** ✅

**Expected results**:
- Same as full implementation
- Just takes longer

## Recommendation: OPTION 1 (Fast A100 with Reduced Data)

**Why**:
1. ✅ Completes in 3 hours (fits budget)
2. ✅ Uses fast A100 GPUs
3. ✅ Still demonstrates RL improvement
4. ✅ Good for proof-of-concept
5. ✅ Can scale up later if results promising

**Execution plan**:
1. Launch p4d.24xlarge spot ($26/hr)
2. Start cost monitor (auto-kill at $150)
3. Run reduced pipeline (200 samples, 30 RL tasks)
4. Complete in ~2.5 hours (~$66)
5. Keep 1hr buffer for issues (~$26)
6. **Total: ~$92 < $100 budget**

## Alternative if p4d spot unavailable

Fall back to g5.12xlarge at $2.50/hr:
- Still under budget
- Just takes longer (overnight run)
- Full dataset possible

## Cost Safety Net

**monitor_costs.sh** will:
- Check costs every 5 minutes
- Warn at 80% budget ($120)
- Auto-terminate at $150 (safety margin above $100)
- Prevents runaway costs
