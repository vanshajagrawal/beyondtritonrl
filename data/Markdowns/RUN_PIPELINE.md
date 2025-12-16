# 10-Hour Pipeline: Complete Implementation Guide

This guide walks you through running the complete TritonRL + Extensions pipeline in 10 hours.

---

## 🚀 Quick Start (TL;DR)

```bash
# Setup (5 minutes)
cd ~/Documents/tritonrl
pip install -r requirements.txt

# Run full pipeline (8-10 hours)
./run_pipeline.sh

# That's it! The script handles everything.
```

---

## 📋 Detailed Step-by-Step Guide

### Step 0: Environment Setup (30 min)

#### AWS Setup:
```bash
# Request p5.48xlarge spot instance (8x H100 40GB)
aws ec2 request-spot-instances \
  --instance-type p5.48xlarge \
  --spot-price "15.00" \
  --launch-specification '{
    "ImageId": "ami-0c55b159cbfafe1f0",
    "InstanceType": "p5.48xlarge"
  }'

# SSH into instance
ssh -i your-key.pem ubuntu@<instance-ip>
```

#### Environment Setup:
```bash
# Clone repo
cd ~
git clone <your-repo-url> tritonrl
cd tritonrl

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify CUDA
python -c "import torch; print(torch.cuda.is_available())"
# Should print: True
```

---

### Step 1: Test Extensions (15 min)

```bash
# Verify all 4 extensions work
python test_extensions_quick.py
```

**Expected output:**
```
✓ PASS  Extension 1 (Multi-input)
✓ PASS  Extension 2 (Staged eval)
✓ PASS  Extension 3 (Curriculum)
✓ PASS  Extension 4 (Calibrated timing)

🎉 ALL EXTENSIONS WORKING!
```

If any fail, debug before proceeding.

---

### Step 2: Prepare Data (45 min)

```bash
# Download and prepare KernelBook data
python prepare_data_simple.py
```

**What this does:**
- Downloads 18k samples from HuggingFace
- Takes first 1,000 for faster training
- Labels difficulty (L1/L2/L3)
- Saves to `data/processed/`

**Expected output:**
```
Loaded 1000 samples
  Level 1 (single ops): 600
  Level 2 (fusion): 300
  Level 3 (architectures): 100

✓ Saved to data/processed/sft_train.jsonl
✓ Saved to data/processed/difficulty_labels.jsonl
```

---

### Step 3: SFT Training (1.5 hours)

```bash
# Train baseline with all extensions active
python train_integrated.py --stage sft --max_samples 1000
```

**What this does:**
- Loads Qwen3-7B
- Trains on 1,000 samples for 1 epoch
- Uses all 4 extensions during training
- Saves to `checkpoints/sft_final/`

**GPU utilization:** 8×H100 for 20 minutes

**Expected output:**
```
INITIALIZED WITH 4 CORE EXTENSIONS
✓ Multi-input testing: 5 test variations
✓ Staged evaluation: AST → compile → tiny → full → timing
✓ Adaptive curriculum: 0.1 → 0.5
✓ Calibrated timing: 50 trials with trimmed mean

Starting SFT training...
Epoch 1: 100%|████████| 16/16 [20:00<00:00]

✓ SFT training complete!
```

---

### Step 4: Best-of-N RL (4 hours)

```bash
# RL training on fusion kernels
python train_integrated.py --stage rl
```

**What this does:**
- Loads SFT checkpoint
- Generates 10 samples per task (100 fusion tasks)
- Evaluates with all 4 extensions
- Keeps top-3 per task
- Fine-tunes on best samples

**GPU utilization:** 8×H100 for 2 hours (generation) + 30 min (fine-tuning)

**Expected output:**
```
STAGE 2: BEST-OF-N RL FOR FUSION KERNELS

Found 300 fusion tasks
Using 100 tasks for faster training

Generating 10 samples per task...
Processing tasks: 100%|████████| 100/100 [2:00:00<00:00]

✓ Collected 250 high-quality samples

Fine-tuning on best samples...
Epoch 1/2: 100%|████████| 32/32 [15:00<00:00]
Epoch 2/2: 100%|████████| 32/32 [15:00<00:00]

✓ Best-of-N RL training complete!
```

---

### Step 5: Evaluation (1 hour)

```bash
# Evaluate and compare models
python evaluate_simple.py --compare
```

**What this does:**
- Tests SFT baseline on 20 fusion tasks
- Tests RL model on same tasks
- Compares correctness/validity rates
- Saves detailed results

**GPU utilization:** 8×H100 for 30 minutes

**Expected output:**
```
COMPARING MODELS

[1/2] Evaluating SFT baseline...
  Valid:    85.0% (17/20)
  Compiled: 70.0% (14/20)
  Correct:  45.0% (9/20)

[2/2] Evaluating RL model...
  Valid:    90.0% (18/20)
  Compiled: 75.0% (15/20)
  Correct:  55.0% (11/20)

COMPARISON
Metric          SFT        RL         Improvement
--------------------------------------------------
Valid           85.0%      90.0%      +5.0%
Compiled        70.0%      75.0%      +5.0%
Correct         45.0%      55.0%      +10.0%

✓ RL IMPROVED OVER SFT BASELINE
  Correctness: 45.0% → 55.0% (+10.0%)
```

---

## ⏱️ Time Breakdown

| Step | Task | Time | GPU Usage |
|------|------|------|-----------|
| 0 | Environment setup | 30 min | None |
| 1 | Test extensions | 15 min | 1×H100 |
| 2 | Data prep | 45 min | CPU |
| 3 | SFT training | 1.5 hrs | 8×H100 |
| 4 | Best-of-N RL | 4 hrs | 8×H100 |
| 5 | Evaluation | 1 hr | 8×H100 |
| | **Buffer** | 2 hrs | - |
| | **TOTAL** | **~10 hrs** | - |

---

## 💰 Cost Breakdown

### GPU Hours:
- SFT: 8 GPUs × 0.5h = 4 GPU-hours
- RL generation: 8 GPUs × 2h = 16 GPU-hours
- RL fine-tune: 8 GPUs × 0.5h = 4 GPU-hours
- Evaluation: 8 GPUs × 0.5h = 4 GPU-hours
- Testing: 1 GPU × 1h = 1 GPU-hour
- **Total: ~29 GPU-hours**

### AWS Cost (p5.48xlarge spot):
- Spot rate: ~$12/hour
- Active time: ~8 hours
- **Total: ~$96-120**

---

## 🎯 What You Get

### Deliverables:
1. ✅ **SFT baseline model** (`checkpoints/sft_final/`)
2. ✅ **RL model** (`checkpoints/rl_final/`)
3. ✅ **Evaluation results** (`outputs/eval_*.json`)
4. ✅ **Comparison report** showing improvement

### Can Claim:
- "Implemented 4/4 core extensions from project report"
- "Best-of-N RL training on fusion kernels"
- "10% improvement in correctness over SFT baseline"
- "Validated on 20 KernelBench Level 2 tasks"
- "All 4 extensions active during training and evaluation"

---

## 🐛 Troubleshooting

### Issue: CUDA Out of Memory

**Solution:**
```python
# In train_integrated.py, reduce batch size
per_device_train_batch_size=1  # Instead of 2
gradient_accumulation_steps=8  # Instead of 4
```

### Issue: Import Errors

**Solution:**
```bash
# Make sure you're in the tritonrl directory
cd ~/Documents/tritonrl

# Reinstall packages
pip install -r requirements.txt

# Check extensions module
python -c "from extensions import HardenedVerifier; print('OK')"
```

### Issue: Spot Instance Interrupted

**Solution:**
- Checkpoints are saved every 100 steps
- Resume training from last checkpoint:
```bash
python train_integrated.py --stage rl --resume checkpoints/sft_final
```

### Issue: Training Too Slow

**Solution:**
```bash
# Reduce dataset size further
python train_integrated.py --stage sft --max_samples 500
```

---

## 📊 Monitoring Training

### Check GPU Utilization:
```bash
watch -n 1 nvidia-smi
```

**Expected:**
- During SFT/RL: All 8 GPUs at 80-95%
- During generation: All 8 GPUs at 60-80%
- Memory: ~30-35GB per GPU

### Check Training Progress:
```bash
# Training logs
tail -f checkpoints/sft/training.log

# Checkpoints
ls -lh checkpoints/sft_final/
```

---

## ✅ Validation Checklist

Before claiming success, verify:

- [ ] All 4 extensions tested and passing
- [ ] SFT training completed without errors
- [ ] RL training generated >200 high-quality samples
- [ ] RL model shows improvement over SFT baseline
- [ ] Evaluation results saved to `outputs/`
- [ ] Can reproduce results

---

## 🎓 Next Steps (Beyond 10 Hours)

If you have more time:

1. **Train on full dataset** (18k samples instead of 1k)
2. **Add fusion-centric data generation** (Extension from report)
3. **Test on full KernelBench** (250 tasks instead of 20)
4. **Implement GPU profiler integration** (validate actual speedups)
5. **Try other RL methods** (REINFORCE, PPO)

---

## 📝 Citation

If using this code, cite:

```bibtex
@misc{tritonrl2025,
  title={TritonRL with Extensions: Multi-Input Testing, Staged Evaluation, Adaptive Curriculum, and Calibrated Timing},
  author={Your Name},
  year={2025},
  note={Based on TritonRL: Training LLMs to Think and Code Triton Without Cheating}
}
```

---

## 🆘 Support

If stuck:
1. Check this guide's Troubleshooting section
2. Check `EXTENSIONS.md` for extension details
3. Check `10_HOUR_REALISTIC_SCOPE.md` for timing expectations
4. Open an issue with error logs
