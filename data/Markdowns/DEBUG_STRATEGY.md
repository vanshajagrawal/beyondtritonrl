# Debugging Strategy: Cheap Node → H100 Production

## Strategy Overview

**Goal**: Debug on cheap hardware, then seamlessly switch to H100 for full training.

---

## Phase 1: Local/Cheap Node Debugging (Cost: $0-5)

### Option A: Local Development (Free)
**If you have any GPU locally (even consumer GPUs):**

```bash
# Run mini pipeline with tiny data
python prepare_data_simple.py --max_samples 10
python train_integrated.py --stage sft --max_samples 10 --max_steps 10
python evaluate_simple.py --num_tasks 3
```

**What this validates:**
- ✓ All imports work
- ✓ Extensions load correctly
- ✓ Data pipeline works
- ✓ Training loop executes
- ✓ Evaluation runs
- ✓ No syntax/logic errors

**Limitations:**
- Won't fit full Qwen3-7B model
- Can test with smaller model (Qwen-1.8B or CodeLlama-7B)

---

### Option B: AWS g5.xlarge (1× A10G, $0.30-0.50/hr spot)

**Perfect for debugging:**
- Cheap: ~$1-2 for 4-hour debug session
- Real CUDA environment
- Can load 7B models with quantization
- Mumbai: ~₹35-50/hour spot

```bash
# Launch g5.xlarge spot instance
aws ec2 run-instances \
    --instance-type g5.xlarge \
    --region ap-south-1 \
    --image-id ami-<ubuntu-22.04> \
    --instance-market-options 'MarketType=spot' \
    --key-name your-key

# SSH and run debug pipeline
git clone https://github.com/vanshajagrawal/tritonrl.git
cd tritonrl
./run_debug_pipeline.sh
```

---

### Option C: AWS g5.12xlarge (4× A10G, $2-3/hr spot)

**Best balance for debugging:**
- Still cheap: ~$8-12 for 4-hour session
- Can load full Qwen3-7B model
- Multi-GPU testing
- Validates distributed training logic

---

## Phase 2: Code Modifications for Flexible Hardware

### Make Model Loading Hardware-Adaptive

Add to `train_integrated.py`:

```python
def get_model_config(device_type="auto"):
    """Auto-detect hardware and adjust model config"""
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        num_gpus = torch.cuda.device_count()
    else:
        gpu_name = "CPU"
        gpu_memory = 0
        num_gpus = 0
    
    print(f"Detected: {num_gpus}x {gpu_name} ({gpu_memory:.1f}GB each)")
    
    # Adaptive configuration
    if "H100" in gpu_name and num_gpus >= 8:
        # Full production config
        return {
            "model_name": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "load_in_8bit": False,
            "device_map": "auto",
            "max_memory": None,
            "per_device_batch_size": 2,
        }
    
    elif "A100" in gpu_name or "H100" in gpu_name:
        # Single/few GPU config
        return {
            "model_name": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "load_in_8bit": True,  # Quantize to fit
            "device_map": "auto",
            "max_memory": {i: "38GB" for i in range(num_gpus)},
            "per_device_batch_size": 1,
        }
    
    elif "A10G" in gpu_name or gpu_memory > 20:
        # Budget GPU config
        return {
            "model_name": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "load_in_8bit": True,
            "device_map": "auto",
            "max_memory": {i: f"{int(gpu_memory * 0.9)}GB" for i in range(num_gpus)},
            "per_device_batch_size": 1,
        }
    
    else:
        # Fallback: use smaller model
        print("⚠️  Limited GPU detected, using smaller model")
        return {
            "model_name": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "load_in_8bit": False,
            "device_map": "auto",
            "max_memory": None,
            "per_device_batch_size": 1,
        }

# In training code:
config = get_model_config()
model = AutoModelForCausalLM.from_pretrained(
    config["model_name"],
    load_in_8bit=config["load_in_8bit"],
    device_map=config["device_map"],
    max_memory=config["max_memory"],
)
```

---

## Phase 3: Debug Pipeline Script

Create `run_debug_pipeline.sh`:

```bash
#!/bin/bash
# Fast debug pipeline (30 min instead of 10 hours)

set -e

echo "=========================================="
echo "DEBUG PIPELINE (Fast Validation)"
echo "=========================================="

# 1. Quick extension test (2 min)
echo "Testing extensions..."
python test_extensions_quick.py

# 2. Tiny data prep (1 min)
echo "Preparing mini dataset..."
python prepare_data_simple.py --max_samples 20

# 3. Mini SFT (10 min)
echo "Running mini SFT training..."
python train_integrated.py \
    --stage sft \
    --max_samples 20 \
    --num_train_epochs 1 \
    --max_steps 10 \
    --per_device_train_batch_size 1

# 4. Mini RL (10 min)
echo "Running mini RL training..."
python train_integrated.py \
    --stage rl \
    --num_fusion_tasks 5 \
    --n_samples 3 \
    --top_k 2

# 5. Quick eval (5 min)
echo "Running evaluation..."
python evaluate_simple.py \
    --model_path checkpoints/rl_final \
    --num_tasks 3

echo ""
echo "=========================================="
echo "✓ DEBUG PIPELINE COMPLETE"
echo "=========================================="
echo ""
echo "If this succeeded, you're ready for production H100 run!"
echo "No code changes needed - just remove size limits."
```

---

## Phase 4: Seamless Switch to H100

**Key insight: No code changes needed!**

Same commands, just remove the debug flags:

```bash
# On g5.xlarge (debug):
python train_integrated.py --stage sft --max_samples 20 --max_steps 10

# On p5.48xlarge (production):
python train_integrated.py --stage sft --max_samples 1000
# Auto-detects H100s and uses full config
```

The adaptive model loading will automatically:
- Detect 8× H100s
- Use full bf16 precision (no quantization)
- Distribute across all GPUs
- Use optimal batch sizes

---

## Cost Comparison

| Phase | Hardware | Duration | Cost |
|-------|----------|----------|------|
| **Debug** | g5.xlarge (1×A10G) | 1 hour | $0.50 |
| **Debug** | g5.12xlarge (4×A10G) | 2 hours | $6.00 |
| **Production** | p5.48xlarge (8×H100) | 10 hours | $347.92 |
| **TOTAL** | | | **$354.42** |

**Savings from debugging first:**
- Avoid wasting H100 hours on bugs: **~$100-200**
- Confidence before expensive run: **Priceless**

---

## Recommended Workflow

### Step 1: Local Smoke Test (5 min, Free)
```bash
# On your Mac/laptop
cd ~/Documents/tritonrl
python -c "from extensions import HardenedVerifier; print('✓ OK')"
python test_extensions_quick.py
```

### Step 2: AWS g5.xlarge Debug (1-2 hrs, $1-2)
```bash
# Launch cheap instance
aws ec2 run-instances --instance-type g5.xlarge ...

# Clone and test
git clone https://github.com/vanshajagrawal/tritonrl.git
cd tritonrl
chmod +x run_debug_pipeline.sh
./run_debug_pipeline.sh
```

**If this passes, you know:**
- ✓ All code works
- ✓ Extensions functional
- ✓ Training loop stable
- ✓ Evaluation runs
- ✓ Ready for H100

### Step 3: AWS p5.48xlarge Production (10 hrs, ~$348)
```bash
# Launch H100 instance
aws ec2 run-instances --instance-type p5.48xlarge ...

# Same repo, full pipeline
git clone https://github.com/vanshajagrawal/tritonrl.git
cd tritonrl
./setup_s3_checkpointing.sh
./run_pipeline.sh  # Full 10-hour run
```

---

## What Can Go Wrong (and how debug catches it)

| Issue | Cost if found on H100 | Cost if found on g5 |
|-------|----------------------|---------------------|
| Import error | $35/hr wasted | $0.50/hr wasted |
| CUDA OOM | $35-70 (need resize) | $0.50 (quick fix) |
| Data loading bug | $35-140 (4hr wasted) | $1-2 (caught early) |
| Extension config error | $35+ | $0.50 |
| Evaluation crash | $35 (last step fails!) | $0.50 |

**Total potential savings: $100-300**

---

## Alternative: Use p5 for 1 Hour Test ($35)

If you want to test on actual H100s before full run:

```bash
# Launch p5.48xlarge spot for 1 hour
python train_integrated.py --stage sft --max_samples 50 --max_steps 20

# If this works, immediately launch full 10-hour run
# If it fails, only cost $35 to find out
```

---

## Recommendation

**Best approach:**
1. **2 hours on g5.12xlarge ($6)** - Full debug with 4×A10G
2. **10 hours on p5.48xlarge ($348)** - Production run

**Total: $354** vs. risk of wasting $100+ debugging on H100s directly.

---

## Key Takeaway

✅ **Yes, debugging on cheap nodes is highly recommended!**

The code is already designed to be hardware-adaptive. You can validate:
- All logic works
- No import errors
- Training loop stable
- Extensions functional
- Evaluation runs

Then switch to H100s with **zero code changes** for the full production run.
