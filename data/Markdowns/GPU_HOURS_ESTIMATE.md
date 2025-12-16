# GPU-Hours Estimate: 8x H100 40GB Node (On-Demand)

## 🖥️ Hardware Setup
- **Node:** 8x H100 40GB
- **Cost:** ~$32/hour for p5.48xlarge (AWS on-demand)
- **Network:** NVLink/NVSwitch interconnect
- **Memory per GPU:** 40GB

---

## ⏱️ GPU-Hours Breakdown by Task

### Phase 1: Data Preparation (Minimal GPU Usage)
**Task:** Load KernelBook, preprocess
**GPUs needed:** 0 (CPU-only)
**Time:** 30 min
**GPU-hours:** 0

---

### Phase 2: SFT Training (Main GPU Usage)

#### Configuration:
- **Model:** Qwen3-7B (7B parameters)
- **Dataset:** 1,000 samples (reduced from 18k)
- **Batch size per GPU:** 2 (conservative for 40GB)
- **Gradient accumulation:** 4
- **Effective batch size:** 8×2×4 = 64
- **Epochs:** 1
- **Sequence length:** 8,192 tokens (plan + code)

#### Time Estimate:
```
Total samples: 1,000
Effective batch size: 64
Steps per epoch: 1,000 / 64 = ~16 steps
Time per step: ~45 seconds (7B model, long sequences)
Total training time: 16 × 45s = ~12 minutes

With overhead (checkpoint, validation): ~20 minutes
```

**GPUs needed:** 8 (data parallel)
**Time:** 20 minutes
**GPU-hours:** 8 GPUs × 0.33 hours = **2.7 GPU-hours**

---

### Phase 3: Extension Implementation (Minimal GPU)
**Tasks:**
- Multi-input testing implementation
- Verification funnel implementation
- Adaptive curriculum implementation
- Calibrated timing implementation

**GPUs needed:** 0-1 (testing only)
**Time:** 4 hours
**GPU-hours:** 1 GPU × 4 hours = **4 GPU-hours**

---

### Phase 4: Verification Testing (Moderate GPU)

#### Testing on Real Kernels:
- **Number of kernels:** 20 test cases
- **Samples per kernel:** 10 generations
- **Total generations:** 200

#### Generation Time:
```
Time per generation: 30 seconds (for Qwen3-7B)
Parallel generations: 8 GPUs
Sequential batches: 200 / 8 = 25 batches
Total time: 25 × 30s = 12.5 minutes
```

#### Verification Time:
```
Per kernel:
- Syntax check: 0.1s
- Compile: 2s
- Tiny-run: 1s
- Full-run: 3s
- Timing (100 trials): 10s
Total per kernel: ~16s

For 200 samples: 200 × 16s = 53 minutes
(Some parallel, so ~30 minutes actual)
```

**GPUs needed:** 8 (for generation) + 1-2 (for verification)
**Time:** 45 minutes
**GPU-hours:** 8 GPUs × 0.75 hours = **6 GPU-hours**

---

### Phase 5: Validation & Metrics (Light GPU)
**Tasks:**
- Collect metrics
- Generate comparison tables
- Run ablations (test with/without extensions)

**GPUs needed:** 1-2
**Time:** 1 hour
**GPU-hours:** 2 GPUs × 1 hour = **2 GPU-hours**

---

## 📊 Total GPU-Hours Summary

| Phase | Task | GPUs | Time | GPU-Hours |
|-------|------|------|------|-----------|
| 1 | Data prep | 0 | 0.5h | 0 |
| 2 | SFT training | 8 | 0.33h | 2.7 |
| 3 | Extensions impl | 1 | 4h | 4.0 |
| 4 | Verification testing | 8 | 0.75h | 6.0 |
| 5 | Validation | 2 | 1h | 2.0 |
| | **Subtotal** | | **6.5h** | **14.7** |
| | **Buffer (30%)** | | **+2h** | **+4.4** |
| | **TOTAL** | | **~8.5h** | **~19** |

---

## 💰 Cost Estimate

### AWS p5.48xlarge Pricing:
- **On-demand rate:** ~$32.77/hour
- **Spot rate:** ~$10-15/hour (70% savings, but can be interrupted)

### Total Cost:
```
On-demand: 8.5 hours × $32.77 = $278
Spot:      8.5 hours × $12.00 = $102
```

**Recommendation:** Use **spot instances** with automatic checkpoint recovery

---

## 🎯 Optimized Schedule (Minimize Cost)

### Strategy: Use GPUs Only When Needed

#### Active GPU Time Breakdown:
| Hours 0-1 | CPUs only | Data prep | $0 |
| Hours 1-1.5 | 8×H100 | SFT training | $16 |
| Hours 1.5-5.5 | 1×H100 | Extensions (testing) | $16 |
| Hours 5.5-6.5 | 8×H100 | Verification | $33 |
| Hours 6.5-7.5 | 2×H100 | Validation | $4 |
| Hours 7.5-10 | CPUs only | Documentation | $0 |

**GPU active time:** ~2.5 hours (29% of 10 hours)
**Actual GPU-hours:** ~19 GPU-hours
**Cost:** ~$82 (if you only pay for active GPU time)

---

## ⚡ Further Optimizations

### 1. Use Smaller Model for Testing
- **Qwen3-1.5B** instead of Qwen3-7B for extension testing
- 4x faster, same validation
- **Saves:** ~2 GPU-hours → **17 GPU-hours total**

### 2. Reduce Test Set
- 10 kernels instead of 20
- 5 samples instead of 10
- **Saves:** ~3 GPU-hours → **14 GPU-hours total**

### 3. Use Single GPU for Extensions
- Most extension code doesn't need 8 GPUs
- Spin down 7 GPUs during hours 1.5-5.5
- **Saves:** 28 GPU-hours → **10 GPU-hours total** (but node still billed)

### 4. Sequential Execution
- Train SFT first (0.5h, 8 GPUs) = 4 GPU-hours
- Implement extensions on CPU (4h) = 0 GPU-hours
- Test everything (1h, 8 GPUs) = 8 GPU-hours
- **Total:** 12 GPU-hours

---

## 🚨 Reality Check: Node Billing

### AWS Billing Model:
When you rent 8×H100 node, you pay for:
- **The entire node** (~$32/hour)
- **All 8 GPUs** (even if you use just 1)
- **Minimum 1-hour blocks**

### Actual Cost Calculation:
```
Hour 0-1:   Setup + data prep            = $32
Hour 1-2:   SFT training (8 GPUs)        = $32
Hour 2-6:   Extensions (1 GPU used)      = $128 (4 hours × $32)
Hour 6-7:   Testing (8 GPUs)             = $32
Hour 7-8:   Validation (2 GPUs)          = $32
Hour 8-10:  Documentation (0 GPUs)       = $64 (or shut down)

Total if you keep node up: $320
Total if you shut down when not needed: ~$160-200
```

**Key Insight:** You're paying for the **entire node**, not per-GPU utilization.

---

## 💡 Cost-Optimized Strategy

### Option 1: Keep Node Running (Simplest)
- **Time:** 10 hours
- **Cost:** $327 ($32.77 × 10)
- **Pros:** No interruptions, simple workflow
- **Cons:** Pay for idle time

### Option 2: Aggressive Shutdown (Cheapest)
- **Active periods:**
  - Hour 1: SFT (8 GPUs) = $33
  - Hour 5-6: Testing (8 GPUs) = $33
  - Hour 7: Validation (2 GPUs, but pay for 8) = $33
- **Total active:** ~3 hours = **$99**
- **Pros:** Saves $230
- **Cons:** Spin-up time, potential data loss

### Option 3: Spot Instances (Best Balance)
- **Time:** 10 hours
- **Cost:** ~$120 ($12 × 10)
- **Pros:** 70% savings
- **Cons:** Can be interrupted (but checkpoints mitigate)

---

## 🎯 RECOMMENDED APPROACH

### Use Spot Instances with Checkpointing

**Setup:**
1. Request p5.48xlarge spot instance (~$12/hour)
2. Set max price: $15/hour (higher than typical spot)
3. Enable auto-recovery with checkpoints

**Schedule:**
```
Hour 0-1:    Data prep + setup
Hour 1-1.5:  SFT training (checkpoint every 5 min)
Hour 1.5-6:  Implement extensions (checkpoint code)
Hour 6-7:    Verification testing
Hour 7-8:    Validation + metrics
Hour 8-10:   Documentation (can shut down node)
```

**Cost:**
- 8 hours × $12 = **$96**
- Insurance buffer: +$20
- **Total: ~$120**

**GPU-hours consumed:** ~19 GPU-hours
**Node-hours billed:** ~8 hours

---

## 📋 Final Recommendations

### For Your 10-Hour Budget:

**Recommended Configuration:**
- **Instance:** p5.48xlarge spot (~$12/hour)
- **Duration:** 8 hours active + 2 hours documentation (can shut down)
- **Expected cost:** $96-120
- **GPU-hours:** ~19 GPU-hours (but billed for 64 node-hours)

### Checkpoint Strategy:
```python
# Save every 5 minutes during training
checkpoint_callback = ModelCheckpoint(
    every_n_train_steps=5,
    save_top_k=-1,  # keep all
)

# Save after each extension implementation
if extension_complete:
    torch.save(model.state_dict(), f'checkpoint_{extension_name}.pt')
```

### Cost Breakdown:
| Scenario | Strategy | Duration | Cost |
|----------|----------|----------|------|
| **Conservative** | On-demand, keep running | 10h | $327 |
| **Balanced** | Spot, keep running | 8h | $96-120 |
| **Aggressive** | Spot, shutdown idle | 3h | $36-45 |

**Recommended:** **Balanced approach** (~$120)

---

## ⚠️ Important Caveats

### 1. Training Might Take Longer
- First time setup: +30 min
- Debugging: +1-2 hours
- **Safe estimate:** 10-12 hours → **$120-150**

### 2. You're Paying for the Node, Not GPU Utilization
- 8×H100 node is $32/hour **regardless** of how many GPUs you use
- Can't "turn off" individual GPUs to save money
- Only way to save: shut down entire node

### 3. Spot Instance Risks
- Can be interrupted with 2-minute warning
- ~5% chance per hour (AWS empirical data)
- For 8 hours: ~35% chance of interruption
- **Mitigation:** Frequent checkpoints

---

## 🎓 Answer to Your Question

### "How many GPU-hours do you think you would need?"

**Answer:** ~**19 GPU-hours** of actual compute

**BUT** since you rent by the node:
- **Node-hours needed:** ~8 hours
- **Cost:** ~$96-120 (spot) or ~$260 (on-demand)

### What this means:
You'll use about **19 GPU-hours** of compute (actual work), but you'll **pay for 64 GPU-hours** (8 GPUs × 8 hours) because that's how node billing works.

**Efficiency:** 19/64 = **30% GPU utilization** (typical for development work)

For production training (where you use all 8 GPUs fully), efficiency would be 80-90%.

---

## 💰 FINAL ANSWER

**Recommended budget:** **$120** (spot) or **$260** (on-demand)
**Time needed:** **8 hours** of node time
**Actual GPU work:** **19 GPU-hours**
**Cost per GPU-hour:** $6.30 (spot) or $13.70 (on-demand)

This covers:
- ✅ Full SFT training
- ✅ All 4 extensions implemented and tested
- ✅ Validation on 20 kernels
- ✅ 30% debugging buffer
