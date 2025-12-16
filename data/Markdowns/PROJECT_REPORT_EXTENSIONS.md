# Extensions from "Beyond TritonRL" Project Report

Based on the project report LaTeX file, here are the **specific extensions** proposed:

---

## 📋 Core Extensions (From Project Abstract & Introduction)

The project proposes **4 main upgrades** over the TritonRL baseline:

### 1️⃣ **Fusion-Centric Training Set**
**Description:**
- Programmatically generate fused operation tasks
- Focus on common patterns:
  - `Conv → BN → ReLU`
  - `GEMM → bias → activation`
  - `LN → GELU`
- Use `torch.fx` graphs and templates
- Enumerate shapes/dtypes/layouts
- Provide strict non-cheating PyTorch references

**Why:**
- TritonRL only got 7% correct on Level 2 (fusion tasks)
- Fusion is where models struggle most
- Explicit fusion focus improves L2 performance

**Implementation Status:** ✅ Implemented in `extensions/fusion_data.py`

---

### 2️⃣ **Hardened Verifier 2.0**

**Components:**

#### 2A. Static AST Linter
- Check presence of `@triton.jit`
- Verify real kernel body exists
- Ensure kernel is actually invoked
- Disallow `torch.nn` usage
- Block tensor constant shortcuts
- Filter forbidden imports

#### 2B. Strict Import Sandbox
- Restricted Python environment
- No file/network access
- Monkey-patched `torch` fallbacks
- Syscall filters (optional)

#### 2C. Multi-Input & Type Tests
- **Multiple test inputs per task** (not just one)
- Randomized shapes (when legal)
- Value perturbations
- Dtype variations
- Metamorphic invariances:
  - Scaling: `f(k*x) ≈ k*f(x)`
  - Permutations (where applicable)

#### 2D. GPU-Event Gating
- Require nontrivial FLOPs/bytes moved
- Verify at least one successful kernel launch
- Optional roofline sanity bands

#### 2E. Calibrated Timing
- Warmup runs before measurement
- Explicit device synchronization
- N timed trials (not just 1)
- **Trimmed mean** (remove outliers)
- Confidence intervals
- Separate tiny-batch and full-batch regimes
- Avoid cache/cold-start confounds

**Why:**
- TritonRL's single-input testing is brittle
- Timing noise destabilizes RL
- Models can overfit to specific test cases
- Need robust verification to prevent reward hacking

**Implementation Status:**
- ✅ Multi-input: `extensions/hardened_verifier.py`
- ✅ Calibrated timing: `extensions/hardened_verifier.py`
- ✅ Sandboxing: `extensions/hardened_verifier.py`
- ⚠️ GPU-event gating: Not implemented (would need CUDA profiling)
- ⚠️ Metamorphic testing: Not implemented (easy to add)

---

### 3️⃣ **Adaptive Curriculum (Capacity-Aware)**

**Components:**

#### 3A. Dynamic L1/L2 Sampling Schedule
- Start with small `p₀` (probability of L2 sampling)
- Increase `p_t` over time as L1 correctness stabilizes
- Explicitly schedule: `p_t = Pr[L2 | time=t]`
- Monitor pass@k correctness on L1
- Trigger curriculum shift when L1 performance plateaus

#### 3B. Staged Reward Shaping
- Reward formula:
  ```
  r = valid · (β · correct + (1-β) · rank_speed)
  ```
- `β` starts high (focus on correctness)
- `β` decreases over training (increase speed emphasis)
- `rank_speed` is pairwise, noise-robust ranking (winsorized)
- Prevents outlier timing from dominating

#### 3C. Cosine Decay for Plan Weight
- Plan weight `α` follows slow cosine decay
- Planning stabilizes early
- Code keeps adapting under stricter speed requirements

**Why:**
- Static L1/L2 mixing underuses capacity
- L2 underexposed early, over-penalized later
- Need adaptive difficulty progression
- Correctness first, then speed

**Implementation Status:**
- ✅ Dynamic sampling: `extensions/curriculum.py`
- ✅ Cosine decay: `extensions/curriculum.py`
- ⚠️ Rank-based speed reward: Not implemented (uses absolute speedup)

---

### 4️⃣ **Verification Funnel for Throughput**

**Pipeline:**
```
AST/imports → compile → tiny-run correctness → full-run correctness → calibrated speed
```

**Details:**
- **Staged evaluation** with early pruning
- Stop at first failure (don't waste compute)
- Tokens receive credit only after passing current stage
- Failures zero out reward immediately

**Stages:**
1. **AST/imports**: Fast static checks
2. **Compile**: Can code compile?
3. **Tiny-run**: Test on small batch (4x4 tensors)
4. **Full-run**: Test on actual batch size
5. **Calibrated speed**: Expensive timing only for correct kernels

**Why:**
- Verifier cost is high if timing attempted on bad code
- Most failures caught early (AST, compile)
- Save expensive GPU time for promising candidates
- 40% throughput improvement estimated

**Implementation Status:** ✅ Implemented in `extensions/staged_eval.py`

---

## 📊 Summary Table

| Extension | From Report | Implemented | Complexity | Impact |
|-----------|-------------|-------------|------------|--------|
| **Fusion-centric data** | ✅ | ✅ | Medium | High |
| Multi-input testing | ✅ | ✅ | Low | High |
| Calibrated timing | ✅ | ✅ | Low | Medium |
| Strict sandboxing | ✅ | ✅ | Medium | High |
| GPU-event gating | ✅ | ❌ | Medium | Medium |
| Metamorphic testing | ✅ | ❌ | Low | High |
| **Adaptive curriculum** | ✅ | ✅ | Low | High |
| Cosine decay α | ✅ | ✅ | Low | Medium |
| Rank-based speed reward | ✅ | ❌ | Low | Medium |
| **Verification funnel** | ✅ | ✅ | Low | High |
| Tiny-batch testing | ✅ | ✅ | Low | Medium |

---

## ✅ What's Implemented (7/11 components)

1. ✅ **Fusion-centric data generation** - Core focus on L2 tasks
2. ✅ **Multi-input testing** - 5 test variations per kernel
3. ✅ **Calibrated timing** - Warmup, sync, trimmed mean
4. ✅ **Strict sandboxing** - Restricted execution environment
5. ✅ **Adaptive curriculum** - Dynamic L1/L2 sampling
6. ✅ **Cosine decay** - Plan weight scheduling
7. ✅ **Verification funnel** - Staged evaluation with early pruning

---

## ⚠️ What's Missing (4/11 components)

### Easy to Add (2-3 hours each):

1. **GPU-Event Gating**
   - Use `torch.cuda.Event()` to verify kernel launched
   - Check FLOPs/bytes moved
   - Roofline model validation
   - **Effort:** 2 hours

2. **Metamorphic Testing**
   - Test scaling invariance: `f(2x) ≈ 2f(x)`
   - Test commutativity where applicable
   - Test permutation invariances
   - **Effort:** 2 hours

3. **Rank-Based Speed Reward**
   - Replace absolute speedup with pairwise ranking
   - Winsorize outliers
   - More robust to timing noise
   - **Effort:** 1 hour

4. **Full Tiny-Batch Coverage**
   - Currently implemented but could be expanded
   - Test more shape variations in tiny mode
   - **Effort:** 1 hour

---

## 🎯 Key Differences from TritonRL Baseline

| Aspect | TritonRL Baseline | Project Report Extensions |
|--------|-------------------|---------------------------|
| **Training Data** | KernelBook as-is | + Fusion-centric generation |
| **Correctness Testing** | Single test input | Multiple inputs + metamorphic |
| **Timing** | Basic measurement | Calibrated with trimmed mean |
| **Curriculum** | Static L1/L2 mix | Adaptive based on progress |
| **Verification** | Single-stage | Staged funnel (5 stages) |
| **Sandbox** | Basic checks | Strict import/syscall isolation |
| **Speed Reward** | Absolute speedup | Rank-based (planned) |
| **Plan Update** | Fixed α | Cosine decay α |

---

## 💡 Recommendations

### For 10-Hour Implementation:
Keep the **7 components already implemented** ✅

### For Extended Work (+2-3 hours):
Add the **2 easiest missing components**:
1. Metamorphic testing (2 hours)
2. Rank-based speed reward (1 hour)

### Total Achievable in 12-13 Hours:
**9/11 components** (82% coverage)

The remaining 2 (GPU-event gating, full tiny-batch) are nice-to-have but not critical for validating the approach.

---

## 📝 Project Report Claims

The report claims these extensions address TritonRL's limitations:

1. ✅ **"Single-input correctness is brittle"** → Multi-input testing
2. ✅ **"Timing noise destabilizes RL"** → Calibrated timing
3. ✅ **"Static L1/L2 mixing underuses capacity"** → Adaptive curriculum
4. ✅ **"Verifier cost is high"** → Verification funnel
5. ✅ **"Kernel Fusion emphasis implicit"** → Fusion-centric data
6. ✅ **"Sandboxing can be tightened"** → Strict import/syscall isolation

All 6 main claims are **addressed by implemented extensions** ✅
