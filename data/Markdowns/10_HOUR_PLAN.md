# Realistic 10-Hour Implementation Plan with Claude Code

## Assumptions
- You have: 10 hours, H100s, Claude Code, GPT API access
- Reality: ~40% time will be debugging, API issues, environment setup
- Effective coding time: ~6 hours
- Buffer for unforeseen issues: 2 hours

---

## ✅ REALISTIC SCOPE (What Claude Code Can Do)

### Phase 1: Core Infrastructure (2 hours)
**What Claude Code handles well:**
- ✅ File structure setup
- ✅ Configuration management
- ✅ Data loading/preprocessing
- ✅ Basic training loops

**Human intervention needed:**
- API key setup
- Environment troubleshooting
- Dependency conflicts

**Deliverables:**
1. Working data pipeline
2. Basic SFT training script
3. Config management

---

### Phase 2: Easiest Extension - Multi-Input Testing (1.5 hours)
**Why this one:**
- Pure Python logic (no RL complexity)
- No external dependencies beyond PyTorch
- Easy to test incrementally
- High value/effort ratio

**What Claude Code does:**
- Implement `_generate_test_inputs()`
- Override `correctness_check()`
- Add shape/value variations
- Unit tests

**Human does:**
- Run tests
- Fix edge cases (dtype mismatches, shape errors)
- Verify correctness

**Expected bugs:**
- Shape broadcasting issues (30 min)
- Dtype conversion errors (15 min)
- OOM on large test suites (15 min)

---

### Phase 3: Verification Funnel (1 hour)
**Why this one:**
- Independent of RL
- Pure control flow
- Easy to validate

**What Claude Code does:**
- Implement staged evaluation pipeline
- Early pruning logic
- Metrics tracking

**Human does:**
- Test on real kernels
- Tune thresholds

**Expected bugs:**
- Exception handling in stages (20 min)
- Metrics not tracking correctly (10 min)

---

### Phase 4: Integration & Testing (2.5 hours)
**The hardest part - where most time goes:**

**What Claude Code does:**
- Wire extensions into existing code
- Write integration tests
- Debug scripts

**Human does (CRITICAL - Claude struggles here):**
- GPU memory issues
- CUDA errors
- Triton compilation failures
- API rate limits
- Distributed training setup

**Expected issues:**
- Import errors (30 min)
- Config loading bugs (20 min)
- GPU out of memory (45 min)
- Verifier timeout issues (30 min)
- Multi-GPU coordination (35 min)

---

### Phase 5: Minimal Training Run (3 hours)
**Goal: Get ONE successful training run on small data**

**What Claude Code does:**
- Debug training loop
- Fix tensor shape mismatches
- Add logging/checkpointing

**Human does (WHERE TIME DISAPPEARS):**
- Wait for training iterations
- Monitor GPU utilization
- Debug CUDA errors
- Handle API failures (OpenAI/Anthropic rate limits)
- Fix data pipeline bottlenecks

**Expected time sinks:**
- First training run crashes (45 min debugging)
- Verifier too slow (30 min optimization)
- GPU memory optimization (45 min)
- Data loading bottlenecks (30 min)
- Checkpoint saving issues (30 min)

---

## ❌ OUT OF SCOPE (Can't Realistically Do in 10 Hours)

### Definitely Skip:
1. **Fusion Data Generation** - 3+ hours
   - Requires torch.fx expertise
   - Many edge cases per pattern
   - Hard to validate correctness

2. **Hardened Sandboxing** - 2+ hours
   - Security requires careful testing
   - Subprocess management bugs
   - Platform-specific issues

3. **Calibrated Timing** - 1.5+ hours
   - Subtle CUDA synchronization bugs
   - Hard to debug performance issues

4. **Adaptive Curriculum** - 2+ hours
   - Requires full RL training to test
   - Scheduling logic bugs only appear late

5. **Full RL Training** - 5+ hours
   - VeRL integration complex
   - Reward computation bugs
   - Hierarchical credit assignment tricky

6. **Full KernelBench Evaluation** - 2+ hours
   - 250 tasks × 10 samples = 2500 generations
   - Each takes 10-30 seconds
   - Total: 7+ hours just for inference

---

## 🎯 RECOMMENDED 10-HOUR PLAN

### Hour 0-1: Setup (Claude Code: 70% automation)
```
✅ Clone repo
✅ Install dependencies
✅ Set API keys
⚠️ Debug CUDA/Triton installation (you do this)
✅ Download small KernelBook subset (100 tasks)
```

### Hour 1-3: Data Pipeline (Claude Code: 80%)
```
✅ Implement data loading
✅ Add caching
✅ Generate 500 SFT samples (not 58k)
⚠️ Debug OpenAI API rate limits (you handle)
✅ Save processed data
```

### Hour 3-4.5: Multi-Input Extension (Claude Code: 75%)
```
✅ Implement HardenedVerifier
✅ Add multi-input generation
✅ Override correctness_check()
⚠️ Fix shape broadcasting bugs (you debug)
✅ Unit tests
```

### Hour 4.5-5.5: Verification Funnel (Claude Code: 85%)
```
✅ Implement StagedEvaluator
✅ Add early pruning
✅ Metrics tracking
⚠️ Test on real kernels (you validate)
```

### Hour 5.5-8: Integration & First Training (Claude Code: 50%)
```
✅ Wire extensions to training
✅ Create minimal SFT script
⚠️ Debug GPU memory (YOU - this takes time)
⚠️ Fix import errors (you + Claude)
⚠️ Handle CUDA errors (YOU)
✅ Run 1 epoch on 100 samples
✅ Save checkpoint
```

### Hour 8-10: Validation & Documentation (Claude Code: 60%)
```
✅ Run verifier on 20 test kernels
⚠️ Fix verifier bugs discovered (you + Claude)
✅ Benchmark speedup
✅ Generate report
⚠️ Unexpected issues buffer (YOU)
```

---

## 📊 REALISTIC OUTCOMES AFTER 10 HOURS

### ✅ What You'll Have:
1. **Working data pipeline** (100 KernelBook tasks)
2. **Multi-input verifier** (5 test variations)
3. **Staged evaluation** (early pruning working)
4. **Minimal SFT training** (1-2 epochs on small data)
5. **Integration tests** (verifier validated on real kernels)
6. **Basic metrics** (correctness, validity tracking)

### ❌ What You Won't Have:
1. Full 58k dataset (would take 6+ hours to generate)
2. RL training (needs 4+ hours + debugging)
3. Fusion data generation (3+ hours)
4. Full KernelBench eval (7+ hours)
5. Sandboxing (2+ hours)
6. Production-ready model (needs 20+ GPU-hours training)

### 📈 What You Can Claim:
- "Implemented multi-input verification system"
- "Built staged evaluation pipeline with 40% speedup"
- "Validated on 20 Triton kernels"
- "Demonstrated improved correctness over baseline verifier"

---

## 🐛 EXPECTED DEBUGGING TIME BREAKDOWN

| Issue Type | Time | Who Fixes |
|------------|------|-----------|
| Import/config errors | 30 min | Claude 60%, You 40% |
| GPU memory issues | 45 min | You 90% |
| Shape mismatches | 30 min | Claude 70%, You 30% |
| CUDA errors | 45 min | You 95% |
| API rate limits | 20 min | You 100% |
| Verifier timeouts | 30 min | Claude 50%, You 50% |
| Data loading bugs | 25 min | Claude 80%, You 20% |
| Multi-GPU issues | 35 min | You 90% |
| **TOTAL DEBUGGING** | **4 hours** | **You: 65%** |

---

## 🎯 MAXIMIZE CLAUDE CODE EFFECTIVENESS

### Claude Code is GREAT at:
1. Boilerplate code generation
2. Config management
3. Pure Python logic
4. Test writing
5. Documentation
6. Refactoring existing code

### Claude Code STRUGGLES with:
1. GPU memory optimization (needs profiling)
2. CUDA errors (needs hardware knowledge)
3. Distributed training (complex state)
4. Performance tuning (needs benchmarking)
5. API rate limit handling (needs retry logic)
6. Environment-specific bugs (needs system knowledge)

### Your Role (Critical):
1. **Decision making**: What to implement first
2. **Validation**: Does the verifier actually work?
3. **GPU debugging**: OOM, CUDA errors
4. **Training monitoring**: Is it learning?
5. **Performance optimization**: Too slow?
6. **Integration testing**: End-to-end flow

---

## 🚀 RECOMMENDED STRATEGY

### Hour 0-2: Let Claude Code Run Wild
- Implement data pipeline
- Set up configs
- Generate small dataset
- You just supervise and fix API keys

### Hour 2-5: Collaborative on Extensions
- Claude implements logic
- You immediately test each function
- Fix bugs together iteratively
- Don't move on until each piece works

### Hour 5-8: You Drive, Claude Assists
- You set up training environment
- Claude fixes errors you identify
- You monitor GPU/memory
- Claude adds logging/debugging

### Hour 8-10: Validation & Polish
- You validate on real kernels
- Claude generates reports
- You benchmark performance
- Claude documents findings

---

## 💡 PRAGMATIC CUTS TO MAKE IT WORK

### Use These Shortcuts:
1. **Tiny dataset**: 100 tasks not 11k
2. **Single GPU**: Skip distributed
3. **No RL**: Just SFT + verification
4. **Subset KernelBench**: 20 tasks not 250
5. **Lower quality**: Focus on "works" not "perfect"
6. **Mock expensive calls**: Cache OpenAI responses
7. **Skip fusion data**: Use existing KernelBook only

### Critical Path Only:
```
Data → Multi-Input Verifier → Staged Eval → Integration Test → Done
```

Skip everything else.

---

## ⏱️ HONEST TIME ALLOCATION

```
Claude Code effective work: 6 hours
Your debugging/validation:  2.5 hours
Unexpected issues buffer:   1.5 hours
-------------------------
Total:                      10 hours
```

**Bottom Line:** You can get a **working proof-of-concept** with multi-input verification and staged evaluation validated on real kernels. You won't have a full trained model, but you'll have the infrastructure to train one later.

---

## 🎓 DELIVERABLE AFTER 10 HOURS

```python
# You'll be able to run:
python train_with_extensions.py \
    --preset multi_input_only \
    --stage sft \
    --num_tasks 100 \
    --num_epochs 2

# Output:
# ✓ Trained on 100 tasks
# ✓ Multi-input verifier working
# ✓ Staged evaluation implemented
# ✓ Validated on 20 test kernels
# ✓ Metrics: 45% correct (vs 30% baseline single-input)
```

This is realistic and achievable. Trying to do more will result in nothing working.
