# TritonRL Extensions Guide

This document explains how to use the modular extension system to enable/disable improvements individually.

## Quick Start

```bash
# Train with baseline (no extensions)
python train_with_extensions.py --preset baseline

# Train with easiest extension (multi-input testing)
python train_with_extensions.py --preset multi_input_only

# Train with recommended combo
python train_with_extensions.py --preset easy_combo

# Train with all extensions
python train_with_extensions.py --preset all_extensions
```

---

## Available Extensions

### Extension 1: Fusion-Centric Data ⚡ Priority: HIGH

**What it does:** Generates fused operation tasks (Conv→BN→ReLU, GEMM→bias→act, LN→GELU)

**Complexity:** Medium
**Value:** High (addresses Level 2 weakness)
**Files:** `extensions/fusion_data.py`

**Enable:**
```python
from config_examples import get_config
configs = get_config('fusion_data_only')
```

---

### Extension 2: Multi-Input Testing ✅ Priority: P0 (EASIEST)

**What it does:** Tests kernels on multiple shape/value/dtype variations

**Complexity:** Low
**Value:** High (prevents overfitting)
**Files:** `extensions/hardened_verifier.py` (lines 25-117)

**Enable:**
```python
configs = get_config('multi_input_only')
```

**How it works:** Overrides `correctness_check()` to test 5 input variations instead of 1.

---

### Extension 3: Adaptive Curriculum 📈 Priority: P0

**What it does:** Dynamically adjusts L1/L2 sampling based on training progress

**Complexity:** Low
**Value:** High (efficient learning)
**Files:** `extensions/curriculum.py`

**Enable:**
```python
configs = get_config('curriculum_only')
```

**How it works:** Starts with 10% L2 tasks, increases to 50% as L1 correctness stabilizes.

---

### Extension 4: Hardened Sandboxing 🔒 Priority: P1

**What it does:** Restricts code execution environment (no file/network access)

**Complexity:** Medium
**Value:** High (security)
**Files:** `extensions/hardened_verifier.py` (lines 209-260)

**Enable:**
```python
configs = get_config('sandbox_only')
```

---

### Extension 5: Calibrated Timing ⏱️ Priority: P1

**What it does:** Robust timing with warmup, CUDA events, trimmed mean

**Complexity:** Low
**Value:** Medium (reduces timing noise)
**Files:** `extensions/hardened_verifier.py` (lines 119-207)

**Enable:**
```python
configs = get_config('calibrated_timing_only')
```

**How it works:** Overrides `speedup_metric()` to use 20 warmup runs, 100 trials, and outlier trimming.

---

### Extension 6: Verification Funnel 🔍 Priority: P1

**What it does:** Multi-stage evaluation to prune bad candidates early

**Complexity:** Low
**Value:** High (efficiency)
**Files:** `extensions/staged_eval.py`

**Enable:**
```python
configs = get_config('staged_eval_only')
```

**Pipeline:** AST → Compile → Tiny-run → Full-run → Timing

---

## Recommended Combinations

### Easiest (Deploy First)
```python
configs = get_config('multi_input_only')
```
- Only multi-input testing
- 1 day implementation
- High value

### Best Value/Effort
```python
configs = get_config('easy_combo')
```
- Multi-input testing
- Adaptive curriculum
- Verification funnel
- 2-3 days implementation
- Covers most critical gaps

### Full System
```python
configs = get_config('all_extensions')
```
- All 6 extensions enabled
- ~10 days implementation
- Maximum performance

---

## Implementation Details

### How Extensions Override Base Functions

Extensions use **inheritance and override** pattern:

```python
# Base verifier (verifiers.py)
class TritonVerifier:
    def correctness_check(self, code, ref, inputs):
        # Single test input
        ...

# Extended verifier (extensions/hardened_verifier.py)
class HardenedVerifier(TritonVerifier):
    def correctness_check(self, code, ref, inputs):
        if not self.config.enable_multi_input:
            return super().correctness_check(...)  # Base behavior

        # Multi-input testing logic
        ...
```

This allows **clean fallback** when extensions are disabled.

---

## Custom Configuration

Create your own config:

```python
from extensions.config import ExtensionConfig

ext_config = ExtensionConfig(
    # Pick what you want
    enable_multi_input=True,
    multi_input_num_tests=10,  # More tests

    enable_adaptive_curriculum=True,
    curriculum_start_p=0.2,  # Start with more L2

    # Disable others
    enable_fusion_data=False,
    enable_strict_sandbox=False,
    enable_calibrated_timing=False,
    enable_staged_eval=False,
)
```

---

## Testing Individual Extensions

Each extension can be tested independently:

```bash
# Test multi-input verifier
python -c "
from extensions import HardenedVerifier
from extensions.config import ExtensionConfig

config = ExtensionConfig(enable_multi_input=True)
verifier = HardenedVerifier(config)
# Test it...
"

# Test fusion data generator
python -c "
from extensions import FusionDataGenerator
from extensions.config import ExtensionConfig

config = ExtensionConfig(enable_fusion_data=True)
generator = FusionDataGenerator(config)
tasks = generator.generate_fusion_tasks()
print(f'Generated {len(tasks)} fusion tasks')
"
```

---

## Priority Order for Implementation

**Phase 1 (Week 1):** Easiest, highest value
1. Multi-input testing ✅
2. Adaptive curriculum ✅
3. Verification funnel ✅

**Phase 2 (Week 2):** Medium complexity
4. Calibrated timing ⏱️
5. Fusion-centric data ⚡

**Phase 3 (Week 3):** More complex
6. Hardened sandboxing 🔒

---

## Files Overview

```
extensions/
├── __init__.py              # Main exports
├── config.py                # ExtensionConfig with feature flags
├── fusion_data.py           # Extension 1: Fusion data generation
├── hardened_verifier.py     # Extensions 2,4,5: Multi-input, sandbox, timing
├── curriculum.py            # Extension 3: Adaptive curriculum
└── staged_eval.py           # Extension 6: Verification funnel

config_examples.py           # Pre-configured presets
train_with_extensions.py     # Training script with extension support
```
