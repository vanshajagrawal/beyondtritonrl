"""
Extension 2: Multi-Input Testing (easiest)
Extension 4: Hardened Sandboxing (more complex)
Extension 5: Calibrated Timing

This module extends the base TritonVerifier with optional enhancements.
Each enhancement can be toggled independently via ExtensionConfig.
"""

import torch
import numpy as np
import time
from typing import List, Tuple, Optional
import sys
from verifiers import TritonVerifier

class HardenedVerifier(TritonVerifier):
    """Extended verifier with multi-input, sandboxing, and calibrated timing"""

    def __init__(self, ext_config):
        super().__init__()
        self.config = ext_config

    # ============================================================================
    # Extension 2: Multi-Input Testing (EASIEST - HIGH VALUE)
    # ============================================================================

    def correctness_check(
        self,
        triton_code: str,
        pytorch_ref_code: str,
        test_inputs: list,
        tolerance: float = 1e-5
    ) -> bool:
        """
        Override base correctness_check to add multi-input testing

        If enable_multi_input=False, falls back to base implementation
        If enable_multi_input=True, tests on multiple input variations
        """
        if not self.config.enable_multi_input:
            # Fall back to base implementation
            return super().correctness_check(triton_code, pytorch_ref_code, test_inputs, tolerance)

        # Multi-input testing
        test_suites = self._generate_test_inputs(pytorch_ref_code, test_inputs)

        for i, test_suite in enumerate(test_suites):
            try:
                pytorch_out = self._execute_pytorch(pytorch_ref_code, test_suite)
                triton_out = self._execute_triton(triton_code, test_suite)

                if pytorch_out is None or triton_out is None:
                    return False

                if not torch.allclose(triton_out, pytorch_out, atol=tolerance):
                    print(f"Multi-input test {i+1} failed")
                    return False

            except Exception as e:
                print(f"Multi-input test {i+1} error: {e}")
                return False

        return True

    def _generate_test_inputs(self, pytorch_ref: str, base_inputs: list) -> List[list]:
        """Generate multiple test input variations"""
        test_suites = [base_inputs]  # Include original

        for _ in range(self.config.multi_input_num_tests - 1):
            variant = []

            for inp in base_inputs:
                if not isinstance(inp, torch.Tensor):
                    variant.append(inp)
                    continue

                # Shape variation (if enabled)
                if self.config.multi_input_shape_variations:
                    new_inp = self._vary_shape(inp)
                else:
                    new_inp = inp.clone()

                # Value variation (if enabled)
                if self.config.multi_input_value_variations:
                    new_inp = torch.randn_like(new_inp)

                # Dtype variation (if enabled)
                if self.config.multi_input_dtype_variations:
                    new_inp = self._vary_dtype(new_inp)

                variant.append(new_inp)

            test_suites.append(variant)

        return test_suites

    def _vary_shape(self, tensor: torch.Tensor) -> torch.Tensor:
        """Slightly vary tensor shape (within reasonable bounds)"""
        shape = list(tensor.shape)

        # Randomly adjust one dimension by ±25%
        if len(shape) > 0:
            idx = np.random.randint(0, len(shape))
            if shape[idx] > 4:  # Only vary if dimension is reasonably large
                delta = int(shape[idx] * 0.25)
                shape[idx] += np.random.randint(-delta, delta + 1)
                shape[idx] = max(1, shape[idx])  # Ensure positive

        return torch.randn(*shape, dtype=tensor.dtype, device=tensor.device)

    def _vary_dtype(self, tensor: torch.Tensor) -> torch.Tensor:
        """Vary tensor dtype (float32 <-> float16)"""
        if tensor.dtype == torch.float32:
            return tensor.half()
        elif tensor.dtype == torch.float16:
            return tensor.float()
        return tensor

    # ============================================================================
    # Extension 5: Calibrated Timing (MEDIUM COMPLEXITY - MEDIUM VALUE)
    # ============================================================================

    def speedup_metric(
        self,
        triton_code: str,
        pytorch_ref_code: str,
        test_inputs: list,
        num_warmup: int = 10,
        num_runs: int = 100
    ) -> float:
        """
        Override base speedup_metric to add calibrated timing

        If enable_calibrated_timing=False, falls back to base implementation
        If enable_calibrated_timing=True, uses robust timing with:
        - More warmup runs
        - Explicit device synchronization
        - Multiple trials with trimmed mean
        - CUDA events for precise timing
        """
        if not self.config.enable_calibrated_timing:
            # Fall back to base implementation
            return super().speedup_metric(triton_code, pytorch_ref_code, test_inputs, num_warmup, num_runs)

        # Use config values
        num_warmup = self.config.timing_num_warmup
        num_runs = self.config.timing_num_trials

        try:
            # Warmup phase
            for _ in range(num_warmup):
                self._execute_pytorch(pytorch_ref_code, test_inputs)
                self._execute_triton(triton_code, test_inputs)

            # Benchmark PyTorch with calibrated timing
            pytorch_times = []
            for _ in range(num_runs):
                if self.config.timing_use_events:
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)

                    start_event.record()
                    self._execute_pytorch(pytorch_ref_code, test_inputs)
                    end_event.record()
                    torch.cuda.synchronize()

                    pytorch_times.append(start_event.elapsed_time(end_event) / 1000.0)  # ms to s
                else:
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    self._execute_pytorch(pytorch_ref_code, test_inputs)
                    torch.cuda.synchronize()
                    pytorch_times.append(time.perf_counter() - start)

            # Benchmark Triton with calibrated timing
            triton_times = []
            for _ in range(num_runs):
                if self.config.timing_use_events:
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)

                    start_event.record()
                    self._execute_triton(triton_code, test_inputs)
                    end_event.record()
                    torch.cuda.synchronize()

                    triton_times.append(start_event.elapsed_time(end_event) / 1000.0)
                else:
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    self._execute_triton(triton_code, test_inputs)
                    torch.cuda.synchronize()
                    triton_times.append(time.perf_counter() - start)

            # Compute trimmed means
            pytorch_time = self._trimmed_mean(pytorch_times, self.config.timing_trim_percent)
            triton_time = self._trimmed_mean(triton_times, self.config.timing_trim_percent)

            return pytorch_time / triton_time if triton_time > 0 else 0.0

        except Exception as e:
            print(f"Calibrated timing error: {e}")
            return 0.0

    def _trimmed_mean(self, values: List[float], trim_percent: float) -> float:
        """Compute mean after trimming outliers"""
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        trim_count = int(n * trim_percent)

        if trim_count > 0:
            trimmed = sorted_vals[trim_count:-trim_count]
        else:
            trimmed = sorted_vals

        return sum(trimmed) / len(trimmed) if trimmed else 0.0

    # ============================================================================
    # Extension 4: Hardened Sandboxing (MORE COMPLEX - HIGH VALUE)
    # ============================================================================

    def _execute_triton(self, code: str, inputs: list):
        """Override to add sandboxing when enabled"""
        if not self.config.enable_strict_sandbox:
            # Fall back to base implementation
            return super()._execute_triton(code, inputs)

        # Execute in restricted environment
        return self._execute_sandboxed(code, inputs)

    def _execute_sandboxed(self, code: str, inputs: list):
        """Execute code in sandboxed environment"""
        # Restricted builtins
        safe_builtins = {
            'abs', 'all', 'any', 'bool', 'dict', 'enumerate', 'filter',
            'float', 'int', 'isinstance', 'len', 'list', 'map', 'max',
            'min', 'range', 'sorted', 'str', 'sum', 'tuple', 'zip',
        }

        restricted_globals = {
            '__builtins__': {k: __builtins__[k] for k in safe_builtins if k in __builtins__},
            'torch': torch,
            'triton': self._get_restricted_triton(),
        }

        # Block file/network operations if configured
        if self.config.sandbox_no_file_access:
            restricted_globals['open'] = self._blocked_open
            restricted_globals['__builtins__']['open'] = self._blocked_open

        namespace = restricted_globals.copy()

        try:
            exec(code, namespace)
            model = namespace['ModelNew']()
            model.cuda().eval()
            with torch.no_grad():
                return model(*inputs)
        except Exception as e:
            print(f"Sandboxed execution failed: {e}")
            return None

    def _get_restricted_triton(self):
        """Return triton module with restricted functionality"""
        import triton
        # Could add additional restrictions here
        return triton

    def _blocked_open(self, *args, **kwargs):
        """Block file operations"""
        raise PermissionError("File operations not allowed in sandbox")
