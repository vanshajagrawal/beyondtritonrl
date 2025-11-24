"""
Extension 6: Verification Funnel (Staged Evaluation)

Implements a multi-stage verification pipeline that prunes bad candidates early
to save compute. Stages: AST -> Compile -> Tiny-run -> Full-run -> Timing.
"""

from typing import Tuple, Optional, Dict
import torch

class StagedEvaluator:
    """Multi-stage verification funnel to optimize evaluation throughput"""

    def __init__(self, verifier, ext_config):
        """
        Args:
            verifier: Base or hardened verifier instance
            ext_config: Extension configuration
        """
        self.verifier = verifier
        self.config = ext_config

    def evaluate_with_funnel(
        self,
        code: str,
        pytorch_ref: str,
        test_inputs: list
    ) -> Dict[str, any]:
        """
        Run staged evaluation, pruning at each failure

        Returns dict with:
            - passed_stage: 'ast', 'compile', 'tiny_run', 'full_run', 'timing', or 'none'
            - rewards: dict of plan_reward, code_reward
            - speedup: float (0.0 if failed before timing)
        """
        if not self.config.enable_staged_eval:
            # Fall back to single-stage evaluation
            return self._evaluate_single_stage(code, pytorch_ref, test_inputs)

        result = {
            'passed_stage': 'none',
            'plan_reward': 0.0,
            'code_reward': 0.0,
            'speedup': 0.0,
        }

        # Stage 1: AST / Syntax / Functionality
        syntax = self.verifier.syntax_check(code)
        if not syntax:
            return result

        func = self.verifier.functionality_check(code)
        if not func:
            return result

        result['passed_stage'] = 'ast'

        # Stage 2: Compilation
        compiled = self.verifier.compilation_check(code)
        if not compiled:
            return result

        result['passed_stage'] = 'compile'

        # Stage 3: Tiny-run correctness (if enabled)
        if self.config.staged_tiny_batch_first:
            tiny_correct = self._tiny_batch_correctness(code, pytorch_ref, test_inputs)
            if not tiny_correct:
                return result
            result['passed_stage'] = 'tiny_run'

        # Stage 4: Full-run correctness
        correct = self.verifier.correctness_check(code, pytorch_ref, test_inputs)
        if not correct:
            return result

        result['passed_stage'] = 'full_run'
        result['code_reward'] = 1.0  # Correctness achieved

        # Stage 5: Timing (expensive, only for correct kernels)
        if not self.config.staged_skip_timing_on_failure:
            speedup = self.verifier.speedup_metric(code, pytorch_ref, test_inputs)
            result['speedup'] = speedup
            result['plan_reward'] = speedup
            result['passed_stage'] = 'timing'
        else:
            # Only time if we passed full correctness
            speedup = self.verifier.speedup_metric(code, pytorch_ref, test_inputs)
            result['speedup'] = speedup
            result['plan_reward'] = speedup
            result['passed_stage'] = 'timing'

        return result

    def _tiny_batch_correctness(
        self,
        code: str,
        pytorch_ref: str,
        test_inputs: list
    ) -> bool:
        """
        Test correctness on a tiny batch (reduced shapes) to fail fast

        This catches many bugs without expensive full-batch execution
        """
        # Create tiny versions of inputs
        tiny_inputs = []
        for inp in test_inputs:
            if isinstance(inp, torch.Tensor):
                # Reduce to minimal size
                shape = list(inp.shape)
                tiny_shape = [min(s, 4) for s in shape]  # Cap at 4 per dimension
                tiny_inp = torch.randn(*tiny_shape, dtype=inp.dtype, device=inp.device)
                tiny_inputs.append(tiny_inp)
            else:
                tiny_inputs.append(inp)

        try:
            pytorch_out = self.verifier._execute_pytorch(pytorch_ref, tiny_inputs)
            triton_out = self.verifier._execute_triton(code, tiny_inputs)

            if pytorch_out is None or triton_out is None:
                return False

            return torch.allclose(triton_out, pytorch_out, atol=1e-5)
        except Exception:
            return False

    def _evaluate_single_stage(
        self,
        code: str,
        pytorch_ref: str,
        test_inputs: list
    ) -> Dict[str, any]:
        """Fallback to non-staged evaluation"""
        syntax = self.verifier.syntax_check(code)
        func = self.verifier.functionality_check(code)

        if not (syntax and func):
            return {
                'passed_stage': 'none',
                'plan_reward': 0.0,
                'code_reward': 0.0,
                'speedup': 0.0,
            }

        correct = self.verifier.correctness_check(code, pytorch_ref, test_inputs)
        speedup = self.verifier.speedup_metric(code, pytorch_ref, test_inputs) if correct else 0.0

        return {
            'passed_stage': 'timing' if correct else 'compile',
            'plan_reward': syntax * func * speedup,
            'code_reward': syntax * func * correct,
            'speedup': speedup,
        }
