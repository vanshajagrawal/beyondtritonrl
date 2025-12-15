import ast
import re
import torch
import triton
from typing import Tuple, Optional
import subprocess
import tempfile
import time

class TritonVerifier:
    """Robust verification system for Triton code"""

    @staticmethod
    def syntax_check(code: str) -> bool:
        """Check if code is syntactically valid Triton code"""
        try:
            ast.parse(code)
            if "@triton.jit" not in code and "triton.jit" not in code:
                return False
            return True
        except SyntaxError:
            return False

    @staticmethod
    def functionality_check(code: str, task_description: str = "") -> bool:
        """Check if code implements valid Triton kernel functionality"""
        # Rule-based checks
        if not TritonVerifier._has_triton_kernel_call(code):
            return False

        if TritonVerifier._uses_pytorch_modules(code):
            return False

        if TritonVerifier._is_trivial_kernel(code):
            return False

        # LLM-based semantic check (placeholder - would use actual LLM)
        # semantic_valid = TritonVerifier._llm_semantic_check(code, task_description)

        return True

    @staticmethod
    def _has_triton_kernel_call(code: str) -> bool:
        """Check if Triton kernel is actually called"""
        # Look for kernel[grid](...) pattern
        kernel_call_pattern = r'\w+\[grid\]\([^)]*\)'
        return bool(re.search(kernel_call_pattern, code))

    @staticmethod
    def _uses_pytorch_modules(code: str) -> bool:
        """Detect use of high-level PyTorch modules"""
        forbidden_patterns = [
            r'torch\.nn\.',
            r'nn\.Conv',
            r'nn\.Linear',
            r'nn\.Module\(\)',
            r'torch\.matmul',
            r'F\.conv',
            r'F\.linear',
        ]

        for pattern in forbidden_patterns:
            if re.search(pattern, code):
                return True
        return False

    @staticmethod
    def _is_trivial_kernel(code: str) -> bool:
        """Check if kernel is trivial (just copying data)"""
        # Look for kernels that only load and store without computation
        triton_kernel_pattern = r'@triton\.jit\s+def\s+\w+\([^)]*\):(.*?)(?=\n@|\ndef\s|\Z)'
        kernels = re.findall(triton_kernel_pattern, code, re.DOTALL)

        for kernel in kernels:
            # Check if kernel has meaningful operations
            has_computation = any(op in kernel for op in ['+', '-', '*', '/', 'tl.dot', 'tl.sum'])
            if not has_computation:
                return True

        return False

    @staticmethod
    def compilation_check(code: str) -> bool:
        """Check if Triton code compiles"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name

            result = subprocess.run(
                ['python', '-c', f'exec(open("{temp_file}").read())'],
                capture_output=True,
                timeout=10,
            )

            return result.returncode == 0
        except Exception:
            return False

    @staticmethod
    def correctness_check(
        triton_code: str,
        pytorch_ref_code: str,
        test_inputs: list,
        tolerance: float = 1e-5
    ) -> bool:
        """Check if Triton code produces correct outputs"""
        try:
            # Execute PyTorch reference
            pytorch_outputs = TritonVerifier._execute_pytorch(pytorch_ref_code, test_inputs)

            # Execute Triton code
            triton_outputs = TritonVerifier._execute_triton(triton_code, test_inputs)

            # Compare outputs
            if pytorch_outputs is None or triton_outputs is None:
                return False

            return torch.allclose(triton_outputs, pytorch_outputs, atol=tolerance)

        except Exception as e:
            return False

    @staticmethod
    def speedup_metric(
        triton_code: str,
        pytorch_ref_code: str,
        test_inputs: list,
        num_warmup: int = 10,
        num_runs: int = 100
    ) -> float:
        """Measure speedup of Triton code vs PyTorch reference"""
        try:
            # Warmup
            for _ in range(num_warmup):
                TritonVerifier._execute_pytorch(pytorch_ref_code, test_inputs)
                TritonVerifier._execute_triton(triton_code, test_inputs)

            # Benchmark PyTorch
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(num_runs):
                TritonVerifier._execute_pytorch(pytorch_ref_code, test_inputs)
            torch.cuda.synchronize()
            pytorch_time = time.perf_counter() - start

            # Benchmark Triton
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(num_runs):
                TritonVerifier._execute_triton(triton_code, test_inputs)
            torch.cuda.synchronize()
            triton_time = time.perf_counter() - start

            return pytorch_time / triton_time

        except Exception:
            return 0.0

    @staticmethod
    def _execute_pytorch(code: str, inputs: list):
        """Execute PyTorch reference code"""
        namespace = {}
        exec(code, namespace)
        model = namespace['Model']()
        model.cuda().eval()
        with torch.no_grad():
            return model(*inputs)

    @staticmethod
    def _execute_triton(code: str, inputs: list):
        """Execute Triton code"""
        namespace = {}
        exec(code, namespace)
        model = namespace['ModelNew']()
        model.cuda().eval()
        with torch.no_grad():
            return model(*inputs)

    @staticmethod
    def compute_reward(
        code: str,
        pytorch_ref: str,
        test_inputs: list,
        reward_type: str = "plan"
    ) -> float:
        """Compute hierarchical reward for plan or code tokens"""
        syntax = float(TritonVerifier.syntax_check(code))
        func = float(TritonVerifier.functionality_check(code))

        if syntax == 0 or func == 0:
            return 0.0

        if reward_type == "plan":
            # Plan tokens: syntax × func × speedup
            speedup = TritonVerifier.speedup_metric(code, pytorch_ref, test_inputs)
            return syntax * func * speedup

        elif reward_type == "code":
            # Code tokens: syntax × func × correct
            correct = float(TritonVerifier.correctness_check(code, pytorch_ref, test_inputs))
            return syntax * func * correct

        else:
            raise ValueError(f"Unknown reward type: {reward_type}")

    @staticmethod
    def _safe_execute_code(code: str, test_inputs: list, timeout: int = 30) -> tuple:
        """
        Safely execute code with timeout protection and resource limits.

        Returns:
            tuple: (success: bool, result: any, execution_time: float, error: str)
        """
        import signal
        from contextlib import contextmanager

        @contextmanager
        def timeout_context(seconds):
            def timeout_handler(signum, frame):
                raise TimeoutError("Code execution timed out")
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)
            try:
                yield
            finally:
                signal.alarm(0)

        start_time = time.time()
        try:
            with timeout_context(timeout):
                # Limit memory usage
                import resource
                resource.setrlimit(resource.RLIMIT_AS, (1 * 1024 * 1024 * 1024, 1 * 1024 * 1024 * 1024))  # 1GB limit

                namespace = {'torch': torch, 'triton': triton}
                exec(code, namespace)

                # Execute with test inputs if available
                if test_inputs and 'Model' in namespace:
                    model = namespace['Model']()
                    if hasattr(model, 'cuda'):
                        model = model.cuda()
                    model.eval()

                    with torch.no_grad():
                        result = model(*test_inputs)
                    return True, result, time.time() - start_time, ""

        except MemoryError:
            return False, None, time.time() - start_time, "Memory limit exceeded"
        except TimeoutError:
            return False, None, time.time() - start_time, "Execution timeout"
        except Exception as e:
            return False, None, time.time() - start_time, str(e)

    @staticmethod
    def _validate_kernel_structure(code: str) -> dict:
        """
        Validate Triton kernel structure and extract metadata.

        Returns:
            dict: Validation results with metadata
        """
        validation_result = {
            'has_kernel_decorator': False,
            'has_kernel_function': False,
            'has_grid_specification': False,
            'uses_triton_operations': False,
            'parameter_count': 0,
            'complexity_score': 0
        }

        # Check for @triton.jit decorator
        if '@triton.jit' in code:
            validation_result['has_kernel_decorator'] = True

        # Check for kernel function definition
        import ast
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    validation_result['has_kernel_function'] = True
                    validation_result['parameter_count'] = len(node.args.args)
                    # Simple complexity heuristic
                    validation_result['complexity_score'] = len(list(ast.walk(node)))
        except:
            pass

        # Check for grid specification
        if 'grid=' in code or 'grid =' in code:
            validation_result['has_grid_specification'] = True

        # Check for Triton operations
        triton_ops = ['tl.load', 'tl.store', 'tl.dot', 'tl.sum', 'tl.arange']
        validation_result['uses_triton_operations'] = any(op in code for op in triton_ops)

        return validation_result
