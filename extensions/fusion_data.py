"""
Extension 1: Fusion-Centric Data Generation

Programmatically generates fused operation tasks (Conv->BN->ReLU, GEMM->bias->act, etc.)
using torch.fx tracing and templates.

Usage:
    generator = FusionDataGenerator(config)
    fusion_tasks = generator.generate_fusion_tasks()
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple
import itertools

class FusionDataGenerator:
    """Generate fusion-centric training data"""

    def __init__(self, ext_config):
        self.config = ext_config
        self.patterns = ext_config.fusion_patterns

    def generate_fusion_tasks(self) -> List[Dict]:
        """Generate all fusion tasks across patterns, shapes, dtypes"""
        tasks = []

        for pattern in self.patterns:
            generator = getattr(self, f'_generate_{pattern}', None)
            if generator is None:
                print(f"Warning: Pattern {pattern} not implemented, skipping")
                continue

            pattern_tasks = generator()
            tasks.extend(pattern_tasks)

        print(f"Generated {len(tasks)} fusion tasks")
        return tasks

    def _generate_conv_bn_relu(self) -> List[Dict]:
        """Conv2d -> BatchNorm2d -> ReLU fusion"""
        tasks = []

        # Shape variations
        shapes = [
            (1, 64, 56, 56),   # ResNet-style
            (4, 128, 28, 28),  # Larger batch
            (1, 256, 14, 14),  # Deeper layers
        ][:self.config.fusion_num_shapes]

        # Dtype variations
        dtypes = [torch.float32, torch.float16][:self.config.fusion_num_dtypes]

        for (B, C, H, W), dtype in itertools.product(shapes, dtypes):
            pytorch_code = self._gen_conv_bn_relu_pytorch(B, C, H, W, dtype)
            instruction = self._gen_fusion_instruction(pytorch_code, "Conv->BN->ReLU")

            tasks.append({
                "id": f"conv_bn_relu_{B}_{C}_{H}_{W}_{dtype}",
                "pattern": "conv_bn_relu",
                "pytorch_code": pytorch_code,
                "instruction": instruction,
                "difficulty": 2,  # Level 2 (fusion)
                "shapes": (B, C, H, W),
                "dtype": str(dtype),
            })

        return tasks

    def _generate_gemm_bias_act(self) -> List[Dict]:
        """Matrix multiply -> Bias add -> Activation fusion"""
        tasks = []

        # Shape variations (M, N, K)
        shapes = [
            (128, 128, 128),
            (256, 256, 256),
            (512, 1024, 512),
        ][:self.config.fusion_num_shapes]

        # Activation variations
        activations = ['relu', 'gelu']

        dtypes = [torch.float32, torch.float16][:self.config.fusion_num_dtypes]

        for (M, N, K), act, dtype in itertools.product(shapes, activations, dtypes):
            pytorch_code = self._gen_gemm_bias_act_pytorch(M, N, K, act, dtype)
            instruction = self._gen_fusion_instruction(pytorch_code, f"GEMM->Bias->{act.upper()}")

            tasks.append({
                "id": f"gemm_bias_{act}_{M}_{N}_{K}_{dtype}",
                "pattern": f"gemm_bias_{act}",
                "pytorch_code": pytorch_code,
                "instruction": instruction,
                "difficulty": 2,
                "shapes": (M, N, K),
                "activation": act,
                "dtype": str(dtype),
            })

        return tasks

    def _generate_ln_gelu(self) -> List[Dict]:
        """LayerNorm -> GELU fusion"""
        tasks = []

        # Shape variations
        shapes = [
            (4, 128, 768),    # Transformer-style
            (8, 256, 1024),
            (1, 512, 2048),
        ][:self.config.fusion_num_shapes]

        dtypes = [torch.float32, torch.float16][:self.config.fusion_num_dtypes]

        for (B, S, D), dtype in itertools.product(shapes, dtypes):
            pytorch_code = self._gen_ln_gelu_pytorch(B, S, D, dtype)
            instruction = self._gen_fusion_instruction(pytorch_code, "LayerNorm->GELU")

            tasks.append({
                "id": f"ln_gelu_{B}_{S}_{D}_{dtype}",
                "pattern": "ln_gelu",
                "pytorch_code": pytorch_code,
                "instruction": instruction,
                "difficulty": 2,
                "shapes": (B, S, D),
                "dtype": str(dtype),
            })

        return tasks

    def _gen_conv_bn_relu_pytorch(self, B, C, H, W, dtype) -> str:
        """Generate PyTorch reference for Conv->BN->ReLU"""
        dtype_str = "torch.float32" if dtype == torch.float32 else "torch.float16"

        return f"""import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d({C}, {C}, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d({C})
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

def get_inputs():
    return [torch.randn({B}, {C}, {H}, {W}, dtype={dtype_str}).cuda()]

def get_init_inputs():
    return []
"""

    def _gen_gemm_bias_act_pytorch(self, M, N, K, act, dtype) -> str:
        """Generate PyTorch reference for GEMM->Bias->Activation"""
        dtype_str = "torch.float32" if dtype == torch.float32 else "torch.float16"
        act_fn = f"torch.nn.functional.{act}"

        return f"""import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn({N}, {K}, dtype={dtype_str}))
        self.bias = nn.Parameter(torch.randn({N}, dtype={dtype_str}))

    def forward(self, x):
        x = torch.matmul(x, self.weight.t())
        x = x + self.bias
        x = {act_fn}(x)
        return x

def get_inputs():
    return [torch.randn({M}, {K}, dtype={dtype_str}).cuda()]

def get_init_inputs():
    return []
"""

    def _gen_ln_gelu_pytorch(self, B, S, D, dtype) -> str:
        """Generate PyTorch reference for LayerNorm->GELU"""
        dtype_str = "torch.float32" if dtype == torch.float32 else "torch.float16"

        return f"""import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm({D})

    def forward(self, x):
        x = self.ln(x)
        x = F.gelu(x)
        return x

def get_inputs():
    return [torch.randn({B}, {S}, {D}, dtype={dtype_str}).cuda()]

def get_init_inputs():
    return []
"""

    def _gen_fusion_instruction(self, pytorch_code, fusion_name) -> str:
        """Generate instruction prompt for fusion task"""
        return f"""Your task is to write a custom Triton kernel to implement the {fusion_name} fusion pattern in a single, optimized kernel.

The reference PyTorch implementation is:
```python
{pytorch_code}
```

Implement a fused Triton kernel that combines these operations for maximum performance. Name your optimized model ModelNew. Output real, compilable code."""
