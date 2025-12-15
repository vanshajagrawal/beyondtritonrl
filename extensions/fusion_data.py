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

    def _generate_attention_fusion(self) -> List[Dict]:
        """Generate attention mechanism fusion tasks (QKV projection + softmax + dropout)"""
        tasks = []

        # Attention configurations
        configs = [
            {'seq_len': 512, 'hidden': 768, 'heads': 12},   # BERT-base
            {'seq_len': 1024, 'hidden': 1024, 'heads': 16},  # Larger model
            {'seq_len': 2048, 'hidden': 1280, 'heads': 20},  # Very large
        ][:self.config.fusion_num_shapes]

        for config in configs:
            seq_len, hidden, heads = config['seq_len'], config['hidden'], config['heads']

            # Generate PyTorch code
            pytorch_code = f"""
import torch
import torch.nn.functional as F

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv_proj = torch.nn.Linear({hidden}, 3 * {hidden})
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x):
        B, S, D = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for attention
        q = q.view(B, S, {heads}, D // {heads}).transpose(1, 2)
        k = k.view(B, S, {heads}, D // {heads}).transpose(1, 2)
        v = v.view(B, S, {heads}, D // {heads}).transpose(1, 2)

        # Attention computation
        scale = (D // {heads}) ** -0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Output projection
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(B, S, D)
        return output
"""

            task = {
                'id': f'attention_{seq_len}_{hidden}_{heads}',
                'pytorch_code': pytorch_code.strip(),
                'difficulty': 2,  # Fusion level
                'task_type': 'attention_fusion',
                'description': f'Attention mechanism fusion (seq_len={seq_len}, hidden={hidden}, heads={heads})'
            }
            tasks.append(task)

        return tasks

    def _generate_transformer_block(self) -> List[Dict]:
        """Generate complete transformer block fusion tasks"""
        tasks = []

        configs = [
            {'seq_len': 512, 'hidden': 768, 'ffn_dim': 3072},  # BERT-base
            {'seq_len': 1024, 'hidden': 1024, 'ffn_dim': 4096}, # Larger
        ][:self.config.fusion_num_shapes]

        for config in configs:
            seq_len, hidden, ffn_dim = config['seq_len'], config['hidden'], config['ffn_dim']

            pytorch_code = f"""
import torch
import torch.nn.functional as F

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_norm = torch.nn.LayerNorm({hidden})
        self.ffn_norm = torch.nn.LayerNorm({hidden})

        # Attention components
        self.qkv_proj = torch.nn.Linear({hidden}, 3 * {hidden})
        self.out_proj = torch.nn.Linear({hidden}, {hidden})

        # FFN components
        self.ffn1 = torch.nn.Linear({hidden}, {ffn_dim})
        self.ffn2 = torch.nn.Linear({ffn_dim}, {hidden})

        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x):
        # Multi-head attention with residual
        attn_input = self.attn_norm(x)
        qkv = self.qkv_proj(attn_input)
        q, k, v = qkv.chunk(3, dim=-1)

        # Simplified attention (just QK^T scaling for demo)
        attn_output = torch.matmul(q, k.transpose(-2, -1)) / ({hidden} ** 0.5)
        attn_output = F.softmax(attn_output, dim=-1)
        attn_output = torch.matmul(attn_output, v)
        attn_output = self.out_proj(attn_output)
        x = x + self.dropout(attn_output)

        # Feed-forward network with residual
        ffn_input = self.ffn_norm(x)
        ffn_output = self.ffn1(ffn_input)
        ffn_output = F.gelu(ffn_output)
        ffn_output = self.ffn2(ffn_output)
        x = x + self.dropout(ffn_output)

        return x
"""

            task = {
                'id': f'transformer_block_{seq_len}_{hidden}',
                'pytorch_code': pytorch_code.strip(),
                'difficulty': 3,  # Advanced fusion
                'task_type': 'transformer_block',
                'description': f'Complete transformer block (seq_len={seq_len}, hidden={hidden})'
            }
            tasks.append(task)

        return tasks

    def _validate_fusion_patterns(self) -> Dict[str, bool]:
        """Validate that all configured fusion patterns are implemented"""
        validation_results = {}

        for pattern in self.config.fusion_patterns:
            method_name = f'_generate_{pattern}'
            has_method = hasattr(self, method_name)
            validation_results[pattern] = has_method

            if not has_method:
                print(f"Warning: Fusion pattern '{pattern}' is not implemented")

        return validation_results

    def get_fusion_statistics(self) -> Dict[str, int]:
        """Get statistics about generated fusion patterns"""
        stats = {
            'total_patterns': len(self.config.fusion_patterns),
            'implemented_patterns': 0,
            'estimated_tasks': 0
        }

        validation = self._validate_fusion_patterns()
        stats['implemented_patterns'] = sum(validation.values())

        # Estimate total tasks
        for pattern in self.config.fusion_patterns:
            if validation.get(pattern, False):
                # Rough estimate: patterns × shapes × dtypes
                stats['estimated_tasks'] += self.config.fusion_num_shapes * len(self.config.fusion_dtypes)

        return stats
