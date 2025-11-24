"""
Example configurations showing how to enable extensions individually or in combination.

Usage:
    from config_examples import get_config

    # Get config with only multi-input testing
    config = get_config('multi_input_only')

    # Get config with all extensions
    config = get_config('all_extensions')
"""

from config import DataConfig, SFTConfig, RLConfig, EvalConfig
from extensions.config import ExtensionConfig

def get_config(preset: str = 'baseline'):
    """
    Get configuration preset

    Available presets:
        - baseline: Original TritonRL (no extensions)
        - multi_input_only: Only multi-input testing (easiest)
        - fusion_data_only: Only fusion-centric data generation
        - curriculum_only: Only adaptive curriculum
        - calibrated_timing_only: Only calibrated timing
        - staged_eval_only: Only verification funnel
        - sandbox_only: Only hardened sandboxing
        - easy_combo: Multi-input + Curriculum + Staged eval
        - all_extensions: All extensions enabled
    """

    base_configs = {
        'data': DataConfig(),
        'sft': SFTConfig(),
        'rl': RLConfig(),
        'eval': EvalConfig(),
    }

    if preset == 'baseline':
        ext_config = ExtensionConfig(
            enable_fusion_data=False,
            enable_multi_input=False,
            enable_adaptive_curriculum=False,
            enable_strict_sandbox=False,
            enable_calibrated_timing=False,
            enable_staged_eval=False,
        )

    elif preset == 'multi_input_only':
        ext_config = ExtensionConfig(
            enable_fusion_data=False,
            enable_multi_input=True,  # ← ENABLED
            multi_input_num_tests=5,
            multi_input_shape_variations=True,
            multi_input_value_variations=True,
            enable_adaptive_curriculum=False,
            enable_strict_sandbox=False,
            enable_calibrated_timing=False,
            enable_staged_eval=False,
        )

    elif preset == 'fusion_data_only':
        ext_config = ExtensionConfig(
            enable_fusion_data=True,  # ← ENABLED
            fusion_patterns=['conv_bn_relu', 'gemm_bias_act', 'ln_gelu'],
            fusion_num_shapes=5,
            enable_multi_input=False,
            enable_adaptive_curriculum=False,
            enable_strict_sandbox=False,
            enable_calibrated_timing=False,
            enable_staged_eval=False,
        )

    elif preset == 'curriculum_only':
        ext_config = ExtensionConfig(
            enable_fusion_data=False,
            enable_multi_input=False,
            enable_adaptive_curriculum=True,  # ← ENABLED
            curriculum_start_p=0.1,
            curriculum_end_p=0.5,
            curriculum_trigger_threshold=0.4,
            enable_strict_sandbox=False,
            enable_calibrated_timing=False,
            enable_staged_eval=False,
        )

    elif preset == 'calibrated_timing_only':
        ext_config = ExtensionConfig(
            enable_fusion_data=False,
            enable_multi_input=False,
            enable_adaptive_curriculum=False,
            enable_strict_sandbox=False,
            enable_calibrated_timing=True,  # ← ENABLED
            timing_num_warmup=20,
            timing_num_trials=100,
            timing_trim_percent=0.1,
            timing_use_events=True,
            enable_staged_eval=False,
        )

    elif preset == 'staged_eval_only':
        ext_config = ExtensionConfig(
            enable_fusion_data=False,
            enable_multi_input=False,
            enable_adaptive_curriculum=False,
            enable_strict_sandbox=False,
            enable_calibrated_timing=False,
            enable_staged_eval=True,  # ← ENABLED
            staged_skip_timing_on_failure=True,
            staged_tiny_batch_first=True,
        )

    elif preset == 'sandbox_only':
        ext_config = ExtensionConfig(
            enable_fusion_data=False,
            enable_multi_input=False,
            enable_adaptive_curriculum=False,
            enable_strict_sandbox=True,  # ← ENABLED
            sandbox_no_file_access=True,
            sandbox_no_network=True,
            enable_calibrated_timing=False,
            enable_staged_eval=False,
        )

    elif preset == 'easy_combo':
        # Easiest extensions that give most value
        ext_config = ExtensionConfig(
            enable_fusion_data=False,
            enable_multi_input=True,  # ← Easy + High value
            multi_input_num_tests=5,
            enable_adaptive_curriculum=True,  # ← Easy + High value
            curriculum_start_p=0.1,
            curriculum_end_p=0.5,
            enable_strict_sandbox=False,
            enable_calibrated_timing=False,
            enable_staged_eval=True,  # ← Easy + High efficiency
            staged_skip_timing_on_failure=True,
        )

    elif preset == 'all_extensions':
        ext_config = ExtensionConfig(
            enable_fusion_data=True,
            fusion_patterns=['conv_bn_relu', 'gemm_bias_act', 'ln_gelu'],
            enable_multi_input=True,
            multi_input_num_tests=5,
            enable_adaptive_curriculum=True,
            curriculum_start_p=0.1,
            curriculum_end_p=0.5,
            enable_strict_sandbox=True,
            sandbox_no_file_access=True,
            enable_calibrated_timing=True,
            timing_num_warmup=20,
            timing_num_trials=100,
            enable_staged_eval=True,
            staged_skip_timing_on_failure=True,
        )

    else:
        raise ValueError(f"Unknown preset: {preset}")

    base_configs['extensions'] = ext_config

    print(f"\nLoaded config preset: {preset}")
    print(f"Enabled extensions: {ext_config.get_enabled_extensions()}\n")

    return base_configs


# Quick access functions for common use cases
def get_baseline_config():
    """Original TritonRL with no extensions"""
    return get_config('baseline')

def get_easiest_config():
    """Multi-input testing only (easiest to deploy)"""
    return get_config('multi_input_only')

def get_recommended_config():
    """Recommended easy combo for best value/effort ratio"""
    return get_config('easy_combo')

def get_full_config():
    """All extensions enabled"""
    return get_config('all_extensions')
