from dataclasses import dataclass

@dataclass
class ExtensionConfig:
    """Feature flags for modular extensions - enable/disable individually"""

    # Extension 1: Fusion-centric data generation
    enable_fusion_data: bool = False
    fusion_patterns: list = None  # ['conv_bn_relu', 'gemm_bias_act', 'ln_gelu']
    fusion_num_shapes: int = 5
    fusion_num_dtypes: int = 2

    # Extension 2: Multi-input testing (easiest to implement)
    enable_multi_input: bool = False
    multi_input_num_tests: int = 5
    multi_input_shape_variations: bool = True
    multi_input_value_variations: bool = True
    multi_input_dtype_variations: bool = False

    # Extension 3: Adaptive curriculum
    enable_adaptive_curriculum: bool = False
    curriculum_start_p: float = 0.1  # Start with 10% L2
    curriculum_end_p: float = 0.5    # End with 50% L2
    curriculum_trigger_threshold: float = 0.4  # L1 correctness threshold to increase p

    # Extension 4: Hardened sandboxing
    enable_strict_sandbox: bool = False
    sandbox_no_file_access: bool = True
    sandbox_no_network: bool = True
    sandbox_syscall_filter: bool = False  # Requires seccomp

    # Extension 5: Calibrated timing
    enable_calibrated_timing: bool = False
    timing_num_warmup: int = 10
    timing_num_trials: int = 100
    timing_trim_percent: float = 0.1  # Trim top/bottom 10%
    timing_use_events: bool = True

    # Extension 6: Verification funnel (staged evaluation)
    enable_staged_eval: bool = False
    staged_skip_timing_on_failure: bool = True
    staged_tiny_batch_first: bool = True

    def __post_init__(self):
        if self.fusion_patterns is None:
            self.fusion_patterns = ['conv_bn_relu', 'gemm_bias_act', 'ln_gelu']

    def get_enabled_extensions(self):
        """Return list of enabled extension names"""
        extensions = []
        if self.enable_fusion_data:
            extensions.append("fusion_data")
        if self.enable_multi_input:
            extensions.append("multi_input")
        if self.enable_adaptive_curriculum:
            extensions.append("adaptive_curriculum")
        if self.enable_strict_sandbox:
            extensions.append("strict_sandbox")
        if self.enable_calibrated_timing:
            extensions.append("calibrated_timing")
        if self.enable_staged_eval:
            extensions.append("staged_eval")
        return extensions
