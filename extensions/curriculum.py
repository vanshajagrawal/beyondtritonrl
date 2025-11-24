"""
Extension 3: Adaptive Curriculum Learning

Dynamically adjusts L1/L2 sampling probability based on training progress.
Starts with mostly L1 tasks, gradually shifts to L2 as correctness stabilizes.
"""

import numpy as np
from typing import Dict, List

class AdaptiveCurriculum:
    """Adaptive curriculum scheduler for L1/L2 task mixing"""

    def __init__(self, ext_config):
        self.config = ext_config
        self.start_p = ext_config.curriculum_start_p
        self.end_p = ext_config.curriculum_end_p
        self.trigger_threshold = ext_config.curriculum_trigger_threshold

        self.current_p = self.start_p
        self.step_count = 0
        self.l1_correctness_history = []

    def sample_tasks(
        self,
        level1_tasks: List[Dict],
        level2_tasks: List[Dict],
        num_samples: int
    ) -> List[Dict]:
        """
        Sample tasks according to current curriculum schedule

        Args:
            level1_tasks: List of Level 1 task dicts
            level2_tasks: List of Level 2 task dicts
            num_samples: Total number of tasks to sample

        Returns:
            Mixed list of tasks
        """
        if not self.config.enable_adaptive_curriculum:
            # Static mixing: use original data_mix config
            # This would be handled by the caller
            raise NotImplementedError("Use base sampling when curriculum disabled")

        num_l2 = int(num_samples * self.current_p)
        num_l1 = num_samples - num_l2

        sampled_l1 = np.random.choice(level1_tasks, size=min(num_l1, len(level1_tasks)), replace=False).tolist()
        sampled_l2 = np.random.choice(level2_tasks, size=min(num_l2, len(level2_tasks)), replace=False).tolist()

        mixed = sampled_l1 + sampled_l2
        np.random.shuffle(mixed)

        return mixed

    def update_curriculum(self, metrics: Dict[str, float]):
        """
        Update curriculum schedule based on training metrics

        Args:
            metrics: Dict containing 'l1_correct', 'l2_correct', etc.
        """
        self.step_count += 1

        # Track L1 correctness
        l1_correct = metrics.get('l1_correct', 0.0)
        self.l1_correctness_history.append(l1_correct)

        # Check if we should increase L2 probability
        if len(self.l1_correctness_history) >= 5:
            recent_avg = np.mean(self.l1_correctness_history[-5:])

            if recent_avg >= self.trigger_threshold:
                # Gradually increase L2 sampling
                self.current_p = min(self.current_p + 0.05, self.end_p)
                print(f"Curriculum update: L2 probability increased to {self.current_p:.2f}")

        return self.current_p

    def get_current_p(self) -> float:
        """Get current L2 sampling probability"""
        return self.current_p

    def get_cosine_decay_alpha(self, current_step: int, total_steps: int, alpha_start: float = 0.1) -> float:
        """
        Compute plan weight alpha with cosine decay

        Args:
            current_step: Current training step
            total_steps: Total training steps
            alpha_start: Starting alpha value

        Returns:
            Decayed alpha value
        """
        if not self.config.enable_adaptive_curriculum:
            return alpha_start

        # Cosine decay from alpha_start to alpha_start/10
        progress = current_step / total_steps
        alpha_end = alpha_start / 10.0

        alpha = alpha_end + (alpha_start - alpha_end) * 0.5 * (1 + np.cos(np.pi * progress))
        return alpha

    def get_reward_beta(self, current_step: int, total_steps: int) -> float:
        """
        Gradually increase speed emphasis in reward over training

        Args:
            current_step: Current training step
            total_steps: Total training steps

        Returns:
            Beta value (correctness weight in reward)
        """
        if not self.config.enable_adaptive_curriculum:
            return 1.0  # All correctness, no speed (conservative)

        # Start with high correctness focus (beta=0.9), end with balanced (beta=0.5)
        progress = current_step / total_steps
        beta_start = 0.9
        beta_end = 0.5

        beta = beta_start - (beta_start - beta_end) * progress
        return beta
