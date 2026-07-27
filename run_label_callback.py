"""
RunLabelCallback — stamps the run name into console output so you can tell
which terminal is which when multiple PPO runs are going simultaneously.

SB3's default verbose output is anonymous — just step counts and rewards.
This prints a banner at training start and a [RUN_NAME] stamp periodically.
"""
from stable_baselines3.common.callbacks import BaseCallback


class RunLabelCallback(BaseCallback):
    """Print [RUN_NAME] at training start and periodically during training.

    Usage:
        from run_label_callback import RunLabelCallback
        label_callback = RunLabelCallback(RUN_NAME)
        callbacks = CallbackList([eval_callback, checkpoint_callback,
                                  memorization_callback, label_callback])
    """
    def __init__(self, run_name, print_freq=1_000_000):
        super().__init__()
        self.run_name = run_name
        self.print_freq = print_freq
        self._last_print = 0

    def _on_training_start(self):
        print(f"\n{'=' * 60}")
        print(f"  {self.run_name}")
        print(f"{'=' * 60}\n")

    def _on_step(self):
        if self.num_timesteps - self._last_print >= self.print_freq:
            print(f"[{self.run_name}] step {self.num_timesteps:,}")
            self._last_print = self.num_timesteps
        return True
