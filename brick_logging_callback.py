"""
BrickLoggingCallback — wraps an EvalCallback and logs bricks_cleared alongside
the raw eval score. Requires BrickCountingVecWrapper on the eval env.
"""
import os
from datetime import datetime
from stable_baselines3.common.callbacks import BaseCallback


class BrickLoggingCallback(BaseCallback):
    def __init__(self, log_path, verbose=1):
        super().__init__(verbose)
        self.log_path = log_path
        self.log_file = None
        self._header_written = False

    def _init_callback(self):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self.log_file = open(self.log_path, "a")

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self):
        if self.log_file and not self.log_file.closed:
            self.log_file.close()

    def log_bricks(self, raw_score, bricks_cleared):
        """Call this after each eval to log the brick count."""
        if self.log_file is None:
            return
        if not self._header_written:
            self.log_file.write("timestamp,raw_score,bricks_cleared\n")
            self._header_written = True
        self.log_file.write(
            f"{datetime.now().isoformat()},{raw_score:.1f},{bricks_cleared}\n"
        )
        self.log_file.flush()
        if self.verbose:
            print(f"  Bricks cleared: {bricks_cleared}  |  "
                  f"Raw score: {raw_score:.1f}  |  "
                  f"Avg points/brick: {raw_score/bricks_cleared:.1f}x"
                  if bricks_cleared > 0 else
                  f"  Bricks cleared: {bricks_cleared}  |  Raw score: {raw_score:.1f}")