"""
BrickCountingVecWrapper — wraps a VecEnv and counts clipped-reward brick hits
per episode. Augments episode info with 'bricks_cleared' so EvalCallback and
watch scripts can log brick counts alongside raw game scores.

Only counts positive (non-zero) rewards to avoid double-counting frame-skip
accumulation. Assumes ClipRewardEnv is active (rewards are {-1, 0, 1}).

Only adds bricks_cleared to an existing 'episode' info dict — never creates
a new one. This preserves 'r' and 'l' from the underlying env (ALE) so that
SB3's ep_rew_mean and ep_len_mean logging still works. At life-loss events
(EpisodicLifeEnv), the ALE hasn't terminated its internal episode, so no
episode dict exists yet — those events are skipped here and bricks are
tracked via episode_brick_buffer for BrickRolloutCallback instead.
"""
import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.common.callbacks import BaseCallback


class BrickCountingVecWrapper(VecEnvWrapper):
    def __init__(self, venv):
        super().__init__(venv)
        self.brick_counts = np.zeros(self.num_envs, dtype=int)
        self.episode_brick_buffer = []  # completed-episode brick totals for logging

    def reset(self):
        self.brick_counts = np.zeros(self.num_envs, dtype=int)
        return self.venv.reset()

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        for i in range(self.num_envs):
            r = float(rewards[i])
            if r > 0:
                self.brick_counts[i] += 1
            if dones[i]:
                lives = infos[i].get("lives", -1)
                if lives == 0:
                    self.episode_brick_buffer.append(int(self.brick_counts[i]))
                    if "episode" in infos[i]:
                        infos[i]["episode"]["bricks_cleared"] = int(self.brick_counts[i])
                    self.brick_counts[i] = 0
        return obs, rewards, dones, infos


class BrickRolloutCallback(BaseCallback):
    """Logs avg bricks_cleared alongside rollout ep_rew_mean each iteration.

    Walks the training env chain to find BrickCountingVecWrapper, reads the
    completed-episode brick buffer, logs the average, and clears the buffer.
    Requires BrickCountingVecWrapper in the training env chain.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        env = self.training_env
        while hasattr(env, 'venv'):
            if isinstance(env, BrickCountingVecWrapper):
                buf = env.episode_brick_buffer
                if buf:
                    avg_bricks = float(np.mean(buf))
                    self.logger.record("rollout/ep_bricks", avg_bricks)
                    buf.clear()
                break
            env = env.venv
        return True