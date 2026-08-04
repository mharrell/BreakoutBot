"""
AnnealingProximityRewardWrapper — proximity reward with decaying scale.

Same as ProximityRewardWrapper but scales the bonus by a schedule function
that decreases over training. This lets the proximity reward shape early
behavior then fade out, preventing the model from converging to a script
that maximizes the combined game+proximity objective.
"""
import numpy as np
import gymnasium as gym


class AnnealingProximityRewardWrapper(gym.Wrapper):
    """Proximity reward with decaying scale.

    Parameters
    ----------
    env : gym.Env
    scale_schedule : callable
        Function of progress_remaining (1.0 → 0.0) returning current scale.
        e.g. lambda p: 0.05 * p  linearly decays from 0.05 to 0.0.
    max_distance : float
    descend_threshold : int
    """

    PADDLE_X_ADDR = 72
    BALL_X_ADDR = 99
    BALL_Y_ADDR = 101

    def __init__(
        self,
        env,
        scale_schedule=None,
        max_distance=80.0,
        descend_threshold=100,
        seed=None,
    ):
        super().__init__(env)
        self.scale_schedule = scale_schedule or (lambda p: 0.05 * p)
        self.max_distance = float(max_distance)
        self.descend_threshold = int(descend_threshold)
        self._total_bonus = 0.0
        self._bonus_steps = 0
        self._total_steps = 0
        self._current_scale = 0.0
        # progress_remaining will be set externally by the training loop
        self.progress_remaining = 1.0

    def _get_ram(self, addr):
        return int(self.env.unwrapped.ale.getRAM()[addr])

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        ball_y = self._get_ram(self.BALL_Y_ADDR)

        bonus = 0.0
        if ball_y > self.descend_threshold:
            paddle_x = self._get_ram(self.PADDLE_X_ADDR)
            ball_x = self._get_ram(self.BALL_X_ADDR)
            distance = abs(paddle_x - ball_x)
            self._current_scale = self.scale_schedule(self.progress_remaining)
            bonus = self._current_scale * max(0.0, 1.0 - distance / self.max_distance)
            reward += bonus
            self._total_bonus += bonus
            self._bonus_steps += 1

        self._total_steps += 1

        if terminated or truncated:
            if isinstance(info, dict):
                info["proximity_bonus"] = self._total_bonus
                info["current_scale"] = self._current_scale

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self._total_bonus = 0.0
        self._bonus_steps = 0
        self._total_steps = 0
        return self.env.reset(**kwargs)
