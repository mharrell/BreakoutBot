"""
BallPositionWrapper — expose ball position from ALE RAM for aux supervision.

Reads ball X (RAM 99) and ball Y (RAM 101) every step and adds them to the
info dict. Used by BallTrackingCallback to provide ground-truth labels for
the auxiliary ball-position prediction task, which forces the CNN features
to encode ball location.

RAM addresses from ALE Breakout:
  - Ball X:  RAM[99]   (0-160, pixel x coordinate)
  - Ball Y:  RAM[101]  (0-210, pixel y coordinate)
  - Paddle X: RAM[72]  (for reference, not used here)
"""
import gymnasium as gym
import numpy as np


class BallPositionWrapper(gym.Wrapper):
    """Add ball_x, ball_y to the info dict from ALE RAM.

    Must be stacked BEFORE wrappers that modify the observation (GrayscaleResize,
    FrameStack), since it reads RAM directly rather than from pixel observations.
    """

    RAM_BALL_X = 99
    RAM_BALL_Y = 101

    def __init__(self, env):
        super().__init__(env)
        self._ale = None

    def _get_ale(self):
        """Cache the ALE interface reference."""
        if self._ale is None:
            self._ale = self.env.unwrapped.ale
        return self._ale

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        ale = self._get_ale()
        ram = ale.getRAM()
        info['ball_x'] = int(ram[self.RAM_BALL_X])
        info['ball_y'] = int(ram[self.RAM_BALL_Y])
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        ale = self._get_ale()
        ram = ale.getRAM()
        info['ball_x'] = int(ram[self.RAM_BALL_X])
        info['ball_y'] = int(ram[self.RAM_BALL_Y])
        return obs, info
