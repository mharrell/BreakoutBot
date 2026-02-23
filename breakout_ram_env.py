import numpy as np
import gymnasium as gym
from gymnasium import spaces

SCREEN_WIDTH = 191.0


class BreakoutRamEnv(gym.Wrapper):

    def __init__(self):
        env = gym.make("ALE/Breakout-v5", obs_type="ram")
        super().__init__(env)

        # Override observation space — 128 RAM values normalized to 0-1
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(128,),
            dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._normalize(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Extract positions
        paddle_x = obs[70]
        ball_x = obs[72]
        ball_y = obs[90]

        # Ball tracking reward — only when ball is in lower half
        if ball_y > 128:
            raw = 1.0 - abs(int(paddle_x) - int(ball_x)) / SCREEN_WIDTH
            tracking_reward = max(0.0, min(1.0, raw))
        else:
            tracking_reward = 0.0

        # Combine original reward with tracking bonus
        shaped_reward = reward + 0.1 * tracking_reward

        return self._normalize(obs), shaped_reward, terminated, truncated, info

    def _normalize(self, obs):
        return obs.astype(np.float32) / 255.0