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
        self._prev_ball_y = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_ball_y = 0
        return self._normalize(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        ball_y = obs[90]

        # Paddle hit detected — ball reversed direction near bottom of screen
        if self._prev_ball_y > ball_y and self._prev_ball_y > 180:
            paddle_hit_reward = 0.5
        else:
            paddle_hit_reward = 0.0
        self._prev_ball_y = ball_y

        info['raw_game_reward'] = reward
        shaped_reward = reward + paddle_hit_reward

        return self._normalize(obs), shaped_reward, terminated, truncated, info

    def _normalize(self, obs):
        return obs.astype(np.float32) / 255.0


class BreakoutRamEvalEnv(BreakoutRamEnv):
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        # Fire to launch ball
        obs, _, _, _, _ = super().step(1)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._normalize(obs), reward, terminated, truncated, info