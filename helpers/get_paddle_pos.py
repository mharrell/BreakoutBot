import ale_py
import gymnasium as gym
import numpy as np

gym.register_envs(ale_py)
env = gym.make("ALE/Breakout-v5", obs_type="ram", render_mode="human")
obs, _ = env.reset()

for step in range(200):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {step}: paddle={obs[70]}, ball_x={obs[72]}, ball_y={obs[90]}")
    if terminated or truncated:
        obs, _ = env.reset()

env.close()
