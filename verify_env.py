import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

env = gym.make("ALE/Breakout-v5", render_mode="human")
obs, info = env.reset()

for _ in range(200):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        env.reset()

env.close()