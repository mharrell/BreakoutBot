import ale_py
import gymnasium as gym
import numpy as np

gym.register_envs(ale_py)

env = gym.make("ALE/Breakout-v5", obs_type="ram", render_mode="human")
obs, _ = env.reset()

print("RAM size:", len(obs))
print("\nWatching RAM changes as game runs...")
print("Move the paddle and watch which addresses change\n")

prev_obs = obs.copy()
for step in range(500):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    # Find addresses that changed
    changed = np.where(obs != prev_obs)[0]
    if len(changed) > 0 and len(changed) < 10:
        print(f"Step {step}: Changed addresses: {changed}, Values: {obs[changed]}")

    prev_obs = obs.copy()
    if terminated or truncated:
        obs, _ = env.reset()
        prev_obs = obs.copy()

env.close()