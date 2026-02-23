import ale_py
import gymnasium as gym
import os
from stable_baselines3 import PPO
from breakout_ram_env import BreakoutRamEnv

gym.register_envs(ale_py)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(project_root, "models", "best_model")

env = BreakoutRamEnv()
env.env = gym.make("ALE/Breakout-v5", obs_type="ram", render_mode="human")

model = PPO.load(model_path, device='cpu')

obs, _ = env.reset()

# Fire to launch first ball
obs, _, _, _, _ = env.step(1)

prev_ram_ball_y = 0

for _ in range(10_000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    # If ball y is 0 the ball isn't in play — fire to launch
    ball_y = int(obs[90] * 255)
    if ball_y == 0:
        obs, _, _, _, _ = env.step(1)

    if terminated or truncated:
        obs, _ = env.reset()
        obs, _, _, _, _ = env.step(1)