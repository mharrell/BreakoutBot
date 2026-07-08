import ale_py
import gymnasium as gym
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

gym.register_envs(ale_py)

RUN_NAME = "PPO_31a"
N_ENVS = 32

model_path = f"../models/{RUN_NAME}/best_model"
if not os.path.exists(model_path + ".zip"):
    import glob
    checkpoints = glob.glob(f"../models/{RUN_NAME}/checkpoint/latest_checkpoint_*_steps.zip")
    if checkpoints:
        model_path = max(checkpoints, key=os.path.getmtime).replace(".zip", "")
    else:
        raise FileNotFoundError(f"No model found for {RUN_NAME}")

env = make_atari_env("ALE/Breakout-v5", n_envs=N_ENVS, seed=None,
                     env_kwargs={"repeat_action_probability": 0.0})
env = VecFrameStack(env, n_stack=4)

print(f"Loading {os.path.abspath(model_path)}")
model = PPO.load(model_path, env=env)

obs = env.reset()
scores = np.zeros(N_ENVS)
frames = np.zeros(N_ENVS, dtype=int)
done_flags = np.zeros(N_ENVS, dtype=bool)

while not done_flags.all():
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, dones, info = env.step(action)

    for i in range(N_ENVS):
        if not done_flags[i]:
            frames[i] += 1
            if dones[i]:
                lives = info[i].get("lives", -1)
                if lives == 0:
                    scores[i] = float(info[i].get("episode", {}).get("r", 0))
                    done_flags[i] = True

env.close()

print(f"\n{N_ENVS} envs, one episode each:\n")
sorted_idx = np.argsort(scores)[::-1]
for rank, i in enumerate(sorted_idx):
    print(f"  {'🥇' if rank == 0 else '🥈' if rank == 1 else '🥉' if rank == 2 else f'{rank+1}.'} "
          f"Env {i:>2}: {scores[i]:>6.0f} pts, {frames[i]:>5} frames")

print(f"\n  Best: {scores.max():.0f}  |  Worst: {scores.min():.0f}  |  "
      f"Avg: {scores.mean():.1f}  |  Median: {np.median(scores):.0f}  |  "
      f"Unique: {len(set(scores))}")