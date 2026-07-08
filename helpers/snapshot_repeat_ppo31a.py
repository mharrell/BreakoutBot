import ale_py
import gymnasium as gym
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

gym.register_envs(ale_py)

RUN_NAME = "PPO_31a"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
CHECKPOINT_PATH = os.path.join(PROJECT_DIR, "models", RUN_NAME, "checkpoint")
BEST_MODEL_PATH = os.path.join(PROJECT_DIR, "models", RUN_NAME, "best_model")

import glob
checkpoints = glob.glob(os.path.join(CHECKPOINT_PATH, "latest_checkpoint_*_steps.zip"))
if checkpoints:
    model_path = max(checkpoints, key=os.path.getmtime).replace(".zip", "")
else:
    model_path = BEST_MODEL_PATH

if not os.path.exists(model_path + ".zip"):
    raise FileNotFoundError(f"No model found at {model_path}.zip")

print(f"Model: {os.path.basename(model_path)}")
print(f"Running 10 trials, 32 envs each, fresh envs per trial...\n")

N_TRIALS = 10
N_ENVS = 32

model = PPO.load(model_path)

all_means = []

for trial in range(N_TRIALS):
    env = make_atari_env("ALE/Breakout-v5", n_envs=N_ENVS, seed=None,
                         env_kwargs={"repeat_action_probability": 0.0})
    env = VecFrameStack(env, n_stack=4)
    model.set_env(env)

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
    all_means.append(scores.mean())
    unique = len(set(scores))
    print(f"Trial {trial+1:>2}: avg={scores.mean():7.1f}  best={scores.max():6.0f}  "
          f"worst={scores.min():6.0f}  unique={unique}  "
          f"{'ALL SAME' if unique == 1 else ''}")

print(f"\nAcross all {N_TRIALS} trials ({N_TRIALS * N_ENVS} episodes):")
print(f"  Mean of trial means: {np.mean(all_means):.1f}")
print(f"  Best trial mean: {np.max(all_means):.1f}")
print(f"  Worst trial mean: {np.min(all_means):.1f}")