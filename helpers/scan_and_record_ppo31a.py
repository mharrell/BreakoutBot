import ale_py
import gymnasium as gym
import os, glob
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

gym.register_envs(ale_py)

CHECKPOINT_DIR = r"C:\Users\Silver Pangolin\PycharmProjects\breakoutBot\models\PPO_31a\checkpoint"
VIDEO_DIR = r"C:\Users\Silver Pangolin\PycharmProjects\breakoutBot\recordings\videos"
os.makedirs(VIDEO_DIR, exist_ok=True)

checkpoints = glob.glob(os.path.join(CHECKPOINT_DIR, "latest_checkpoint_*_steps.zip"))
checkpoints.sort(key=lambda p: int(p.split("_")[-2]))  # sort by step number

print(f"Scanning {len(checkpoints)} checkpoints...\n")

best_score = 0
best_checkpoint = None

for cp in checkpoints:
    cp_name = os.path.basename(cp).replace(".zip", "")
    step = int(cp_name.split("_")[-2])

    env = make_atari_env("ALE/Breakout-v5", n_envs=1, seed=None,
                         env_kwargs={"repeat_action_probability": 0.0})
    env = VecFrameStack(env, n_stack=4)
    model = PPO.load(cp.replace(".zip", ""), env=env)

    obs = env.reset()
    done_flag = False
    while not done_flag:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, info = env.step(action)
        if dones[0]:
            lives = info[0].get("lives", -1)
            if lives == 0:
                score = float(info[0].get("episode", {}).get("r", 0))
                done_flag = True
    env.close()

    marker = ""
    if score > best_score:
        best_score = score
        best_checkpoint = cp_name
        marker = " <-- NEW BEST"
    if score > 200:
        print(f"  Step {step:>10,}: {score:6.0f}{marker}")

print(f"\nBest: {best_score:.0f} at {best_checkpoint}")

if best_score >= 100:
    print(f"\nRecording from {best_checkpoint} with render_mode='rgb_array'...")
    env = gym.make("ALE/Breakout-v5", render_mode="rgb_array", frameskip=1,
                   repeat_action_probability=0.0)
    env = gym.wrappers.RecordVideo(
        env, VIDEO_DIR,
        episode_trigger=lambda ep: True,
        name_prefix=f"ppo31a_{best_checkpoint}"
    )
    env = gym.wrappers.AtariPreprocessing(env, frame_skip=4, screen_size=84,
                                           grayscale_obs=True, scale_obs=False)
    env = gym.wrappers.FrameStackObservation(env, 4)

    model = PPO.load(
        os.path.join(CHECKPOINT_DIR, best_checkpoint), env=env)

    obs, _ = env.reset()
    done = False
    frames = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        frames += 1

    score = info.get("episode", {}).get("r", 0)
    env.close()
    print(f"Recorded: {score:.0f} pts, {frames} frames -> {VIDEO_DIR}")
    if score != best_score:
        print(f"WARNING: recorded score ({score:.0f}) != headless score ({best_score:.0f})")
else:
    print("No checkpoint scored >= 100. Nothing to record.")