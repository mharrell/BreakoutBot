import ale_py
import gymnasium as gym
import glob
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

gym.register_envs(ale_py)

RUN_NAME = "PPO_31a"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
CHECKPOINT_PATH = os.path.join(PROJECT_DIR, "models", RUN_NAME, "checkpoint")
BEST_MODEL_PATH = os.path.join(PROJECT_DIR, "models", RUN_NAME, "best_model")
USE_BEST = False
HEADLESS = True   # True = no render window, fast. False = render window.

def get_latest_checkpoint(path):
    checkpoints = glob.glob(os.path.join(path, "latest_checkpoint_*_steps.zip"))
    if not checkpoints:
        return None
    latest = max(checkpoints, key=os.path.getmtime)
    return latest.replace(".zip", "")

if USE_BEST:
    model_path = BEST_MODEL_PATH
else:
    model_path = get_latest_checkpoint(CHECKPOINT_PATH)
    if not model_path:
        model_path = BEST_MODEL_PATH

if not os.path.exists(model_path + ".zip"):
    raise FileNotFoundError(f"No model found at {model_path}.zip")

env_kwargs = {"repeat_action_probability": 0.0}
if not HEADLESS:
    env_kwargs["render_mode"] = "human"
env = make_atari_env("ALE/Breakout-v5", n_envs=1, seed=None,
                     env_kwargs=env_kwargs)
env = VecFrameStack(env, n_stack=4)

print(f"Loading {os.path.abspath(model_path)}.zip")
model = PPO.load(model_path, env=env)

obs = env.reset()
episode = 1
scores = []
prev_score = None
frame = 0

while True:
    if not HEADLESS:
        env.render()
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    frame += 1

    lives = info[0].get("lives", -1)

    if done[0]:
        if lives == 0:
            real_score = info[0].get('episode', {}).get('r', '?')
            scores.append(real_score)
            avg = sum(scores) / len(scores)
            best = max(scores)
            repeat = " *** REPEAT ***" if real_score == prev_score else ""
            print(f"Game {episode:>3} | Score: {real_score:>6} | "
                  f"Frames: {frame:>5} | Avg: {avg:>6.1f} | "
                  f"Best: {best:>6}{repeat}")
            prev_score = real_score
            episode += 1
            frame = 0
            obs = env.reset()
        else:
            obs, _, _, _ = env.step([0])