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
VIDEO_DIR = os.path.join(PROJECT_DIR, "recordings", "videos")

os.makedirs(VIDEO_DIR, exist_ok=True)

checkpoints = glob.glob(os.path.join(CHECKPOINT_PATH, "latest_checkpoint_*_steps.zip"))
if checkpoints:
    model_path = max(checkpoints, key=os.path.getmtime).replace(".zip", "")
else:
    model_path = os.path.join(PROJECT_DIR, "models", RUN_NAME, "best_model")

env = make_atari_env("ALE/Breakout-v5", n_envs=1, seed=None,
                     env_kwargs={"repeat_action_probability": 0.0})
env = VecFrameStack(env, n_stack=4)

print(f"Loading {os.path.basename(model_path)}")
model = PPO.load(model_path, env=env)

obs = env.reset()
episode = 1
frame = 0

while episode <= 3:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    frame += 1

    if done[0]:
        lives = info[0].get("lives", -1)
        if lives == 0:
            real_score = info[0].get("episode", {}).get("r", "?")
            print(f"Game {episode}: {real_score} pts, {frame} frames")
            episode += 1
            frame = 0
            obs = env.reset()
        else:
            obs, _, _, _ = env.step([0])

env.close()
model = None

# Now record one game with rendering
print("\nRecording one game with render_mode='rgb_array'...")
env = gym.make("ALE/Breakout-v5", render_mode="rgb_array",
               repeat_action_probability=0.0)
env = gym.wrappers.RecordVideo(
    env, VIDEO_DIR,
    episode_trigger=lambda ep: True,
    name_prefix="ppo31a_headless"
)
env = gym.wrappers.AtariPreprocessing(env, frame_skip=4, screen_size=84,
                                       grayscale_obs=True, scale_obs=False)
env = gym.wrappers.FrameStackObservation(env, 4)

model = PPO.load(model_path, env=env)

obs, _ = env.reset()
done = False
score = 0
frames = 0
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    score = info.get("episode", {}).get("r", 0)
    frames += 1

env.close()
print(f"Recorded: {score:.0f} pts, {frames} frames")
print(f"Video saved to: {VIDEO_DIR}")