"""
Funnel recorder for PPO_32 — sticky-OFF verification (Experiment 4).
Confirms whether low-sticky single-phase training (p=0.05 from scratch) prevents
memorization or merely masks it like the Experiment 3 models.

Prediction table (from EXPERIMENTS.md Experiment 4):
  <=2 unique scores: p=0.05 training did NOT prevent memorization — model collapsed
  3-9 unique scores: Partial success — prevents total collapse but not full reactivity
  >=10 unique scores: Low-sticky training prevented memorization — recipe works

Comparison baselines (both confirmed memorized without sticky):
  PPO_30b_nosticky: 2 unique scores (0, 69), 99.8% zero-score
  PPO_31b_nosticky: 2 unique scores (29, 31), all games 31.0 points in 178 frames

500 games, sticky off, persistent env, seed=None, deterministic=True.
Usable at 200M midpoint (load checkpoint) and 400M completion (load final_model).
"""
import ale_py
import gymnasium as gym
import os
import csv
import cv2
import time
import subprocess
import stable_baselines3.common.utils as sb3_utils
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

gym.register_envs(ale_py)

RUN_NAME = "PPO_32_nosticky"
STICKY_ACTIONS = False
MODEL_PATH = f"models/PPO_32/final_model"

FUNNEL_THRESHOLD = 400
NUM_GAMES = 500
OUTPUT_DIR = "recordings"
LOG_PATH = os.path.join(OUTPUT_DIR, f"{RUN_NAME}_funnel_log.csv")
PLAYBACK_FPS = 60
SLOW_FACTOR = 0.5

os.makedirs(OUTPUT_DIR, exist_ok=True)

env_kwargs = {"repeat_action_probability": 0.25 if STICKY_ACTIONS else 0.0}

sb3_utils.check_for_correct_spaces = lambda *args, **kwargs: None
env = make_atari_env("ALE/Breakout-v5", n_envs=1, seed=None, env_kwargs=env_kwargs)
env = VecFrameStack(env, n_stack=4)

print(f"Loading model from: {os.path.abspath(MODEL_PATH)}")
model = PPO.load(MODEL_PATH, custom_objects={"n_envs": 1})
model.set_env(env)


def get_frame(env):
    frame = env.venv.envs[0].unwrapped.ale.getScreenRGB()
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def write_video(frames, path):
    if not frames:
        return
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), PLAYBACK_FPS, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()


def slow_down_video(input_path, factor=0.5):
    slow_path = input_path.replace(".mp4", "_slow.mp4")
    result = subprocess.run([
        "ffmpeg", "-i", input_path,
        "-filter:v", f"setpts={1/factor}*PTS",
        "-y", slow_path
    ], capture_output=True)
    if result.returncode == 0:
        return slow_path
    return None


log_is_new = not os.path.exists(LOG_PATH)
log_file = open(LOG_PATH, "a", newline="")
log_writer = csv.writer(log_file)
if log_is_new:
    log_writer.writerow(["episode", "real_score", "is_funnel", "frame_count",
                          "agent_fps", "timestamp"])
    log_file.flush()

obs = env.reset()
episode = 1
scores = []
funnel_count = 0
frame_buffer = [get_frame(env)]
game_start_time = time.time()

print(f"Run: {RUN_NAME} | Sticky: {STICKY_ACTIONS} | Threshold: {FUNNEL_THRESHOLD} | "
      f"Cap: {NUM_GAMES}")
print(f"Per-game log: {LOG_PATH}")
print(f"Experiment 4 memorization check: PPO_32 (trained p=0.05) evaluated sticky-off")
print(f"Baselines (both memorized without sticky):")
print(f"  PPO_30b_nosticky: 2 unique (0, 69), 99.8% zero-score")
print(f"  PPO_31b_nosticky: 2 unique (29, 31), all 31.0 pts, 178 frames")
print(f"Prediction: <=2 unique = MEMORIZED, 3-9 = partial, >=10 = GENERALIZING")
print("-" * 60)

while episode <= NUM_GAMES:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    frame_buffer.append(get_frame(env))

    lives = info[0].get("lives", -1)

    if done[0]:
        if lives == 0:
            real_score = float(info[0].get("episode", {}).get("r", 0))
            scores.append(real_score)
            avg = sum(scores) / len(scores)
            best = max(scores)
            elapsed = time.time() - game_start_time
            game_frames = len(frame_buffer)
            agent_fps = game_frames / elapsed if elapsed > 0 else 60

            is_funnel = real_score >= FUNNEL_THRESHOLD
            if is_funnel:
                funnel_count += 1
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(
                    OUTPUT_DIR,
                    f"{RUN_NAME}_funnel_{int(real_score)}pts_{timestamp}.mp4"
                )
                write_video(frame_buffer, save_path)
                slow_path = slow_down_video(save_path, factor=SLOW_FACTOR)
                print(f"*** FUNNEL SAVED: {save_path} ***")
                if slow_path:
                    print(f"*** SLOW: {slow_path} ***")

            funnel_rate = f"{funnel_count}/{episode} ({100*funnel_count/episode:.1f}%)"
            print(f"Game {episode:>5} | Score: {real_score:>6.0f} | Avg: {avg:>6.1f} | "
                  f"Best: {best:>6.0f} | Frames: {game_frames:>5} | "
                  f"Agent FPS: {agent_fps:>5.0f} | Funnels: {funnel_rate}")

            log_writer.writerow([episode, real_score, int(is_funnel), game_frames,
                                  round(agent_fps, 1), datetime.now().isoformat()])
            log_file.flush()

            episode += 1
            obs = env.reset()
            frame_buffer = [get_frame(env)]
            game_start_time = time.time()
        else:
            obs, _, _, _ = env.step([0])

env.close()
log_file.close()

print("-" * 60)
print(f"--- Final Results: {RUN_NAME} ({len(scores)} games) ---")
print(f"Average Score:    {sum(scores)/len(scores):.1f}")
print(f"Best Score:       {max(scores):.1f}")
print(f"Worst Score:      {min(scores):.1f}")
unique_scores = len(set(scores))
zero_count = sum(1 for s in scores if s == 0)
print(f"Unique Scores:    {unique_scores}")
print(f"Zero-score games: {zero_count}/{len(scores)} ({100*zero_count/len(scores):.1f}%)")
print(f"Funnel Rate ({FUNNEL_THRESHOLD}+ pts): {funnel_count}/{len(scores)} "
      f"({100*funnel_count/len(scores):.2f}%)")
print()
print("Comparison baselines (both memorized):")
print(f"  PPO_30b_nosticky: 2 unique (0, 69), 99.8% zero-score")
print(f"  PPO_31b_nosticky: 2 unique (29, 31), all 31.0 pts, 178 frames")
print()
if unique_scores <= 2:
    print("*** VERDICT: PPO_32 is MEMORIZED — low-sticky training did NOT prevent collapse ***")
elif unique_scores <= 9:
    print(f"VERDICT: PARTIAL SUCCESS — {unique_scores} unique scores prevents total collapse "
          f"but does not confirm full generalization")
else:
    print(f"VERDICT: GENERALIZING — {unique_scores} unique scores confirms low-sticky "
          f"training prevents memorization")
