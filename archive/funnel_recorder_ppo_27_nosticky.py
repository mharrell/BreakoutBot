"""
Funnel recorder for PPO_27 — sticky-OFF verification.
PPO_27 is the only model trained entirely with sticky actions (p=0.25 from
step one, ~1B steps). Unlike every other sticky-trained model (PPO_26, PPO_28,
PPO_29, PPO_30b, PPO_31b), PPO_27 NEVER had a non-sticky pretraining phase.

This is the critical test: can training with stochasticity from step one prevent
memorization, even if it produces a weaker policy? PPO_27 was the worst single-env
performer (21.3% zero-score with sticky on) but holds the all-time eval record
(147.02). It may be the only genuinely generalizing model in the project.

If PPO_27 collapses without sticky: sticky actions during training also don't
  prevent memorization — they just produce different-looking scripts. No model
  in this project has ever generalized.

If PPO_27 does NOT collapse: training with stochasticity from step one is
  the only path that prevents memorization. The performance cost (21% zero-score)
  is the price of genuine reactivity.

500 games, sticky off, persistent env, seed=None, deterministic=True.
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

RUN_NAME = "PPO_27_nosticky"
STICKY_ACTIONS = False
MODEL_PATH = f"models/PPO_27/final_model"

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
print(f"PPO_27: p=0.25 from scratch, ~1B steps — the only model without non-sticky pretraining")
print(f"PPO_27 sticky-ON baseline: worst single-env performer (21.3% zero-score)")
print(f"PPO_27 sticky-ON eval record: 147.02 (all-time best)")
print(f"Sticky-trained models tested nosticky so far: ALL COLLAPSED")
print(f"  PPO_26: 1 unique score (60 pts, 264 frames)")
print(f"  PPO_30b: 2 unique scores (0, 69), 99.8% zeros")
print(f"  PPO_31b: 2 unique scores (29, 31), all 31 pts, 178 frames")
print(f"PPO_27 is the last untested sticky-trained model — and the only one trained")
print(f"with stochasticity from step one. If this also collapses, no model in this")
print(f"project has ever genuinely generalized.")
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
print("Comparison — every sticky-trained model tested nosticky:")
print(f"  PPO_26:  1 unique score (60 pts, 264 frames) — MEMORIZED")
print(f"  PPO_30b: 2 unique scores (0, 69), 99.8% zeros — MEMORIZED")
print(f"  PPO_31b: 2 unique scores (29, 31), all 31 pts — MEMORIZED")
print(f"  PPO_27:  {unique_scores} unique scores")
print()
if unique_scores <= 2:
    print("*** VERDICT: PPO_27 is MEMORIZED — sticky-from-scratch also doesn't prevent collapse ***")
    print("    No model in this project has ever genuinely generalized.")
elif unique_scores <= 9:
    print(f"VERDICT: PARTIAL — {unique_scores} unique scores with det=True suggests some reactivity")
else:
    print(f"VERDICT: PPO_27 GENERALIZES — {unique_scores} unique scores confirms training with")
    print(f"    stochasticity from step one prevents memorization, even if performance suffers.")
