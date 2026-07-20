"""
Funnel recorder for PPO_30a — non-sticky Phase 1 model (100M steps).
Uses persistent env with seed=None (same approach as PPO_25 funnel recorder).
Fresh-env-per-game does NOT help for Breakout — ball direction is determined
by ALE internal state, not the seed integer. Persistent env lets natural
game-to-game state accumulation provide variation.

Includes memorization check: if unique_scores == 1 after 20 games, the model
has memorized a fixed sequence. Stop early and note it rather than running 1000.
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

RUN_NAME = "PPO_30a"
STICKY_ACTIONS = False
MODEL_PATH = f"models/{RUN_NAME}/best_model"

FUNNEL_THRESHOLD = 400
NUM_GAMES = 1000      # capped — non-sticky models may memorize
MEMORIZATION_CHECK_AT = 20  # stop early if all scores identical by this point
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
print(f"Memorization check at game {MEMORIZATION_CHECK_AT} — "
      f"will stop early if all scores identical (fixed sequence detected)")
print(f"Per-game log: {LOG_PATH}")
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

            # Memorization check
            if episode == MEMORIZATION_CHECK_AT:
                unique = len(set(scores))
                if unique <= 2:
                    print(f"\n*** MEMORIZATION DETECTED at game {episode}: "
                          f"only {unique} unique score(s) across {episode} games ***")
                    print("Model is playing a fixed sequence. Stopping early.")
                    print("This Phase 1 model may not generalize — "
                          "check whether Phase 2 (PPO_30b) fixes this.")
                    break
                else:
                    print(f"  [Memorization check passed: {unique} unique scores "
                          f"across {episode} games — model is generalizing]")

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
zero_count = sum(1 for s in scores if s == 0)
print(f"Zero-score games: {zero_count}/{len(scores)} ({100*zero_count/len(scores):.1f}%)")
print(f"Unique scores:    {len(set(scores))} "
      f"(1-2 = memorization, many = genuine generalization)")
print(f"Funnel Rate ({FUNNEL_THRESHOLD}+ pts): {funnel_count}/{len(scores)} "
      f"({100*funnel_count/len(scores):.1f}%)")
