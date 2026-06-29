import ale_py
import gymnasium as gym
import os
import csv
import cv2
import time
import subprocess
import statistics
import stable_baselines3.common.utils as sb3_utils
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

gym.register_envs(ale_py)

# --- Which model to investigate ---
RUN_NAME = "PPO_26"
STICKY_ACTIONS = True   # must match this run's training env config
MODEL_PATH = f"../models/{RUN_NAME}/best_model"

# --- What counts as a "quick death" worth capturing ---
# Tuned from the PPO_26 1,013-game sample: several score<=15 games clustered
# tightly around frame_count==95. Widen the frame window a bit either side
# to catch near-misses of the same pattern, not just exact matches.
QUICK_DEATH_MAX_SCORE = 15
QUICK_DEATH_FRAME_MIN = 10
QUICK_DEATH_FRAME_MAX = 110

# --- Run controls ---
MAX_GAMES = 5000          # safety cap so this doesn't run forever unattended
STOP_AFTER_CAPTURES = 5   # stop once this many matching games are saved
RENDER_LIVE = False       # True = pop up a game window as it plays (slower)

# --- Wall-pinning detection (paddle x ranges roughly 0-191) ---
PADDLE_X_MAX = 191
WALL_MARGIN = 10          # within this many px of an edge counts as "at the wall"
STATIONARY_STDEV = 1.5    # paddle_x stdev below this over the tail counts as "not moving"
TAIL_FRAMES = 20          # how many frames before death to evaluate for the verdict

# --- Output ---
OUTPUT_DIR = f"../recordings/quickdeath_{RUN_NAME}"
PLAYBACK_FPS = 60
SLOW_FACTOR = 0.25        # quarter speed — these events are short, so slow it down more than usual

os.makedirs(OUTPUT_DIR, exist_ok=True)

env_kwargs = {"repeat_action_probability": 0.25} if STICKY_ACTIONS else {}
if RENDER_LIVE:
    env_kwargs["render_mode"] = "human"

sb3_utils.check_for_correct_spaces = lambda *args, **kwargs: None
env = make_atari_env("ALE/Breakout-v5", n_envs=1, seed=None, env_kwargs=env_kwargs)
env = VecFrameStack(env, n_stack=4)

print(f"Loading model from: {os.path.abspath(MODEL_PATH)}")
model = PPO.load(MODEL_PATH, custom_objects={"n_envs": 1})
model.set_env(env)

action_meanings = env.venv.envs[0].unwrapped.get_action_meanings()
print(f"Action meanings: {action_meanings}")


def get_frame(env):
    frame = env.venv.envs[0].unwrapped.ale.getScreenRGB()
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def get_ram(env):
    return env.venv.envs[0].unwrapped.ale.getRAM()


def write_video(frames, path):
    if not frames:
        return
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), PLAYBACK_FPS, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()


def slow_down_video(input_path, factor=0.25):
    slow_path = input_path.replace(".mp4", "_slow.mp4")
    result = subprocess.run([
        "ffmpeg", "-i", input_path,
        "-filter:v", f"setpts={1/factor}*PTS",
        "-y", slow_path
    ], capture_output=True)
    if result.returncode == 0:
        return slow_path
    print(f"ffmpeg error: {result.stderr.decode()}")
    return None


def write_trace(trace, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "action_idx", "action_label", "paddle_x", "ball_x", "ball_y", "lives"])
        writer.writerows(trace)


def verdict_for_tail(trace, tail_n=TAIL_FRAMES):
    """Look at the last tail_n frames of paddle_x and classify what was happening."""
    tail = [row[3] for row in trace[-tail_n:]]  # paddle_x column
    if len(tail) < 3:
        return "not enough frames to judge"
    spread = statistics.pstdev(tail)
    avg_x = sum(tail) / len(tail)
    at_left_wall = avg_x <= WALL_MARGIN
    at_right_wall = avg_x >= (PADDLE_X_MAX - WALL_MARGIN)
    if spread < STATIONARY_STDEV:
        if at_left_wall:
            return f"PINNED AT LEFT WALL (paddle_x~{avg_x:.0f}, stdev {spread:.2f})"
        elif at_right_wall:
            return f"PINNED AT RIGHT WALL (paddle_x~{avg_x:.0f}, stdev {spread:.2f})"
        else:
            return f"STATIONARY MID-SCREEN (paddle_x~{avg_x:.0f}, stdev {spread:.2f})"
    else:
        return f"ACTIVELY MOVING (paddle_x range in tail: {min(tail):.0f}-{max(tail):.0f}, stdev {spread:.2f})"


obs = env.reset()
episode = 1
captures = 0
scores = []
frame_buffer = [get_frame(env)]

ram0 = get_ram(env)
trace_buffer = [(0, None, "(start)", int(ram0[70]), int(ram0[72]), int(ram0[90]), 5)]
game_start_time = time.time()

print(f"Run: {RUN_NAME} | Sticky actions: {STICKY_ACTIONS}")
print(f"Hunting for: score <= {QUICK_DEATH_MAX_SCORE} AND "
      f"{QUICK_DEATH_FRAME_MIN} <= frames <= {QUICK_DEATH_FRAME_MAX}")
print(f"Will stop after {STOP_AFTER_CAPTURES} captures or {MAX_GAMES} games")
print(f"Captures saved to: {OUTPUT_DIR}")
print("-" * 60)

while episode <= MAX_GAMES and captures < STOP_AFTER_CAPTURES:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    frame_buffer.append(get_frame(env))

    ram = get_ram(env)
    action_idx = int(action[0])
    action_label = action_meanings[action_idx] if action_idx < len(action_meanings) else str(action_idx)
    lives = info[0].get("lives", -1)
    trace_buffer.append((len(frame_buffer) - 1, action_idx, action_label,
                          int(ram[70]), int(ram[72]), int(ram[90]), lives))

    if done[0]:
        if lives == 0:
            real_score = float(info[0].get("episode", {}).get("r", 0))
            scores.append(real_score)
            game_frames = len(frame_buffer)
            elapsed = time.time() - game_start_time
            agent_fps = game_frames / elapsed if elapsed > 0 else 60

            is_match = (real_score <= QUICK_DEATH_MAX_SCORE
                        and QUICK_DEATH_FRAME_MIN <= game_frames <= QUICK_DEATH_FRAME_MAX)

            tag = ""
            if is_match:
                captures += 1
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base = os.path.join(
                    OUTPUT_DIR,
                    f"{RUN_NAME}_quickdeath_{int(real_score)}pts_{game_frames}f_{timestamp}"
                )
                video_path = base + ".mp4"
                trace_path = base + "_trace.csv"

                write_video(frame_buffer, video_path)
                slow_path = slow_down_video(video_path, factor=SLOW_FACTOR)
                write_trace(trace_buffer, trace_path)
                verdict = verdict_for_tail(trace_buffer)

                tag = (f"*** CAPTURED ({captures}/{STOP_AFTER_CAPTURES}): {video_path}\n"
                       f"    trace CSV: {trace_path}\n"
                       f"    verdict (last {TAIL_FRAMES} frames before death): {verdict}")
                if slow_path:
                    tag += f"\n    slow version: {slow_path}"

            print(f"Game {episode:>5} | Score: {real_score:>6.0f} | Frames: {game_frames:>5} | "
                  f"Agent FPS: {agent_fps:>5.0f}{'  <-- match' if is_match else ''}")
            if tag:
                print(tag)

            episode += 1
            obs = env.reset()
            frame_buffer = [get_frame(env)]
            ram0 = get_ram(env)
            trace_buffer = [(0, None, "(start)", int(ram0[70]), int(ram0[72]), int(ram0[90]), 5)]
            game_start_time = time.time()
        else:
            obs, _, _, _ = env.step([0])
            ram = get_ram(env)
            trace_buffer.append((len(frame_buffer) - 1, 0, "NOOP(refire)",
                                  int(ram[70]), int(ram[72]), int(ram[90]), lives))

env.close()

print("-" * 60)
if captures >= STOP_AFTER_CAPTURES:
    print(f"Done — captured {captures} matching games across {episode - 1} total games played.")
else:
    print(f"Stopped at {MAX_GAMES} games with only {captures} matches captured. "
          f"Consider widening QUICK_DEATH_FRAME_MIN/MAX or raising MAX_GAMES.")