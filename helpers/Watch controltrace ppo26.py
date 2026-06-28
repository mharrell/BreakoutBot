import ale_py
import gymnasium as gym
import os
import csv
import time
import statistics
import stable_baselines3.common.utils as sb3_utils
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from collections import Counter

gym.register_envs(ale_py)

# --- Which model to investigate ---
RUN_NAME = "PPO_26"
STICKY_ACTIONS = True   # must match this run's training env config
MODEL_PATH = f"../models/{RUN_NAME}/best_model"

# --- Control sample size ---
# Unlike watch_quickdeath_ppo26.py, this does NOT filter for bad outcomes —
# it logs whatever games come up next, good or bad, to get a baseline
# direction-correctness rate for comparison against the quick-death cluster.
NUM_GAMES = 20

# --- Direction-correctness analysis settings (must match the quick-death analysis) ---
BALL_Y_THRESHOLD = 150   # only count frames where ball is in the lower half
MIN_GAP = 10              # ignore frames where paddle/ball are already close

# --- Output ---
OUTPUT_DIR = f"../recordings/controltrace_{RUN_NAME}"
SAVE_PER_GAME_CSV = True  # set False if you only want the aggregate summary

os.makedirs(OUTPUT_DIR, exist_ok=True)

env_kwargs = {"repeat_action_probability": 0.25} if STICKY_ACTIONS else {}

sb3_utils.check_for_correct_spaces = lambda *args, **kwargs: None
env = make_atari_env("ALE/Breakout-v5", n_envs=1, seed=None, env_kwargs=env_kwargs)
env = VecFrameStack(env, n_stack=4)

print(f"Loading model from: {os.path.abspath(MODEL_PATH)}")
model = PPO.load(MODEL_PATH, custom_objects={"n_envs": 1})
model.set_env(env)

action_meanings = env.venv.envs[0].unwrapped.get_action_meanings()
print(f"Action meanings: {action_meanings}")


def get_ram(env):
    return env.venv.envs[0].unwrapped.ale.getRAM()


def write_trace(trace, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "action_idx", "action_label", "paddle_x", "ball_x", "ball_y", "lives"])
        writer.writerows(trace)


def direction_correctness(trace):
    """Same methodology as the quick-death analysis: compare each LEFT/RIGHT
    action against the PRE-action state (the frame the model actually saw),
    only counting frames where the ball is in the lower half with a real gap."""
    correct = 0
    wrong = 0
    for i in range(1, len(trace)):
        action_label = trace[i][2]
        if action_label not in ("LEFT", "RIGHT"):
            continue
        paddle_x_prev = trace[i - 1][3]
        ball_x_prev = trace[i - 1][4]
        ball_y_prev = trace[i - 1][5]
        if ball_y_prev < BALL_Y_THRESHOLD:
            continue
        gap = abs(ball_x_prev - paddle_x_prev)
        if gap < MIN_GAP:
            continue
        needs_increase = ball_x_prev > paddle_x_prev  # LEFT increases paddle_x in this RAM convention
        if needs_increase:
            if action_label == "LEFT":
                correct += 1
            else:
                wrong += 1
        else:
            if action_label == "RIGHT":
                correct += 1
            else:
                wrong += 1
    return correct, wrong


obs = env.reset()
episode = 1
scores = []
frame_count = 0

ram0 = get_ram(env)
trace_buffer = [(0, None, "(start)", int(ram0[70]), int(ram0[72]), int(ram0[90]), 5)]
game_start_time = time.time()

total_correct = 0
total_wrong = 0

print(f"Run: {RUN_NAME} | Sticky actions: {STICKY_ACTIONS}")
print(f"Logging {NUM_GAMES} unfiltered games as a control baseline")
print(f"Saving per-game traces to: {OUTPUT_DIR}" if SAVE_PER_GAME_CSV else "Not saving per-game CSVs (summary only)")
print("-" * 60)

while episode <= NUM_GAMES:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    frame_count += 1

    ram = get_ram(env)
    action_idx = int(action[0])
    action_label = action_meanings[action_idx] if action_idx < len(action_meanings) else str(action_idx)
    lives = info[0].get("lives", -1)
    trace_buffer.append((frame_count, action_idx, action_label,
                          int(ram[70]), int(ram[72]), int(ram[90]), lives))

    if done[0]:
        if lives == 0:
            real_score = float(info[0].get("episode", {}).get("r", 0))
            scores.append(real_score)
            game_frames = len(trace_buffer)
            elapsed = time.time() - game_start_time
            agent_fps = game_frames / elapsed if elapsed > 0 else 60

            correct, wrong = direction_correctness(trace_buffer)
            total_correct += correct
            total_wrong += wrong
            pct = (100 * correct / (correct + wrong)) if (correct + wrong) > 0 else float("nan")

            if SAVE_PER_GAME_CSV:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                trace_path = os.path.join(
                    OUTPUT_DIR,
                    f"{RUN_NAME}_control_{episode}_{int(real_score)}pts_{game_frames}f_{timestamp}_trace.csv"
                )
                write_trace(trace_buffer, trace_path)

            print(f"Game {episode:>3} | Score: {real_score:>6.0f} | Frames: {game_frames:>5} | "
                  f"Agent FPS: {agent_fps:>5.0f} | Direction-correct: {correct}/{correct+wrong} ({pct:.1f}%)")

            episode += 1
            obs = env.reset()
            frame_count = 0
            ram0 = get_ram(env)
            trace_buffer = [(0, None, "(start)", int(ram0[70]), int(ram0[72]), int(ram0[90]), 5)]
            game_start_time = time.time()
        else:
            obs, _, _, _ = env.step([0])
            frame_count += 1
            ram = get_ram(env)
            trace_buffer.append((frame_count, 0, "NOOP(refire)",
                                  int(ram[70]), int(ram[72]), int(ram[90]), lives))

env.close()

print("-" * 60)
print(f"--- Control Baseline: {RUN_NAME} ({NUM_GAMES} unfiltered games) ---")
print(f"Average score: {sum(scores)/len(scores):.1f}")
print(f"Score range: {min(scores):.0f} - {max(scores):.0f}")
total = total_correct + total_wrong
overall_pct = (100 * total_correct / total) if total > 0 else float("nan")
print(f"Direction-correctness (overall): {total_correct}/{total} ({overall_pct:.1f}%)")
print()
print("Compare this percentage against the quick-death cluster's 61.3% to see whether")
print("those failure games show a genuinely elevated wrong-direction rate, or whether")
print("this baseline sits in the same range (which would point to sticky-action noise")
print("or ordinary variance rather than a distinct failure mode).")