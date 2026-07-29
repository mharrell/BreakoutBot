"""
Quick diagnostic: isolate whether frameskip=1 or the wrapper is the problem.
"""
import sys
import numpy as np
import gymnasium as gym
from stable_baselines3.common.atari_wrappers import FireResetEnv
import ale_py
gym.register_envs(ale_py)

from adversarial_ball_wrapper import AdversarialBallWrapper

BALL_X_ADDR = 99
BALL_Y_ADDR = 101
PADDLE_X_ADDR = 72

NOOP, FIRE, RIGHT, LEFT = 0, 1, 2, 3


def perfect_track(ball_x, paddle_x, ball_y, frame):
    """Perfect tracking: paddle matches ball_x, fire when needed."""
    if ball_y > 180:
        return FIRE
    if paddle_x < ball_x:
        return RIGHT
    elif paddle_x > ball_x:
        return LEFT
    return NOOP


def run_one(env, n_games=10, max_frames=5000):
    scores = []
    for _ in range(n_games):
        env.reset()
        score = 0.0
        for f in range(max_frames):
            ram = env.unwrapped.ale.getRAM()
            bx, by, px = int(ram[BALL_X_ADDR]), int(ram[BALL_Y_ADDR]), int(ram[PADDLE_X_ADDR])
            action = perfect_track(bx, px, by, f)
            obs, reward, terminated, truncated, info = env.step(action)
            score += reward
            if terminated or truncated:
                break
        scores.append(score)
    return np.mean(scores), np.std(scores), np.min(scores), np.max(scores)


print("=" * 70)
print("Diagnostic: perfect tracking score across configs")
print("=" * 70)
print(f"{'Config':<45} {'Mean':>6} {'±':>5} {'Min':>5} {'Max':>5}")
print("-" * 70)

configs = [
    # (label, frameskip, adversarial, dead_zone, gain, max_push)
    ("fs=4, NO wrapper (baseline)", 4, False, 0, 0, 0),
    ("fs=1, NO wrapper", 1, False, 0, 0, 0),
    ("fs=4, push max=4", 4, True, 4.0, 0.5, 4.0),
    ("fs=4, push max=2", 4, True, 4.0, 0.5, 2.0),
    ("fs=1, push max=4 (current)", 1, True, 4.0, 0.5, 4.0),
    ("fs=1, push max=2", 1, True, 4.0, 0.5, 2.0),
    ("fs=1, push max=1", 1, True, 4.0, 0.5, 1.0),
    ("fs=1, push max=0.5", 1, True, 4.0, 0.5, 0.5),
    ("fs=1, dead=8 max=2", 1, True, 8.0, 0.5, 2.0),
    ("fs=1, dead=8 max=1", 1, True, 8.0, 0.5, 1.0),
]

for label, fs, adv, dz, gain, mp in configs:
    env = gym.make("ALE/Breakout-v5", frameskip=fs, repeat_action_probability=0)
    env = FireResetEnv(env)
    if adv:
        env = AdversarialBallWrapper(env, dead_zone=dz, proportional_gain=gain,
                                      paddle_zone_y=140, max_push=mp)
    mean, std, lo, hi = run_one(env, n_games=10)
    env.close()
    print(f"{label:<45} {mean:6.1f} {std:5.1f} {lo:5.0f} {hi:5.0f}")

print()
print("If fs=1 NO wrapper is also broken -> frameskip=1 itself is the problem.")
print("If fs=4 push max=4 works -> we should use frameskip=4 with push.")
