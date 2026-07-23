"""
Playback script — watch PPO_47 play with and without teleports.

Shows the model playing on:
  1. Clean Breakout (no teleport) — the eval environment
  2. Teleport Breakout (60% mid-flight) — the training environment

Each game: 5 lives, rendered to screen.
Uses a manual frame stack since VecFrameStack requires VecEnv.
"""
import sys
import time
import cv2
import numpy as np
import gymnasium as gym
from collections import deque
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv, NoopResetEnv, FireResetEnv, EpisodicLifeEnv,
)
from ale_breakout_flight_randomized import ALEBreakoutFlightRandomized
from autoreset_wrapper import AutoResetWrapper

import ale_py
gym.register_envs(ale_py)

MODEL_PATH = "./models/PPO_47/best_model.zip"
TELEPORT_PROB = 0.60
GAMES_PER_ENV = 5
FRAME_STACK = 4


# ---------------------------------------------------------------------------
# Preprocessing (matches training pipeline)
# ---------------------------------------------------------------------------
class PreprocessObs(gym.ObservationWrapper):
    """Grayscale + 84x84 resize -> (84, 84, 1) uint8."""
    def __init__(self, env, width=84, height=84):
        super().__init__(env)
        # Set attributes BEFORE observation_space (which may trigger observation())
        self.p_width = width
        self.p_height = height
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(height, width, 1), dtype=np.uint8)

    def observation(self, obs):
        if obs.ndim == 3 and obs.shape[-1] == 3:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        if obs.ndim == 2:
            obs = obs[:, :, None]
        obs = cv2.resize(obs, (self.p_width, self.p_height),
                         interpolation=cv2.INTER_AREA)
        if obs.ndim == 2:
            obs = obs[:, :, None]
        return obs


class SingleFrameStack(gym.Wrapper):
    """Frame stack for a single (non-vectorized) env. -> (84, 84, n_stack)."""
    def __init__(self, env, n_stack=4):
        super().__init__(env)
        self.n_stack = n_stack
        self._frames = deque(maxlen=n_stack)
        obs_shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(obs_shape[0], obs_shape[1], n_stack),
            dtype=np.uint8)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.n_stack):
            self._frames.append(obs)
        return self._stack(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(obs)
        return self._stack(), reward, terminated, truncated, info

    def _stack(self):
        return np.concatenate(list(self._frames), axis=-1)


# ---------------------------------------------------------------------------
# Environment builders
# ---------------------------------------------------------------------------
def make_clean_env():
    """Clean ALE Breakout with rendering."""
    env = gym.make("ALE/Breakout-v5", frameskip=4, repeat_action_probability=0,
                   render_mode="human")
    env = NoopResetEnv(env, noop_max=30)
    env = FireResetEnv(env)
    env = EpisodicLifeEnv(env)
    env = PreprocessObs(env, width=84, height=84)
    env = ClipRewardEnv(env)
    env = AutoResetWrapper(env)
    env = SingleFrameStack(env, n_stack=FRAME_STACK)
    return env


def make_teleport_env():
    """60% mid-flight teleport with rendering.
    Uses frameskip=1 (training uses 1 + VecFrameStack for history)."""
    env = gym.make("ALE/Breakout-v5", frameskip=1, repeat_action_probability=0,
                   render_mode="human")
    env = NoopResetEnv(env, noop_max=30)
    env = ALEBreakoutFlightRandomized(env, teleport_prob=TELEPORT_PROB)
    env = FireResetEnv(env)
    env = EpisodicLifeEnv(env)
    env = PreprocessObs(env, width=84, height=84)
    env = ClipRewardEnv(env)
    env = AutoResetWrapper(env)
    env = SingleFrameStack(env, n_stack=FRAME_STACK)
    return env


# ---------------------------------------------------------------------------
# Game runner
# ---------------------------------------------------------------------------
def play_single_game(env, model, game_num, label):
    """Play one full game (5 lives). EpisodicLifeEnv splits lives into
    episodes; AutoResetWrapper handles autoreset between them."""
    obs, _ = env.reset()
    life_scores = []
    current_life_score = 0
    current_life_steps = 0
    total_steps = 0
    lives_completed = 0
    interventions = 0

    while lives_completed < 5:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        current_life_score += reward
        current_life_steps += 1
        total_steps += 1

        if terminated or truncated:
            lives_completed += 1
            life_scores.append((current_life_score, current_life_steps))
            current_life_score = 0
            current_life_steps = 0

        # Track interventions for teleport env
        env_leaf = env
        for _ in range(12):
            if hasattr(env_leaf, 'intervention_count'):
                interventions = env_leaf.intervention_count
                break
            if hasattr(env_leaf, 'env'):
                env_leaf = env_leaf.env
            else:
                break

        # Safety timeout: 50k steps per life
        if current_life_steps > 50_000:
            print(f"    ⚠ Life {lives_completed+1}: TIMEOUT at {current_life_steps} steps")
            break

    total_score = int(sum(s for s, _ in life_scores))
    life_detail = " | ".join(
        f"L{i+1}:{int(s):+d}/{st}" for i, (s, st) in enumerate(life_scores)
    )
    print(f"  [{label}] Game {game_num}: total={total_score:+d}, "
          f"lives={lives_completed}, interventions={interventions}")
    print(f"    {life_detail}")
    return total_score, interventions, total_steps


def run_session(env, model, n_games, label):
    """Run n_games on the given environment."""
    print(f"\n{'─'*50}")
    print(f"  {label}")
    print(f"{'─'*50}")
    scores = []
    all_interventions = []
    for g in range(n_games):
        try:
            score, interventions, steps = play_single_game(env, model, g + 1, label)
            scores.append(score)
            all_interventions.append(interventions)
        except KeyboardInterrupt:
            print("\n  Interrupted.")
            break
    if scores:
        print(f"  Summary: avg={np.mean(scores):.1f}, "
              f"min={min(scores)}, max={max(scores)}, "
              f"unique={len(set(scores))}")
        if any(i > 0 for i in all_interventions):
            print(f"  Avg interventions/game: {np.mean(all_interventions):.0f}")
    return scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Loading model: {MODEL_PATH}")
    model = PPO.load(MODEL_PATH, device="cuda")
    print(f"Model steps: {model.num_timesteps:,}")
    print()
    print(f"Playing {GAMES_PER_ENV} games per environment")
    print(f"Model action: deterministic (argmax)")
    print(f"Wrapper: AutoResetWrapper + SingleFrameStack(4)")
    print()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║  CLEAN BREAKOUT first — watch the model on normal game  ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print("\nStarting in 3 seconds...")
    time.sleep(3)

    clean_env = make_clean_env()
    clean_scores = run_session(clean_env, model, GAMES_PER_ENV, "CLEAN")
    clean_env.close()

    print("\n\n")
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  TELEPORT BREAKOUT — 60% mid-flight, frameskip=1       ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print("\nStarting in 3 seconds...")
    time.sleep(3)

    teleport_env = make_teleport_env()
    teleport_scores = run_session(teleport_env, model, GAMES_PER_ENV, "TELEPORT")
    teleport_env.close()

    print(f"\n{'='*50}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*50}")
    if clean_scores:
        print(f"  Clean:    avg={np.mean(clean_scores):.1f}, "
              f"min={min(clean_scores)}, max={max(clean_scores)}, "
              f"unique={len(set(clean_scores))}")
    if teleport_scores:
        print(f"  Teleport: avg={np.mean(teleport_scores):.1f}, "
              f"min={min(teleport_scores)}, max={max(teleport_scores)}, "
              f"unique={len(set(teleport_scores))}")
