"""
PPO_130 — Continue PPO_128 (scale=0.025) from 25M to 50M.

Part of Experiment 34 scale sweep. PPO_128 (low scale=0.025) never broke
SINGLE_SCRIPT through 25M (only bimodal pattern, no MULTIPLE_SCRIPTS) and
had lower peak scores (stoch best=107 vs 216 for baseline at 25M).
Continuing to 50M tests whether low scale suppresses the oscillation
entirely or just pushes it later.

Design:
  - CONTINUATION from ./models/PPO_128/final_model.zip (25M steps)
  - Same ProximityRewardWrapper(scale=0.025, max_distance=80)
  - Target: 50M total (25M additional)
  - Standard PPO: NatureCNN, ent_coef=0.006
"""
import os
import numpy as np
import glob
import cv2
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, NoopResetEnv, FireResetEnv, EpisodicLifeEnv
from autoreset_wrapper import AutoResetWrapper
from run_label_callback import RunLabelCallback
from proximity_reward_wrapper import ProximityRewardWrapper

import ale_py
gym.register_envs(ale_py)

RUN_NAME = "PPO_130"
TARGET_STEPS = 50_000_000
CHECKPOINT_PATH = f"./models/{RUN_NAME}/checkpoint"
RESUME_FROM = "./models/PPO_128/final_model.zip"

ENT_COEF = 0.006
PROXIMITY_SCALE = 0.025
PROXIMITY_MAX_DIST = 80.0
PROXIMITY_DESCEND_THRESHOLD = 100


class GrayscaleResize(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84):
        super().__init__(env)
        self._width = width
        self._height = height
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(height, width, 1), dtype=np.uint8)

    def observation(self, obs):
        if obs.ndim == 3 and obs.shape[2] == 3:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(obs, (self._width, self._height),
                             interpolation=cv2.INTER_AREA)
        return resized[:, :, None] if resized.ndim == 2 else resized


def linear_schedule(start: float, end: float):
    def schedule(progress_remaining: float) -> float:
        return end + (start - end) * progress_remaining
    return schedule


def get_latest_checkpoint(path):
    checkpoints = glob.glob(os.path.join(path, "latest_checkpoint_*_steps.zip"))
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getmtime)


def make_training_env():
    env = gym.make("ALE/Breakout-v5", frameskip=4, repeat_action_probability=0)
    env = NoopResetEnv(env, noop_max=30)
    env = FireResetEnv(env)
    env = EpisodicLifeEnv(env)
    env = GrayscaleResize(env, width=84, height=84)
    env = ClipRewardEnv(env)
    env = ProximityRewardWrapper(
        env, scale=PROXIMITY_SCALE, max_distance=PROXIMITY_MAX_DIST,
        descend_threshold=PROXIMITY_DESCEND_THRESHOLD,
    )
    env = Monitor(env)
    return env


def make_eval_env():
    env = gym.make("ALE/Breakout-v5", frameskip=4, repeat_action_probability=0)
    env = NoopResetEnv(env, noop_max=30)
    env = FireResetEnv(env)
    env = EpisodicLifeEnv(env)
    env = GrayscaleResize(env, width=84, height=84)
    env = ClipRewardEnv(env)
    env = Monitor(env)
    env = AutoResetWrapper(env)
    return env


if __name__ == "__main__":
    print(f"{RUN_NAME} — Continue PPO_128 (LOW scale=0.025) 25M -> 50M")
    print(f"  Resume from: {RESUME_FROM}")
    print(f"  Scale: {PROXIMITY_SCALE}, Max distance: {PROXIMITY_MAX_DIST}")
    print(f"  Training: ALE/Breakout-v5, fs=4, EpisodicLifeEnv + ProximityReward")
    print(f"  Eval: Standard Breakout (NO proximity reward) — transfer test")
    print(f"  Target: {TARGET_STEPS:,} total steps (25M additional)")
    print()

    os.makedirs(CHECKPOINT_PATH, exist_ok=True)

    env = DummyVecEnv([make_training_env for _ in range(32)])
    env = VecFrameStack(env, n_stack=4)

    eval_env = DummyVecEnv([make_eval_env])
    eval_env = VecFrameStack(eval_env, n_stack=4)

    eval_callback = EvalCallback(
        eval_env, best_model_save_path=f"./models/{RUN_NAME}",
        log_path=f"./logs/{RUN_NAME}", eval_freq=100_000,
        n_eval_episodes=50, deterministic=True, render=False, verbose=1)

    checkpoint_callback = CheckpointCallback(
        save_freq=156_250, save_path=CHECKPOINT_PATH,
        name_prefix="latest_checkpoint", save_replay_buffer=False, verbose=1)

    label_callback = RunLabelCallback(RUN_NAME)
    callbacks = CallbackList([eval_callback, checkpoint_callback, label_callback])

    print(f"Loading PPO_128 from {RESUME_FROM}...")
    model = PPO.load(RESUME_FROM, env=env, device="cuda")
    print(f"  Loaded model at {model.num_timesteps:,} steps")

    remaining = TARGET_STEPS - model.num_timesteps
    if remaining <= 0:
        print("Target already reached.")
    else:
        print(f"  Training for {remaining:,} more steps ({TARGET_STEPS:,} total)")
        model.learn(total_timesteps=remaining, callback=callbacks,
                    reset_num_timesteps=False, tb_log_name=RUN_NAME)

    model.save(f"./models/{RUN_NAME}/final_model")
    print(f"\n{RUN_NAME} complete at {model.num_timesteps:,} total steps.")
    env.close(); eval_env.close()
