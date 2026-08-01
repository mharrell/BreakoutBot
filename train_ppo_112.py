"""
PPO_112 -- Experiment 21: Episode-Randomized Cursor Parameters (Variant A)

If cursor timing varies per episode, the policy cannot memorize a fixed
counter-strategy. approach_speed, push_magnitude, cooldown_frames,
warning_frames, tracking_threshold, and threat_radius are all randomized
from configurable distributions on every reset().

Hypothesis: unpredictable timing → policy must attend to actual cursor
state in real time → breaks SINGLE_SCRIPT ceiling.

Parameters:
  - approach_speed: log-uniform(1.0, 8.0)
  - push_magnitude: log-uniform(1.0, 16.0)
  - cooldown_frames: uniform_int(30, 150)
  - warning_frames: uniform_int(2, 12)
  - tracking_threshold: uniform(4, 20)
  - threat_radius: uniform(4, 20)
  cursor_size = 4 (fixed)

Design:
  - Training: ALE/Breakout-v5, fs=4, EpisodicLifeEnv
              + EpisodeRandomizedCursorWrapper
  - Eval/Check: Standard ALE/Breakout-v5 (NO wrapper) — test transfer
  - FROM SCRATCH (seed=112)
  - Target: 50M steps
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
from memorization_check_callback import MemorizationCheckCallback
from cursor_variants import EpisodeRandomizedCursorWrapper
from autoreset_wrapper import AutoResetWrapper
from run_label_callback import RunLabelCallback

import ale_py
gym.register_envs(ale_py)

RUN_NAME = "PPO_112"
TARGET_STEPS = 50_000_000
CHECKPOINT_PATH = f"./models/{RUN_NAME}/checkpoint"

ENT_COEF = 0.006
SEED = 112

CURSOR_PARAMS = dict(
    param_ranges={
        'approach_speed': (1.0, 8.0, 'log_uniform'),
        'push_magnitude': (1.0, 16.0, 'log_uniform'),
        'cooldown_frames': (30, 150, 'uniform_int'),
        'warning_frames': (2, 12, 'uniform_int'),
        'tracking_threshold': (4, 20, 'uniform'),
        'threat_radius': (4, 20, 'uniform'),
    },
    cursor_size=4,
)


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
    """Breakout WITH EpisodeRandomizedCursorWrapper, fs=4."""
    env = gym.make("ALE/Breakout-v5", frameskip=4, repeat_action_probability=0)
    env = NoopResetEnv(env, noop_max=30)
    env = FireResetEnv(env)
    env = EpisodeRandomizedCursorWrapper(env, **CURSOR_PARAMS)
    env = EpisodicLifeEnv(env)
    env = GrayscaleResize(env, width=84, height=84)
    env = ClipRewardEnv(env)
    env = Monitor(env)
    return env


def make_eval_env():
    """Standard Breakout WITHOUT cursor wrapper — test transfer."""
    env = gym.make("ALE/Breakout-v5", frameskip=4, repeat_action_probability=0)
    env = NoopResetEnv(env, noop_max=30)
    env = FireResetEnv(env)
    env = EpisodicLifeEnv(env)
    env = GrayscaleResize(env, width=84, height=84)
    env = ClipRewardEnv(env)
    env = Monitor(env)
    env = AutoResetWrapper(env)
    return env


def make_check_env():
    """Standard Breakout WITHOUT cursor wrapper — test transfer."""
    env = gym.make("ALE/Breakout-v5", frameskip=4, repeat_action_probability=0)
    env = NoopResetEnv(env, noop_max=30)
    env = FireResetEnv(env)
    env = EpisodicLifeEnv(env)
    env = GrayscaleResize(env, width=84, height=84)
    env = ClipRewardEnv(env)
    env = Monitor(env)
    env = AutoResetWrapper(env)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)
    return env


if __name__ == "__main__":
    ranges_str = '; '.join(f'{k}=[{lo},{hi}]' for k, (lo, hi, _)
                           in CURSOR_PARAMS['param_ranges'].items())
    print(f"{RUN_NAME} -- Experiment 21: Episode-Randomized Cursor (Variant A)")
    print(f"  Ranges: {ranges_str}")
    print(f"  Resample: every reset() (game start + life loss)")
    print(f"  Training: ALE/Breakout-v5, fs=4, EpisodicLifeEnv, EpisodeRandomizedCursorWrapper")
    print(f"  Eval/Check: Standard Breakout (NO cursor wrapper) — test transfer")
    print(f"  FROM SCRATCH (seed={SEED}), target {TARGET_STEPS:,} steps")
    print()

    env = DummyVecEnv([make_training_env for _ in range(32)])
    env = VecFrameStack(env, n_stack=4)

    eval_env = DummyVecEnv([make_eval_env])
    eval_env = VecFrameStack(eval_env, n_stack=4)

    eval_callback = EvalCallback(
        eval_env, best_model_save_path=f"./models/{RUN_NAME}",
        log_path=f"./logs/{RUN_NAME}", eval_freq=100_000,
        n_eval_episodes=50, deterministic=True, render=False, verbose=1)

    # save_freq empirically corresponds to steps/n_envs (not iterations).
    # save_freq=156,250 → saves every ~5M steps.
    checkpoint_callback = CheckpointCallback(
        save_freq=156_250, save_path=CHECKPOINT_PATH,
        name_prefix="latest_checkpoint", save_replay_buffer=False, verbose=1)

    memorization_callback = MemorizationCheckCallback(
        run_name=RUN_NAME, sticky_actions=False, check_freq=1_000_000,
        n_games=20, max_check_steps=5_000_000, max_steps_per_game=10_000,
        make_env_fn=make_check_env, check_deterministic_false=True,
        summary_lines=[
            f"PPO_112 -- Experiment 21: Episode-Randomized Cursor (Variant A)",
            f"Ranges: {ranges_str}",
            f"Resample: every reset() — 5 lives × 32 envs = ~160 param combos/episode",
            f"Training: ALE/Breakout-v5, fs=4, EpisodeRandomizedCursorWrapper",
            f"Eval/Check: Standard Breakout (no cursor) — test transfer",
            f"Hypothesis: unpredictable timing → real-time attention → breaks SINGLE_SCRIPT",
        ])

    label_callback = RunLabelCallback(RUN_NAME)
    callbacks = CallbackList([eval_callback, checkpoint_callback,
                              memorization_callback, label_callback])

    resume_path = get_latest_checkpoint(CHECKPOINT_PATH)
    if resume_path:
        print(f"Resuming {RUN_NAME} from {resume_path}...")
        model = PPO.load(resume_path, env=env, device="cuda")
        reset_num_timesteps = False
    else:
        print(f"Starting {RUN_NAME} from scratch (seed={SEED})...")
        model = PPO("CnnPolicy", env, verbose=1, device="cuda",
                    tensorboard_log=f"./tensorboard/{RUN_NAME}",
                    n_steps=128, batch_size=1024, n_epochs=4, gamma=0.99,
                    learning_rate=linear_schedule(2.5e-4, 1e-5),
                    clip_range=linear_schedule(0.2, 0.05),
                    ent_coef=ENT_COEF, vf_coef=0.5,
                    seed=SEED)
        reset_num_timesteps = True

    remaining = TARGET_STEPS - model.num_timesteps
    if remaining <= 0:
        print("Target already reached.")
    else:
        model.learn(total_timesteps=remaining, callback=callbacks,
                    reset_num_timesteps=reset_num_timesteps, tb_log_name=RUN_NAME)

    model.save(f"./models/{RUN_NAME}/final_model")
    print(f"\n{RUN_NAME} complete at {model.num_timesteps:,} total steps.")
    env.close(); eval_env.close()
