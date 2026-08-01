"""
PPO_114 -- Experiment 23: Multiple Independent Cursors (Variant C)

Two asymmetric cursor adversaries with independent state machines:
  Cursor A (fast/light): speed=5, push=2, threshold=4, warning=3, cooldown=40
    Spawns left side. Tight tracking requirement. Quick, annoying attacks.
  Cursor B (slow/heavy): speed=1.5, push=8, threshold=16, warning=8, cooldown=80
    Spawns right side. Loose tracking requirement. Slow, devastating attacks.

Parallel threat timelines cannot be simultaneously satisfied by a single
scripted paddle response. The policy must trade off between:
  - Tracking tightly enough for Cursor A (threshold=4 → within 4px of ball)
  - Never ignoring the ball long enough for Cursor B to detonate (8-frame warning)
  - Managing two independent approach/cooldown cycles

Hypothesis: two adversaries with different demands → policy must genuinely
attend to the ball AND both cursor states → breaks SINGLE_SCRIPT ceiling.

Design:
  - Training: ALE/Breakout-v5, fs=4, EpisodicLifeEnv
              + MultiCursorWrapper (2 asymmetric cursors)
  - Eval/Check: Standard ALE/Breakout-v5 (NO wrapper) — test transfer
  - FROM SCRATCH (seed=114)
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
from cursor_variants import MultiCursorWrapper
from autoreset_wrapper import AutoResetWrapper
from run_label_callback import RunLabelCallback

import ale_py
gym.register_envs(ale_py)

RUN_NAME = "PPO_114"
TARGET_STEPS = 50_000_000
CHECKPOINT_PATH = f"./models/{RUN_NAME}/checkpoint"

ENT_COEF = 0.006
SEED = 114

# Two asymmetric cursors:
#   A: fast/light — approach_speed=5, push=2, tracking_threshold=4, warning=3, cooldown=40
#   B: slow/heavy — approach_speed=1.5, push=8, tracking_threshold=16, warning=8, cooldown=80
CURSOR_CONFIGS = [
    {
        'approach_speed': 5.0,
        'tracking_threshold': 4,
        'threat_radius': 8,
        'warning_frames': 3,
        'push_magnitude': 2.0,
        'cooldown_frames': 40,
        'cursor_size': 4,
    },
    {
        'approach_speed': 1.5,
        'tracking_threshold': 16,
        'threat_radius': 8,
        'warning_frames': 8,
        'push_magnitude': 8.0,
        'cooldown_frames': 80,
        'cursor_size': 4,
    },
]


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
    """Breakout WITH MultiCursorWrapper, fs=4."""
    env = gym.make("ALE/Breakout-v5", frameskip=4, repeat_action_probability=0)
    env = NoopResetEnv(env, noop_max=30)
    env = FireResetEnv(env)
    env = MultiCursorWrapper(env, cursor_configs=CURSOR_CONFIGS)
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
    print(f"{RUN_NAME} -- Experiment 23: Multiple Independent Cursors (Variant C)")
    print(f"  Cursor A (fast/light): speed=5.0, push=2.0, threshold=4, warning=3, cooldown=40")
    print(f"    Spawns left side. Tight tracking. Quick, annoying attacks.")
    print(f"  Cursor B (slow/heavy): speed=1.5, push=8.0, threshold=16, warning=8, cooldown=80")
    print(f"    Spawns right side. Loose tracking. Slow, devastating attacks.")
    print(f"  Attack combination: independent (capped at 20px/step total)")
    print(f"  Training: ALE/Breakout-v5, fs=4, EpisodicLifeEnv, MultiCursorWrapper")
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
            f"PPO_114 -- Experiment 23: Multiple Independent Cursors (Variant C)",
            f"Cursor A: speed=5.0, push=2.0, threshold=4, warning=3, cooldown=40 (left)",
            f"Cursor B: speed=1.5, push=8.0, threshold=16, warning=8, cooldown=80 (right)",
            f"Training: ALE/Breakout-v5, fs=4, MultiCursorWrapper (2 asymmetric cursors)",
            f"Eval/Check: Standard Breakout (no cursor) — test transfer",
            f"Hypothesis: parallel threats → no single script works → breaks SINGLE_SCRIPT",
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
