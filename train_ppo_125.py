"""
PPO_125 -- Experiment 32: Randomized Brick Pre-Clearing

Every approach so far either changes the environment dynamically (bumpers)
or modifies the reward function (entropy, proximity). PPO_125 takes a
different angle: randomize the INITIAL state.

At each reset, 15-25 random bricks are cleared from the wall. Each of the
32 parallel envs gets a different pattern of holes. A script targeting
specific brick positions fails on episodes where those bricks are gone.
The policy sees the holes in the observation and must adapt its targeting
to the specific layout it's given.

Uses 1-life training (no EpisodicLifeEnv) for more frequent resets:
  - Standard: ~5M frames / 5000 fps = 1000 episodes = 1000 layouts
  - 1-life: same frames = 5000 episodes = 5000 layouts
5x more layout variation per training step.

Unlike per-episode dynamics randomization (PPO_33), the brick layout IS the
visual observation — the policy can't "condition and ignore" because the
layout IS what it must interact with. To score, it must see where the bricks
are and aim accordingly.

Key test: eval on CLEAN Breakout (full 36-brick wall, EpisodicLifeEnv).
Can the policy trained on randomly-depleted walls adapt to a full wall?
If it learned visual reactivity, it should perform well on any layout.

Design:
  - Training: ALE/Breakout-v5, fs=4, NO EpisodicLifeEnv (1-life episodes)
              + BrickPreclearWrapper(min_clear=15, max_clear=25)
  - Eval/Check: Standard Breakout WITH EpisodicLifeEnv (5 lives, full wall)
  - FROM SCRATCH (seed=125)
  - Target: 25M steps
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
from autoreset_wrapper import AutoResetWrapper
from run_label_callback import RunLabelCallback
from brick_preclear_wrapper import BrickPreclearWrapper

import ale_py
gym.register_envs(ale_py)

RUN_NAME = "PPO_125"
TARGET_STEPS = 25_000_000
CHECKPOINT_PATH = f"./models/{RUN_NAME}/checkpoint"

ENT_COEF = 0.006
SEED = 125

# Brick pre-clearing
MIN_CLEAR = 15
MAX_CLEAR = 25


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
    """Breakout + brick pre-clearing. 1-life episodes (no EpisodicLifeEnv)."""
    env = gym.make("ALE/Breakout-v5", frameskip=4, repeat_action_probability=0)
    env = NoopResetEnv(env, noop_max=30)
    env = FireResetEnv(env)
    # NO EpisodicLifeEnv — 1-life episodes = more frequent pre-clearing
    # BrickPreclearWrapper MUST be before GrayscaleResize so the NOOP
    # step's observation goes through the full processing chain.
    env = BrickPreclearWrapper(env, min_clear=MIN_CLEAR, max_clear=MAX_CLEAR)
    env = GrayscaleResize(env, width=84, height=84)
    env = ClipRewardEnv(env)
    env = Monitor(env)
    return env


def make_eval_env():
    """Standard Breakout WITH EpisodicLifeEnv, full wall -- transfer test."""
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
    """Standard Breakout WITH EpisodicLifeEnv -- memcheck on full wall."""
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
    print(f"{RUN_NAME} -- Experiment 32: Randomized Brick Pre-Clearing")
    print(f"  Clear range: {MIN_CLEAR}-{MAX_CLEAR} bricks per reset")
    print(f"  Training: ALE/Breakout-v5, fs=4, 1-life (NO EpisodicLifeEnv)")
    print(f"           + BrickPreclearWrapper")
    print(f"  Eval/Check: Standard Breakout WITH EpisodicLifeEnv, full wall")
    print(f"  FROM SCRATCH (seed={SEED}), target {TARGET_STEPS:,} steps")
    print(f"  Entropy coef: {ENT_COEF}")
    print()

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

    memorization_callback = MemorizationCheckCallback(
        run_name=RUN_NAME, sticky_actions=False, check_freq=1_000_000,
        n_games=20, max_check_steps=5_000_000, max_steps_per_game=10_000,
        make_env_fn=make_check_env, check_deterministic_false=True,
        summary_lines=[
            f"PPO_125 -- Experiment 32: Randomized Brick Pre-Clearing",
            f"Clear: {MIN_CLEAR}-{MAX_CLEAR} bricks per reset",
            f"Training: ALE/Breakout-v5, 1-life + BrickPreclearWrapper",
            f"Eval/Check: Standard Breakout, 5-life, FULL wall -- transfer test",
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
