"""
PPO_101 — Experiment 10: Life-Loss Penalty

BeamRider proved that hard failure constraints force reactive PPO policies.
Breakout's failure mode is soft — lose the ball, bricks stay broken, re-serve.
Scripts that break 3-5 bricks per life are locally optimal.

This experiment adds a negative reward on every life loss, making memorized
sweep scripts net-negative:
    Sweep script:  +4 (bricks) - 10 (penalty) = -6 net
    Do nothing:     0 reward
    Reactive play: +30 (bricks) - 10 (penalty) = +20 net

The penalty is annealed from 0→10 over 5M steps to avoid crushing exploration
before the agent learns basic gameplay.

Design:
  - LifeLossPenalty BEFORE EpisodicLifeEnv, AFTER ClipRewardEnv
    (so penalty isn't clipped to -1 by ClipRewardEnv)
  - Wrapper order: ALE → NoopReset → FireReset → EpisodicLife →
    GrayscaleResize → ClipRewardEnv → LifeLossPenalty → Monitor
  - Standard PPO: NatureCNN, ent_coef=0.006
  - Training: ALE/Breakout-v5, frameskip=1
  - Eval/Check: Clean ALE/Breakout-v5 (no penalty, standard wrappers)
  - Target: 50M steps
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
from life_loss_penalty import LifeLossPenalty
from run_label_callback import RunLabelCallback

import ale_py
gym.register_envs(ale_py)

RUN_NAME = "PPO_101"
TARGET_STEPS = 50_000_000
CHECKPOINT_PATH = f"./models/{RUN_NAME}/checkpoint"

PENALTY = 10.0
ANNEAL_STEPS = 5_000_000
ENT_COEF = 0.006
SEED = 101


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
    env = gym.make("ALE/Breakout-v5", frameskip=1, repeat_action_probability=0)
    # Standard Atari wrappers first (closest to ALE)
    env = NoopResetEnv(env, noop_max=30)
    env = FireResetEnv(env)
    env = EpisodicLifeEnv(env)
    env = GrayscaleResize(env, width=84, height=84)
    env = ClipRewardEnv(env)
    # LifeLossPenalty AFTER ClipRewardEnv so penalty isn't clipped to [-1,+1]
    env = LifeLossPenalty(env, penalty=PENALTY, anneal_steps=ANNEAL_STEPS)
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


def make_check_env():
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
    print(f"{RUN_NAME} -- Experiment 10: Life-Loss Penalty ({PENALTY}/life)")
    print(f"  Penalty: {PENALTY}/life loss, annealed over {ANNEAL_STEPS:,} steps")
    print(f"  Training: Clean ALE + life-loss penalty (after ClipRewardEnv)")
    print(f"  Eval/Check: Clean ALE (no penalty, standard wrappers)")
    print(f"  Hypothesis: scripts net-negative -> PPO forced toward reactive play")
    print()

    env = DummyVecEnv([make_training_env for _ in range(32)])
    env = VecFrameStack(env, n_stack=4)

    eval_env = DummyVecEnv([make_eval_env])
    eval_env = VecFrameStack(eval_env, n_stack=4)

    eval_callback = EvalCallback(
        eval_env, best_model_save_path=f"./models/{RUN_NAME}",
        log_path=f"./logs/{RUN_NAME}", eval_freq=50_000,
        n_eval_episodes=50, deterministic=True, render=False, verbose=1)

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000, save_path=CHECKPOINT_PATH,
        name_prefix="latest_checkpoint", save_replay_buffer=False, verbose=1)

    memorization_callback = MemorizationCheckCallback(
        run_name=RUN_NAME, sticky_actions=False, check_freq=1_000_000,
        n_games=20, make_env_fn=make_check_env, check_deterministic_false=True,
        summary_lines=[
            f"PPO_101 -- Experiment 10: Life-Loss Penalty ({PENALTY}/life)",
            f"Training: clean ALE + {PENALTY}/life penalty (annealed {ANNEAL_STEPS:,} steps)",
            f"Eval/Check: Clean ALE (no penalty)",
            f"Hypothesis: scripts net-negative -> forced toward reactive play",
            f"Policy: NatureCNN, ent_coef={ENT_COEF}",
        ])

    label_callback = RunLabelCallback(RUN_NAME)
    callbacks = CallbackList([eval_callback, checkpoint_callback, memorization_callback, label_callback])

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
