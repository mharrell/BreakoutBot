"""
PPO_119 -- Experiment 27: Trajectory Entropy (Objective Function Change #2)

After 120 experiments, no PPO model has ever genuinely generalized. Every
environment wrapper and perturbation was defeated by the same fundamental
problem: PPO maximizes E[Σ game_reward], and in deterministic environments
the expected-return-maximizing policy IS a memorized script.

This experiment changes the OBJECTIVE, not the environment.

TrajectoryEntropyWrapper adds a per-step bonus: bonus = scale × (1 - p(action))
where p(action) is the fraction of parallel envs that took the same action.
- Script: all 32 envs take same action → p=1.0 → zero bonus
- Reactive: different ball positions → different actions → p<1.0 → positive bonus

This directly attacks the defining property of a script: identical actions at
identical timesteps across episodes. A script CANNOT earn the entropy bonus
because by definition, every env takes the same action at the same step.

PPO now maximizes: E[Σ game_reward + Σ trajectory_bonus]

Key test: eval on CLEAN Breakout (no wrapper). If trajectory entropy baked
diverse action-selection into the policy, the diversity persists on eval.

Design:
  - Training: ALE/Breakout-v5, fs=4, EpisodicLifeEnv (5 lives)
              + TrajectoryEntropyWrapper(scale=0.01) at VecEnv level
  - Eval/Check: Standard Breakout (NO entropy wrapper) -- transfer test
  - FROM SCRATCH (seed=119)
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
from trajectory_entropy_wrapper import TrajectoryEntropyWrapper

import ale_py
gym.register_envs(ale_py)

RUN_NAME = "PPO_119"
TARGET_STEPS = 25_000_000
CHECKPOINT_PATH = f"./models/{RUN_NAME}/checkpoint"

ENT_COEF = 0.006
SEED = 119

# Trajectory entropy -- bonus for taking different actions across parallel envs
ENTROPY_SCALE = 0.01


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
    """Standard Breakout. TrajectoryEntropyWrapper added at VecEnv level."""
    env = gym.make("ALE/Breakout-v5", frameskip=4, repeat_action_probability=0)
    env = NoopResetEnv(env, noop_max=30)
    env = FireResetEnv(env)
    env = EpisodicLifeEnv(env)
    env = GrayscaleResize(env, width=84, height=84)
    env = ClipRewardEnv(env)
    env = Monitor(env)
    return env


def make_eval_env():
    """Standard Breakout WITHOUT entropy wrapper -- transfer test."""
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
    """Standard Breakout WITHOUT entropy wrapper -- memcheck on clean physics."""
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
    print(f"{RUN_NAME} -- Experiment 27: Trajectory Entropy")
    print(f"  Mechanism: cross-env action-diversity bonus")
    print(f"  Scale: {ENTROPY_SCALE} (max bonus per step)")
    print(f"  Formula: bonus = {ENTROPY_SCALE} × (1 - p(action_across_envs))")
    print(f"  Training: ALE/Breakout-v5, fs=4, EpisodicLifeEnv")
    print(f"           + TrajectoryEntropyWrapper at VecEnv level")
    print(f"  Eval/Check: Standard Breakout (NO wrapper) -- transfer test")
    print(f"  FROM SCRATCH (seed={SEED}), target {TARGET_STEPS:,} steps")
    print(f"  Entropy coef: {ENT_COEF}")
    print(f"  Hypothesis: trajectory entropy forces argmax action diversity")
    print()

    # Build VecEnv: envs → FrameStack → TrajectoryEntropy
    env = DummyVecEnv([make_training_env for _ in range(32)])
    env = VecFrameStack(env, n_stack=4)
    env = TrajectoryEntropyWrapper(env, entropy_scale=ENTROPY_SCALE)

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
            f"PPO_119 -- Experiment 27: Trajectory Entropy",
            f"Mechanism: cross-env action-diversity bonus (scale={ENTROPY_SCALE})",
            f"Training: ALE/Breakout-v5 + TrajectoryEntropyWrapper",
            f"Eval/Check: Standard Breakout (NO wrapper) -- transfer test",
            f"Hypothesis: trajectory entropy forces argmax action diversity",
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
