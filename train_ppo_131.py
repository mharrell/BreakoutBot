"""
PPO_131 — Experiment 35a: Proximity Reward Linear Fading

Scale decays linearly from 0.05 to 0.0 over 25M steps. Tests whether
establishing ball-tracking early (when scale is high) then removing the
proximity bonus prevents the oscillation between reactive and script basins.

Hypothesis: early tracking gets baked into the policy structure before the
game-reward gradient can pull it into a script basin. Without the proximity
bonus in later training, the model focuses purely on brick-clearing while
retaining the tracking behavior it learned early.

Design:
  - Training: ALE/Breakout-v5 + FadingProximityRewardWrapper(0.05→0.0)
  - Eval/Check: Standard Breakout (NO proximity reward) — transfer test
  - FROM SCRATCH (seed=131)
  - Target: 25M steps
  - Scale: linear_schedule(0.05, 0.0) via progress_remaining
  - Standard PPO: NatureCNN, ent_coef=0.006
"""
import os
import numpy as np
import glob
import cv2
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, NoopResetEnv, FireResetEnv, EpisodicLifeEnv
from memorization_check_callback import MemorizationCheckCallback
from autoreset_wrapper import AutoResetWrapper
from run_label_callback import RunLabelCallback
from fading_proximity_wrapper import FadingProximityRewardWrapper

import ale_py
gym.register_envs(ale_py)

RUN_NAME = "PPO_131"
TARGET_STEPS = 25_000_000
CHECKPOINT_PATH = f"./models/{RUN_NAME}/checkpoint"

ENT_COEF = 0.006
SEED = 131

PROXIMITY_MAX_DIST = 80.0
PROXIMITY_DESCEND_THRESHOLD = 100
INITIAL_SCALE = 0.05
FINAL_SCALE = 0.0


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


class ScaleUpdateCallback(BaseCallback):
    """Updates the wrapper's progress_remaining on each rollout end."""
    def __init__(self, target_steps):
        super().__init__()
        self._target = target_steps

    def _on_step(self):
        progress = 1.0 - (self.model.num_timesteps / self._target)
        progress = max(0.0, min(1.0, progress))
        # Update all envs in the VecEnv
        env = self.model.get_env()
        if hasattr(env, 'envs'):
            for e in env.envs:
                self._update_wrappers(e, progress)
        elif hasattr(env, 'unwrapped'):
            self._update_wrappers(env, progress)
        return True

    def _update_wrappers(self, env, progress):
        while env is not None:
            if isinstance(env, FadingProximityRewardWrapper):
                env.progress_remaining = progress
            env = getattr(env, 'env', None)


def make_training_env():
    """Breakout + annealing proximity reward."""
    env = gym.make("ALE/Breakout-v5", frameskip=4, repeat_action_probability=0)
    env = NoopResetEnv(env, noop_max=30)
    env = FireResetEnv(env)
    env = EpisodicLifeEnv(env)
    env = GrayscaleResize(env, width=84, height=84)
    env = ClipRewardEnv(env)
    scale_fn = lambda p: FINAL_SCALE + (INITIAL_SCALE - FINAL_SCALE) * p
    env = FadingProximityRewardWrapper(
        env, scale_schedule=scale_fn,
        max_distance=PROXIMITY_MAX_DIST,
        descend_threshold=PROXIMITY_DESCEND_THRESHOLD,
    )
    env = Monitor(env)
    return env


def make_eval_env():
    """Standard Breakout WITHOUT proximity reward — transfer test."""
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
    """Standard Breakout WITHOUT proximity reward."""
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
    print(f"{RUN_NAME} — Experiment 35a: Proximity Reward Linear Fading")
    print(f"  Scale: {INITIAL_SCALE} -> {FINAL_SCALE} over {TARGET_STEPS:,} steps")
    print(f"  Max distance: {PROXIMITY_MAX_DIST}, Threshold: ball_y > {PROXIMITY_DESCEND_THRESHOLD}")
    print(f"  Training: ALE/Breakout-v5 + FadingProximityRewardWrapper")
    print(f"  Eval/Check: Standard Breakout (NO proximity reward)")
    print(f"  FROM SCRATCH (seed={SEED}), target {TARGET_STEPS:,} steps")
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

    memorization_callback = MemorizationCheckCallback(
        run_name=RUN_NAME, sticky_actions=False, check_freq=1_000_000,
        n_games=20, max_check_steps=5_000_000, max_steps_per_game=10_000,
        make_env_fn=make_check_env, check_deterministic_false=True,
        summary_lines=[
            f"PPO_131 — Experiment 35a: Proximity Reward Linear Fading",
            f"Scale: {INITIAL_SCALE} -> {FINAL_SCALE} over {TARGET_STEPS:,} steps",
            f"Training: ALE/Breakout-v5 + FadingProximityRewardWrapper",
            f"Eval/Check: Standard Breakout (NO proximity reward)",
        ])

    scale_callback = ScaleUpdateCallback(TARGET_STEPS)
    label_callback = RunLabelCallback(RUN_NAME)
    callbacks = CallbackList([eval_callback, checkpoint_callback,
                              memorization_callback, label_callback, scale_callback])

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
