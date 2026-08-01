"""
PPO_116 -- Experiment 25: Randomized Brick Layouts During Training

Every model tested on the brick layout test showed the same pattern:
deterministic on full layout (unique=1), binary succeed/fail on half-bricks.
The policy memorizes a fixed paddle pattern that targets specific bricks.

This experiment trains on Breakout where the brick layout is DIFFERENT every
episode, randomized via setRAM(). If the policy can't predict which bricks
exist, it MUST track the ball to score. This directly attacks the
generalization gap revealed by the brick layout test.

Design:
  - Training: ALE/Breakout-v5, frameskip=4, EpisodicLifeEnv (5 lives)
              + AdversarialCursorWrapper (standard params, as PPO_107)
              + RandomizedBrickWrapper (new: randomizes brick layout per reset)
  - Eval/Check: Standard Breakout (NO wrapper) -- test transfer
  - FROM SCRATCH (seed=116)
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
from adversarial_cursor_wrapper import AdversarialCursorWrapper
from autoreset_wrapper import AutoResetWrapper
from run_label_callback import RunLabelCallback

import ale_py
gym.register_envs(ale_py)

RUN_NAME = "PPO_116"
TARGET_STEPS = 50_000_000
CHECKPOINT_PATH = f"./models/{RUN_NAME}/checkpoint"

ENT_COEF = 0.006
SEED = 116

# Standard cursor params (PPO_107 baseline)
CURSOR_PARAMS = dict(
    approach_speed=2.0,
    tracking_threshold=8,
    threat_radius=8,
    warning_frames=5,
    push_magnitude=4.0,
    cooldown_frames=60,
    cursor_size=4,
)

# Brick RAM: 36 bytes (0-35), bit-packed playfield registers.
# We randomly zero out portions of the brick RAM at each reset.
BRICK_RAM_START = 0
BRICK_RAM_END = 36


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


class RandomizedBrickWrapper(gym.Wrapper):
    """Randomize brick layout at every reset().

    Strategies (randomly chosen per reset):
      - "full":       no change, standard layout
      - "right_half": clear RAM[0-17]  → right-side bricks removed
      - "left_half":  clear RAM[18-35] → left-side bricks removed
      - "sparse":     clear 50% of brick bytes randomly
      - "bands":      clear alternating rows

    This prevents the policy from memorizing any specific brick layout.
    """

    STRATEGIES = ["full", "right_half", "left_half", "sparse", "bands"]
    # "full" has weight 2 so the standard layout appears often enough
    # for the policy to learn brick-breaking at all
    WEIGHTS = [2, 1, 1, 1, 1]

    def __init__(self, env, seed=None):
        super().__init__(env)
        self._rng = np.random.default_rng(seed)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        strategy = self._rng.choice(self.STRATEGIES, p=np.array(self.WEIGHTS) / sum(self.WEIGHTS))
        ram = self.unwrapped.ale.getRAM()

        if strategy == "right_half":
            for addr in range(0, 18):
                self.unwrapped.ale.setRAM(addr, 0)
        elif strategy == "left_half":
            for addr in range(18, 36):
                self.unwrapped.ale.setRAM(addr, 0)
        elif strategy == "sparse":
            # Randomly zero 40-60% of brick bytes
            addrs = list(range(36))
            self._rng.shuffle(addrs)
            n_clear = self._rng.integers(14, 22)  # ~40-60%
            for addr in addrs[:n_clear]:
                self.unwrapped.ale.setRAM(addr, 0)
        elif strategy == "bands":
            # Clear alternating pairs of rows (0-1, 4-5, etc.)
            for row_start in [0, 4]:  # rows 0-1 and 4-5
                for addr in range(row_start * 6, (row_start + 2) * 6):
                    self.unwrapped.ale.setRAM(addr, 0)
        # "full": no change

        return obs, info


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
    """Breakout WITH cursor wrapper + randomized bricks."""
    env = gym.make("ALE/Breakout-v5", frameskip=4, repeat_action_probability=0)
    env = NoopResetEnv(env, noop_max=30)
    env = FireResetEnv(env)
    env = RandomizedBrickWrapper(env, seed=SEED)
    env = AdversarialCursorWrapper(env, **CURSOR_PARAMS)
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
    print(f"{RUN_NAME} -- Experiment 25: Randomized Brick Layouts")
    print(f"  Cursor params: {CURSOR_PARAMS}")
    print(f"  Brick strategies: {RandomizedBrickWrapper.STRATEGIES}")
    print(f"  Training: ALE/Breakout-v5, fs=4, EpisodicLifeEnv, varied brick layouts")
    print(f"  Eval/Check: Standard Breakout (NO cursor, standard bricks)")
    print(f"  FROM SCRATCH (seed={SEED}), target {TARGET_STEPS:,} steps")
    print(f"  Checkpoints: every ~5M steps")
    print()

    env = DummyVecEnv([make_training_env for _ in range(32)])
    env = VecFrameStack(env, n_stack=4)

    eval_env = DummyVecEnv([make_eval_env])
    eval_env = VecFrameStack(eval_env, n_stack=4)

    eval_callback = EvalCallback(
        eval_env, best_model_save_path=f"./models/{RUN_NAME}",
        log_path=f"./logs/{RUN_NAME}", eval_freq=100_000,
        n_eval_episodes=50, deterministic=True, render=False, verbose=1)

    # save_freq=156_250 → ~5M step intervals
    checkpoint_callback = CheckpointCallback(
        save_freq=156_250, save_path=CHECKPOINT_PATH,
        name_prefix="latest_checkpoint", save_replay_buffer=False, verbose=1)

    memorization_callback = MemorizationCheckCallback(
        run_name=RUN_NAME, sticky_actions=False, check_freq=1_000_000,
        n_games=20, max_check_steps=5_000_000, max_steps_per_game=10_000,
        make_env_fn=make_check_env, check_deterministic_false=True,
        summary_lines=[
            f"PPO_116 -- Experiment 25: Randomized Brick Layouts",
            f"Cursor: standard params (PPO_107 baseline)",
            f"Brick strategies: {RandomizedBrickWrapper.STRATEGIES}",
            f"Training: ALE/Breakout-v5, fs=4, AdversarialCursorWrapper + RandomizedBrickWrapper",
            f"Eval/Check: Standard Breakout (no cursor, standard bricks)",
            f"Key: can't memorize bricks → MUST track ball",
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
