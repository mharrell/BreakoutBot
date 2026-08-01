"""
PPO_117 -- Experiment 26: Randomized + Multi-Cursor Combined

PPO_112 (randomized params) and PPO_114 (multi-cursor) both independently
achieved ~41% reversal with 8-score memcheck diversity. This experiment
combines both mechanisms:

  - Two asymmetric cursors (like PPO_114): fast/light left + slow/heavy right
  - Per-episode parameter randomization (like PPO_112): each cursor's
    speed, push, cooldown, warning, threshold, and threat_radius are
    sampled from distributions on every reset()

The hypothesis: unpredictable adversaries (multi-cursor) that are themselves
unpredictable (randomized params) create a threat landscape that no single
script can satisfy. This should push reversal above the ~41% ceiling.

Design:
  - Training: ALE/Breakout-v5, frameskip=4, EpisodicLifeEnv (5 lives)
              + RandomizedMultiCursorWrapper (2 cursors, params per reset)
  - Eval/Check: Standard ALE/Breakout-v5 (NO wrapper) -- test transfer
  - FROM SCRATCH (seed=117)
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
from cursor_variants import CursorAgent, _draw_cursor_on_obs
from autoreset_wrapper import AutoResetWrapper
from run_label_callback import RunLabelCallback

import ale_py
gym.register_envs(ale_py)

RUN_NAME = "PPO_117"
TARGET_STEPS = 50_000_000
CHECKPOINT_PATH = f"./models/{RUN_NAME}/checkpoint"

ENT_COEF = 0.006
SEED = 117

# Known RAM map
BALL_X, BALL_Y, PADDLE_X = 99, 101, 72


# ---------------------------------------------------------------------------
# Randomized Multi-Cursor Wrapper
# ---------------------------------------------------------------------------

class RandomizedMultiCursorWrapper(gym.Wrapper):
    """Two asymmetric cursors with per-episode randomized parameters.

    Follows the MultiCursorWrapper pattern (post-step): call env.step(),
    then apply cursor logic to the resulting observation.

    Cursor A (left-spawn): fast, light push, tight tracking
    Cursor B (right-spawn): slow, heavy push, loose tracking

    Total push capped at 20px/step. Cursors update independently with
    ball_x re-read between cursor updates.
    """

    CURSOR_SIZE = 4
    MAX_TOTAL_PUSH = 20.0
    BALL_X_ADDR = 99
    BALL_Y_ADDR = 101
    PADDLE_X_ADDR = 72
    BALL_DIR_ADDR = 105
    MIN_X = 8
    MAX_X = 152

    def __init__(self, env, seed=None):
        super().__init__(env)
        self._rng = np.random.RandomState(seed)  # CursorAgent requires RandomState
        self.cursors = []

    def _sample_params_a(self):
        return dict(
            approach_speed=float(2 ** self._rng.uniform(1.0, np.log2(10.0))),
            tracking_threshold=float(self._rng.randint(3, 9)),
            threat_radius=float(self._rng.randint(3, 9)),
            warning_frames=int(self._rng.randint(2, 7)),
            push_magnitude=float(2 ** self._rng.uniform(0.0, np.log2(6.0))),
            cooldown_frames=int(self._rng.randint(20, 81)),
        )

    def _sample_params_b(self):
        return dict(
            approach_speed=float(2 ** self._rng.uniform(np.log2(0.75), 2.0)),
            tracking_threshold=float(self._rng.randint(8, 25)),
            threat_radius=float(self._rng.randint(8, 25)),
            warning_frames=int(self._rng.randint(4, 15)),
            push_magnitude=float(2 ** self._rng.uniform(2.0, np.log2(20.0))),
            cooldown_frames=int(self._rng.randint(40, 161)),
        )

    def _read_ram(self):
        ram = self.env.unwrapped.ale.getRAM()
        return (int(ram[self.BALL_X_ADDR]),
                int(ram[self.BALL_Y_ADDR]),
                int(ram[self.PADDLE_X_ADDR]))

    def _clamp_x(self, x):
        return max(self.MIN_X, min(self.MAX_X, int(x)))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Create fresh cursor agents with newly sampled params
        params_a = self._sample_params_a()
        params_b = self._sample_params_b()
        self.cursors = [
            CursorAgent(self._rng, params_a, agent_id=0, spawn_side="left"),
            CursorAgent(self._rng, params_b, agent_id=1, spawn_side="right"),
        ]

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        ball_x, ball_y, paddle_x = self._read_ram()
        total_push = 0.0

        for cursor in self.cursors:
            tracking = abs(ball_x - paddle_x) <= cursor.tracking_threshold
            result = cursor.update(ball_x, ball_y, paddle_x, tracking)

            if result['did_attack']:
                push_dir = 1 if ball_x >= paddle_x else -1
                push_amount = push_dir * cursor.push_magnitude

                remaining = self.MAX_TOTAL_PUSH - abs(total_push)
                if abs(push_amount) > remaining:
                    push_amount = np.sign(push_amount) * remaining

                new_ball_x = self._clamp_x(ball_x + total_push + push_amount)
                self.env.unwrapped.ale.setRAM(self.BALL_X_ADDR, new_ball_x)
                new_dir = 255 if push_dir > 0 else 1
                self.env.unwrapped.ale.setRAM(self.BALL_DIR_ADDR, new_dir)
                total_push += push_amount

                ball_x = int(self.env.unwrapped.ale.getRAM()[self.BALL_X_ADDR])

            if result['is_visible'] and result['brightness'] is not None:
                _draw_cursor_on_obs(obs, cursor.cursor_x, cursor.cursor_y,
                                    self.CURSOR_SIZE, result['brightness'])

        return obs, reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# Standard wrappers
# ---------------------------------------------------------------------------

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
    """Breakout WITH randomized multi-cursor wrapper."""
    env = gym.make("ALE/Breakout-v5", frameskip=4, repeat_action_probability=0)
    env = NoopResetEnv(env, noop_max=30)
    env = FireResetEnv(env)
    env = RandomizedMultiCursorWrapper(env, seed=SEED)
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
    print(f"{RUN_NAME} -- Experiment 26: Randomized + Multi-Cursor Combined")
    print(f"  2 asymmetric cursors with per-episode randomized parameters")
    print(f"  Cursor A (left):  fast/light, tight tracking")
    print(f"  Cursor B (right): slow/heavy, loose tracking")
    print(f"  Training: ALE/Breakout-v5, fs=4, EpisodicLifeEnv, RandomizedMultiCursorWrapper")
    print(f"  Eval/Check: Standard Breakout (NO cursor) -- test transfer")
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

    # save_freq=156_250 -> ~5M step intervals
    checkpoint_callback = CheckpointCallback(
        save_freq=156_250, save_path=CHECKPOINT_PATH,
        name_prefix="latest_checkpoint", save_replay_buffer=False, verbose=1)

    memorization_callback = MemorizationCheckCallback(
        run_name=RUN_NAME, sticky_actions=False, check_freq=1_000_000,
        n_games=20, max_check_steps=5_000_000, max_steps_per_game=10_000,
        make_env_fn=make_check_env, check_deterministic_false=True,
        summary_lines=[
            f"PPO_117 -- Experiment 26: Randomized + Multi-Cursor Combined",
            f"2 asymmetric cursors with per-episode randomized parameters",
            f"Cursor A: fast/light (speed~3-10, push~1-6, threshold~3-8)",
            f"Cursor B: slow/heavy (speed~1-4, push~4-20, threshold~8-24)",
            f"Training: ALE/Breakout-v5, fs=4, RandomizedMultiCursorWrapper",
            f"Eval/Check: Standard Breakout (no cursor) -- test transfer",
            f"Key: unpredictable threats from unpredictable adversaries",
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
