"""
PPO_57b — MODERATE ENTROPY CONTINUATION (ent_coef: 0.006 → 0.02, 3.33×)

================================================================================
ENTROPY INTERVENTION EXPERIMENT — 3 variants from PPO_55's 10M checkpoint

The det=False diversity peak for PPO_55 was at 10M (9 unique, avg 11.6,
best 16), independently confirmed by PPO_57's 10M peak (10 unique, avg 11.2,
best 17). Checkpoints save every 3.2M steps, so 9.6M is the closest available
checkpoint to the peak (on the rising edge: 8 unique, avg 8.4 at 9M).
The det=True argmax collapses by 11-16M in all replicates. These three
variants test whether raising ent_coef before the collapse prevents it.

VARIANTS (all from PPO_55 step 10M, all Y-only, cooldown=30, prob=10%):
  PPO_55a: ent_coef = 0.010  (1.67×, mild)
  PPO_55b: ent_coef = 0.020  (3.33×, moderate)    ← THIS SCRIPT
  PPO_55c: ent_coef = 0.040  (6.67×, aggressive)

CONTROLS (ent_coef = 0.006 fixed):
  PPO_55, PPO_57, PPO_58 — Y-only replicates

If mild works: we learn the minimum entropy pressure needed.
If only aggressive works: the optimizer strongly prefers collapse.
If none work: entropy alone doesn't prevent argmax collapse.
================================================================================
"""
import os
import glob
import cv2
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, NoopResetEnv, FireResetEnv, EpisodicLifeEnv
from memorization_check_callback import MemorizationCheckCallback
from ale_breakout_y_perturb import ALEBreakoutYPerturb
from autoreset_wrapper import AutoResetWrapper

import ale_py
gym.register_envs(ale_py)

RUN_NAME = "PPO_57b"
SOURCE_RUN = "PPO_57"
SOURCE_CHECKPOINT_STEP = 9_600_000
ENT_COEF_PRE = 0.006
ENT_COEF_POST = 0.020    # moderate — 3.33× increase
TARGET_STEPS = 50_000_000
PERTURB_COOLDOWN = 30
PERTURB_PROB = 0.10
PERTURB_RANGE = 8

CHECKPOINT_PATH = f"./models/{RUN_NAME}/checkpoint"
SOURCE_CHECKPOINT_PATH = f"./models/{SOURCE_RUN}/checkpoint"


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


# ---------------------------------------------------------------------------
# Environment builders (identical to PPO_55)
# ---------------------------------------------------------------------------

def make_training_env():
    env = gym.make("ALE/Breakout-v5", frameskip=1, repeat_action_probability=0)
    env = NoopResetEnv(env, noop_max=30)
    env = ALEBreakoutYPerturb(env, cooldown_frames=PERTURB_COOLDOWN,
                              perturb_prob=PERTURB_PROB, perturb_range=PERTURB_RANGE)
    env = FireResetEnv(env)
    env = EpisodicLifeEnv(env)
    env = GrayscaleResize(env, width=84, height=84)
    env = ClipRewardEnv(env)
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
    # Check for own checkpoint first (resume), fall back to source (fresh start)
    own_checkpoint = get_latest_checkpoint(CHECKPOINT_PATH)
    if own_checkpoint:
        checkpoint_file = own_checkpoint
        fresh_start = False
    else:
        checkpoint_file = os.path.join(
            SOURCE_CHECKPOINT_PATH,
            f"latest_checkpoint_{SOURCE_CHECKPOINT_STEP}_steps.zip"
        )
        fresh_start = True

    if not os.path.exists(checkpoint_file):
        print(f"ERROR: Checkpoint not found: {checkpoint_file}")
        if fresh_start:
            available = sorted(glob.glob(os.path.join(SOURCE_CHECKPOINT_PATH, "*.zip")))
            print(f"Available checkpoints in {SOURCE_CHECKPOINT_PATH}:")
        else:
            available = sorted(glob.glob(os.path.join(CHECKPOINT_PATH, "*.zip")))
            print(f"Available checkpoints in {CHECKPOINT_PATH}:")
        for f in available:
            print(f"  {os.path.basename(f)}")
        exit(1)

    label = {0.010: "MILD", 0.020: "MODERATE", 0.040: "AGGRESSIVE"}.get(ENT_COEF_POST, "CUSTOM")
    mult = ENT_COEF_POST / ENT_COEF_PRE

    print("=" * 80)
    print(f"{RUN_NAME} — {label} ENTROPY CONTINUATION ({ENT_COEF_PRE} → {ENT_COEF_POST}, {mult:.1f}×)")
    print("=" * 80)
    print(f"  Source:         {SOURCE_RUN} @ {SOURCE_CHECKPOINT_STEP//1_000_000}M steps")
    print(f"  Entropy before: {ENT_COEF_PRE}")
    print(f"  Entropy after:  {ENT_COEF_POST} ({mult:.1f}×)")
    print(f"  Variants:       55a={0.010} (mild)  55b={0.020} (moderate)  55c={0.040} (aggressive)")
    print(f"  Controls:       PPO_55, PPO_57, PPO_58 (ent_coef={ENT_COEF_PRE} fixed)")
    print("=" * 80)

    env = DummyVecEnv([make_training_env for _ in range(32)])
    env = VecFrameStack(env, n_stack=4)

    eval_env = DummyVecEnv([make_eval_env])
    eval_env = VecFrameStack(eval_env, n_stack=4)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/{RUN_NAME}",
        log_path=f"./logs/{RUN_NAME}",
        eval_freq=50_000,
        n_eval_episodes=50,
        deterministic=True,
        render=False,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=CHECKPOINT_PATH,
        name_prefix="latest_checkpoint",
        save_replay_buffer=False,
        verbose=1,
    )

    memorization_callback = MemorizationCheckCallback(
        run_name=RUN_NAME,
        sticky_actions=False,
        check_freq=1_000_000,
        n_games=20,
        make_env_fn=make_check_env,
        check_deterministic_false=True,
        summary_lines=[
            f"PPO_55b — {label} ENTROPY ({ENT_COEF_PRE}→{ENT_COEF_POST}, {mult:.1f}×) from {SOURCE_RUN} 10M",
            f"Y-Perturb: cooldown={PERTURB_COOLDOWN}f, prob={PERTURB_PROB}, ±{PERTURB_RANGE}px",
            f"Source: PPO_55 @ 10M — det=False peak: 9 unique, avg 11.6, best 16",
            f"Controls: PPO_55/57/58 (ent_coef={ENT_COEF_PRE} fixed)",
            f"Hypothesis: ent_coef={ENT_COEF_POST} prevents argmax collapse",
        ],
    )

    callbacks = CallbackList([eval_callback, checkpoint_callback, memorization_callback])

    if fresh_start:
        print(f"Loading source checkpoint: {checkpoint_file}")
        model = PPO.load(checkpoint_file, env=env, device="cuda")
        old_ent = model.ent_coef
        model.ent_coef = ENT_COEF_POST
        print(f"  ent_coef: {old_ent} → {model.ent_coef}  (intervention applied)")
    else:
        print(f"Resuming from own checkpoint: {checkpoint_file}")
        model = PPO.load(checkpoint_file, env=env, device="cuda")
        print(f"  ent_coef: {model.ent_coef}  (preserved from checkpoint)")

    print(f"  Step: {model.num_timesteps:,} → target {TARGET_STEPS:,}")

    remaining = TARGET_STEPS - model.num_timesteps
    if remaining <= 0:
        print("Target already reached. Nothing to do.")
    else:
        model.learn(
            total_timesteps=remaining,
            callback=callbacks,
            reset_num_timesteps=False,
            tb_log_name=RUN_NAME,
        )

    model.save(f"./models/{RUN_NAME}/final_model")
    print(f"\n{RUN_NAME} complete at {model.num_timesteps:,} total steps.")
    env.close()
    eval_env.close()
