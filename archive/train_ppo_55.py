"""
PPO_55 — ALE Experiment 4b: Y-Perturb Only (cooldown=30, prob=10%, ±8px)

Clean ablation: PPO_51 tested X-mirror alone and got 5-18 pt SINGLE_SCRIPT.
This run tests Y-perturb alone at the same settings, isolating the timing axis.
If Y-only produces comparable scores to X-only, timing and position are equally
valuable axes. If Y-only does significantly better or worse, we learn which
axis the model relies on more.

Y-perturb shifts ball Y by ±8 pixels mid-flight — changes arrival timing by
3-5 frames without altering horizontal trajectory. A timed script that works
at "frame 17" fails at frame 15 or 21.

Design:
  - Training:  ALE/Breakout-v5 + ALEBreakoutYPerturb(cooldown=30, prob=0.10, range=8)
               Ball Y shifts by ±8px with 30-frame cooldown between perturbations.
  - Eval:      ALE/Breakout-v5 (clean, no perturbation)
  - Check:     ALE/Breakout-v5 (clean)
  - Arch:      NatureCNN (standard, no dropout)
  - Target:    50M steps

Hyperparams: n_envs=32, n_steps=128, batch_size=1024, n_epochs=4,
             gamma=0.99, lr=2.5e-4->1e-5, clip=0.2->0.05, ent_coef=0.006

Predecessors:
  PPO_51 (X-mirror only, cooldown=30, prob=10%): 5-18 pt SINGLE_SCRIPT
    — breaks WHERE ball arrives, produces memorized but scoring scripts
  PPO_54 (X+Y, cooldown=30, prob=10% each): running — breaks both axes
  PPO_55 (this run): isolates the Y/timing axis to measure its standalone effect
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

RUN_NAME = "PPO_55"
TARGET_STEPS = 50_000_000
CHECKPOINT_PATH = f"./models/{RUN_NAME}/checkpoint"
PERTURB_COOLDOWN = 30   # frames — guaranteed clean window between Y-shifts
PERTURB_PROB = 0.10     # per-frame probability once cooldown expires
PERTURB_RANGE = 8       # ±pixels shift in ball Y


class GrayscaleResize(gym.ObservationWrapper):
    """Resize grayscale to (height, width, 1) — compatible with VecFrameStack."""
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
# Environment builders
# ---------------------------------------------------------------------------

def make_training_env():
    """ALE/Breakout-v5 with Y-perturb only.

    Pipeline: ALE -> NoopResetEnv -> ALEBreakoutYPerturb ->
              FireResetEnv -> EpisodicLifeEnv -> GrayscaleResize ->
              ClipRewardEnv -> Monitor
    """
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
    """Clean ALE/Breakout-v5 — no perturbation, the generalization test.

    Pipeline: ALE -> NoopResetEnv -> FireResetEnv -> EpisodicLifeEnv ->
              GrayscaleResize -> ClipRewardEnv -> Monitor -> AutoResetWrapper
    """
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
    """Clean ALE/Breakout-v5 for memorization checks.

    NO EpisodicLifeEnv — the MemorizationCheckCallback has its own life-loss
    handling (step([1]) to respawn) which is incompatible with EpisodicLifeEnv
    setting done=True on every life.

    Pipeline: ALE -> NoopResetEnv -> FireResetEnv ->
              GrayscaleResize -> ClipRewardEnv -> Monitor ->
              DummyVecEnv[1] -> VecFrameStack(4)
    """
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"{RUN_NAME} — ALE Experiment 4b: Y-Perturb Only (clean ablation)")
    print(f"  Y-Perturb: cooldown={PERTURB_COOLDOWN}f, prob={PERTURB_PROB*100:.0f}%, range=±{PERTURB_RANGE}px")
    print(f"  Training: ALE/Breakout-v5 + ALEBreakoutYPerturb")
    print(f"    Ball Y shifted ±{PERTURB_RANGE}px = 3-5 frame arrival timing change")
    print(f"    {PERTURB_COOLDOWN}f cooldown = clean trajectory windows between shifts")
    print(f"    No X-mirror — isolates the timing axis")
    print(f"  Eval:    ALE/Breakout-v5 — clean, no perturbation")
    print(f"  Check:   ALE/Breakout-v5 — clean, no perturbation")
    print(f"           det=True + det=False every 1M steps")
    print(f"  Arch:    NatureCNN (standard, no dropout)")
    print(f"  Target:  {TARGET_STEPS:,} steps (~50M)")
    print(f"  Envs:    32 parallel")
    print(f"  LR:      2.5e-4 -> 1e-5 (linear), clip: 0.2 -> 0.05 (linear)")
    print(f"  ent_coef=0.006, batch_size=1024, n_steps=128, n_epochs=4, gamma=0.99")
    print()
    print(f"  Hypothesis: Y-perturb breaks WHEN the ball arrives. A memorized")
    print(f"  sequence encodes timing. If the ball arrives 3-5 frames early or")
    print(f"  late, a fixed paddle-position-at-frame-N sequence fails. The model")
    print(f"  must visually track the ball to intercept it.")
    print(f"  Clean ablation: compare against PPO_51 (X-only, same settings)")
    print(f"  to determine which axis contributes more to forcing reactivity.")
    print(f"  Predecessors:")
    print(f"    PPO_51 (X-mirror only, cooldown=30, prob=10%): 5-18 pt SINGLE_SCRIPT")
    print(f"    PPO_54 (X+Y, cooldown=30, prob=10% each): running")

    # -------------------------------------------------------------------
    # Vectorized environments
    # -------------------------------------------------------------------
    env = DummyVecEnv([make_training_env for _ in range(32)])
    env = VecFrameStack(env, n_stack=4)

    eval_env = DummyVecEnv([make_eval_env])
    eval_env = VecFrameStack(eval_env, n_stack=4)

    # -------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------
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
            f"PPO_55 — ALE Experiment 4b (Y-perturb only, cooldown={PERTURB_COOLDOWN}f)",
            f"Y-Perturb: cooldown={PERTURB_COOLDOWN}f, prob={PERTURB_PROB}, ±{PERTURB_RANGE}px",
            f"Training: ALE/Breakout-v5 + ALEBreakoutYPerturb (Y-only, no X-mirror)",
            f"Policy: NatureCNN (standard, no dropout)",
            f"Memorization check env: ALE/Breakout-v5 (clean — no perturbation)",
            f"LR 2.5e-4->1e-5, clip 0.2->0.05, ent_coef=0.006, batch_size=1024",
            f"Clean ablation: isolates Y/timing axis vs PPO_51 (X/position axis)",
            f"Predecessor PPO_51 (X-only @ same settings): 5-18 pt SINGLE_SCRIPT",
        ],
    )

    callbacks = CallbackList([
        eval_callback,
        checkpoint_callback,
        memorization_callback,
    ])

    # -------------------------------------------------------------------
    # Model setup — NatureCNN (standard, no custom feature extractor)
    # -------------------------------------------------------------------
    resume_path = get_latest_checkpoint(CHECKPOINT_PATH)

    if resume_path:
        print(f"Resuming {RUN_NAME} from {resume_path}...")
        model = PPO.load(resume_path, env=env, device="cuda")
        reset_num_timesteps = False
    else:
        print(f"Starting {RUN_NAME} from scratch...")
        model = PPO(
            "CnnPolicy",
            env,
            verbose=1,
            device="cuda",
            tensorboard_log=f"./tensorboard/{RUN_NAME}",
            n_steps=128,
            batch_size=1024,
            n_epochs=4,
            gamma=0.99,
            learning_rate=linear_schedule(2.5e-4, 1e-5),
            clip_range=linear_schedule(0.2, 0.05),
            ent_coef=0.006,
            vf_coef=0.5,
        )
        reset_num_timesteps = True

    # -------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------
    remaining = TARGET_STEPS - model.num_timesteps
    print(f"{RUN_NAME}: current step {model.num_timesteps:,}, "
          f"training {remaining:,} more steps to reach {TARGET_STEPS:,}")

    if remaining <= 0:
        print("Target already reached. Nothing to do.")
    else:
        model.learn(
            total_timesteps=remaining,
            callback=callbacks,
            reset_num_timesteps=reset_num_timesteps,
            tb_log_name=RUN_NAME,
        )

    model.save(f"./models/{RUN_NAME}/final_model")
    print(f"\n{RUN_NAME} complete at {model.num_timesteps:,} total steps.")
    env.close()
    eval_env.close()
