"""
PPO_54 — ALE Experiment 4: X-Mirror + Y-Perturb (cooldown=30, prob=10% each)

PPO_51 proved that X-mirror (cooldown=30, prob=10%) allows the model to learn
clean-ALE scores of 5-18 points — up from the 0-1 point floor of higher-rate
mirrors. But the model is still SINGLE_SCRIPT: same score every game, meaning
it's a fixed action sequence, not reactive tracking.

X-mirror breaks *where* the ball arrives horizontally. This run adds Y-perturb:
±8 pixel vertical shifts that change *when* the ball reaches the paddle by 3-5
frames. A script that works at "frame 17" fails at frame 15 or 21 unless the
model actually tracks the ball and adjusts timing.

The two perturbations operate independently — each has its own 30-frame
cooldown. Expected: ~1-2 X-mirrors and ~1-2 Y-shifts per life, each followed
by a clean approach window.

Design:
  - Training:  ALE/Breakout-v5 + ALEBreakoutXMirror(cooldown=30, prob=0.10)
                                + ALEBreakoutYPerturb(cooldown=30, prob=0.10, range=8)
               Ball X reflects across center AND ball Y shifts by ±8 pixels.
               Independent 30-frame cooldowns guarantee clean trajectory windows.
  - Eval:      ALE/Breakout-v5 (clean, no perturbation)
  - Check:     ALE/Breakout-v5 (clean)
  - Arch:      NatureCNN (standard, no dropout)
  - Target:    50M steps

Hyperparams: n_envs=32, n_steps=128, batch_size=1024, n_epochs=4,
             gamma=0.99, lr=2.5e-4->1e-5, clip=0.2->0.05, ent_coef=0.006

Predecessors:
  PPO_47-48 (mid-flight position teleport 60-80%): degenerate stationary-paddle
  PPO_49-50 (X-mirror 60-80% no cooldown): too aggressive, 0-1 pt scripts
  PPO_51 (X-mirror cooldown=30, prob=10%): 5-18 pt SINGLE_SCRIPT, first to
    break 0-1 floor on clean ALE
  PPO_52 (X-mirror cooldown=30, prob=20%): frozen at 0-1 pts, killed
  PPO_53 (X-mirror cooldown=60, prob=5%): converged to 3-5 pt script by 7M
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
from ale_breakout_x_mirror import ALEBreakoutXMirror
from ale_breakout_y_perturb import ALEBreakoutYPerturb
from autoreset_wrapper import AutoResetWrapper

import ale_py
gym.register_envs(ale_py)

RUN_NAME = "PPO_54"
TARGET_STEPS = 50_000_000
CHECKPOINT_PATH = f"./models/{RUN_NAME}/checkpoint"
MIRROR_COOLDOWN = 30    # frames — guaranteed clean window between X-mirrors
MIRROR_PROB = 0.10      # per-frame probability once cooldown expires
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
    """ALE/Breakout-v5 with X-mirror + Y-perturb.

    Pipeline: ALE -> NoopResetEnv -> ALEBreakoutXMirror -> ALEBreakoutYPerturb ->
              FireResetEnv -> EpisodicLifeEnv -> GrayscaleResize ->
              ClipRewardEnv -> Monitor
    """
    env = gym.make("ALE/Breakout-v5", frameskip=1, repeat_action_probability=0)
    env = NoopResetEnv(env, noop_max=30)
    env = ALEBreakoutXMirror(env, cooldown_frames=MIRROR_COOLDOWN, mirror_prob=MIRROR_PROB)
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
    print(f"{RUN_NAME} — ALE Experiment 4: X-Mirror + Y-Perturb")
    print(f"  X-Mirror:  cooldown={MIRROR_COOLDOWN}f, prob={MIRROR_PROB*100:.0f}%")
    print(f"  Y-Perturb: cooldown={PERTURB_COOLDOWN}f, prob={PERTURB_PROB*100:.0f}%, range=±{PERTURB_RANGE}px")
    print(f"  Training: ALE/Breakout-v5 + ALEBreakoutXMirror + ALEBreakoutYPerturb")
    print(f"    X: ball reflected across playfield center with 30f cooldown")
    print(f"    Y: ball shifted ±{PERTURB_RANGE}px = 3-5 frame arrival timing change")
    print(f"    Independent cooldowns: X and Y perturbations fire on separate timers")
    print(f"    Expected: ~2-4 interventions/life total, clean windows between each")
    print(f"  Eval:    ALE/Breakout-v5 — clean, no perturbation")
    print(f"  Check:   ALE/Breakout-v5 — clean, no perturbation")
    print(f"           det=True + det=False every 1M steps")
    print(f"  Arch:    NatureCNN (standard, no dropout)")
    print(f"  Target:  {TARGET_STEPS:,} steps (~50M)")
    print(f"  Envs:    32 parallel")
    print(f"  LR:      2.5e-4 -> 1e-5 (linear), clip: 0.2 -> 0.05 (linear)")
    print(f"  ent_coef=0.006, batch_size=1024, n_steps=128, n_epochs=4, gamma=0.99")
    print()
    print(f"  Hypothesis: X-mirror breaks WHERE the ball arrives. Y-perturb breaks")
    print(f"  WHEN the ball arrives. A memorized sequence encodes both position and")
    print(f"  timing — we now attack both axes. If the model can't predict when the")
    print(f"  ball will reach the paddle, a fixed paddle-position sequence fails.")
    print(f"  The model must visually track the ball to time its intercept.")
    print(f"  Predecessors:")
    print(f"    PPO_51 (X-mirror only, cooldown=30, prob=10%): 5-18 pt SINGLE_SCRIPT")
    print(f"      — highest clean-ALE scores in the project, but still memorized")
    print(f"    PPO_52 (X-mirror only, cooldown=30, prob=20%): frozen 0-1 pts")
    print(f"    PPO_53 (X-mirror only, cooldown=60, prob=5%): 3-5 pt script by 7M")

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
            f"PPO_54 — ALE Experiment 4 (X-mirror + Y-perturb, cooldown=30f each)",
            f"X-Mirror: cooldown={MIRROR_COOLDOWN}f, prob={MIRROR_PROB} — reflect X across center",
            f"Y-Perturb: cooldown={PERTURB_COOLDOWN}f, prob={PERTURB_PROB}, ±{PERTURB_RANGE}px — shift Y mid-flight",
            f"Training: ALE/Breakout-v5 + ALEBreakoutXMirror + ALEBreakoutYPerturb",
            f"Policy: NatureCNN (standard, no dropout)",
            f"Memorization check env: ALE/Breakout-v5 (clean — no perturbation)",
            f"LR 2.5e-4->1e-5, clip 0.2->0.05, ent_coef=0.006, batch_size=1024",
            f"X-mirror breaks WHERE ball arrives. Y-perturb breaks WHEN ball arrives.",
            f"Predecessor PPO_51 (X-only): 5-18 pt SINGLE_SCRIPT — memorized but scoring",
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
