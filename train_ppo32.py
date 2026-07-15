"""
PPO_32 — Experiment 4: Low-Sticky Single-Phase Training
Trains from scratch with repeat_action_probability=0.05 for 400M steps.
Single phase — no pretraining/fine-tuning split, no confounded phase transitions.

Hypothesis: p=0.05 provides enough stochasticity to prevent deterministic script
formation (unlike p=0.0, which produces memorized policies in every case) but not
so much noise that the policy fails to build reliable reactive foundations (unlike
p=0.25 from scratch, which produced PPO_27's 21% zero-score rate).

Comparison groups (all at 400M total steps):
  PPO_32: p=0.05 x 400M (single phase, tested here)
  PPO_30b: p=0.0 x 100M -> p=0.25 x 300M (confirmed memorized + noise)
  PPO_31b: p=0.0 x 300M -> p=0.25 x 100M (confirmed memorized + noise)

Key design decision: MemorizationCheckCallback uses sticky_actions=False so
verdicts are valid (the callback doesn't support p=0.05, and sticky-off is the
only reliable behavioral test per FLAWS.md F-001). The model trained at p=0.05
should generalize fine to p=0.0 — the 5% noise gap is small.

See EXPERIMENTS.md Experiment 4 for full design, prediction table, and protocol.
"""
import os
import glob
import ale_py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from memorization_check_callback import MemorizationCheckCallback
from brick_counter import BrickCountingVecWrapper, BrickRolloutCallback

gym.register_envs(ale_py)

RUN_NAME = "PPO_32"
TARGET_STEPS = 400_000_000
STICKY_PROB = 0.05  # the variable being tested — Goldilocks hypothesis
CHECKPOINT_PATH = f"./models/{RUN_NAME}/checkpoint"


def linear_schedule(start: float, end: float):
    def schedule(progress_remaining: float) -> float:
        return end + (start - end) * progress_remaining
    return schedule


def get_latest_checkpoint(path):
    checkpoints = glob.glob(os.path.join(path, "latest_checkpoint_*_steps.zip"))
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getmtime)


if __name__ == "__main__":
    # Single phase: p=0.05 throughout, both training and eval
    env = make_atari_env("ALE/Breakout-v5", n_envs=32, seed=None,
                         env_kwargs={"repeat_action_probability": STICKY_PROB})
    env = VecFrameStack(env, n_stack=4)
    env = BrickCountingVecWrapper(env)

    eval_env = make_atari_env("ALE/Breakout-v5", n_envs=1, seed=None,
                              env_kwargs={"repeat_action_probability": STICKY_PROB})
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

    # Memorization checks use sticky_actions=False so verdicts are valid.
    memorization_callback = MemorizationCheckCallback(
        run_name=RUN_NAME,
        sticky_actions=False,
        check_freq=1_000_000,
        n_games=20,
        summary_lines=[
            "PPO_32 — Experiment 4 (low-sticky single-phase training)",
            f"p={STICKY_PROB} from scratch, 32 envs, fresh agent, target 400M steps",
            "LR 2.5e-4->1e-5, clip 0.2->0.05, ent_coef=0.006, batch_size=1024",
            "Memorization checks run sticky-OFF for valid verdicts (FLAWS.md F-001)",
            "Tests Goldilocks hypothesis: p=0.05 prevents memorization without causing fragility",
        ],
    )

    callbacks = CallbackList([eval_callback, checkpoint_callback, memorization_callback,
                              BrickRolloutCallback()])

    resume_path = get_latest_checkpoint(CHECKPOINT_PATH)

    if resume_path:
        print(f"Resuming {RUN_NAME} from {resume_path}...")
        model = PPO.load(resume_path, env=env, device="cuda")
        reset_num_timesteps = False
    else:
        print(f"Starting {RUN_NAME} from scratch with repeat_action_probability={STICKY_PROB}...")
        print("Experiment 4: Low-sticky single-phase training (Goldilocks hypothesis)")
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

    # Absolute step target — safe across restarts
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
    print("Next steps:")
    print(f"  1. Run funnel_recorder_ppo_32.py (10k games, sticky on) — gold standard")
    print(f"  2. Run funnel_recorder_ppo_32_nosticky.py (10k games, sticky off) — memorization verification")
    print(f"  3. Run sticky probability sweep at p=0.05, 0.10, 0.15, 0.20, 0.25")
    env.close()
    eval_env.close()
