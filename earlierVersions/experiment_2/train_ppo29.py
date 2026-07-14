import os
import glob
import ale_py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
gym.register_envs(ale_py)

RUN_NAME = "PPO_29"
BASE_MODEL = "PPO_27"  # Source model to continue from
ADDITIONAL_STEPS = 500_000_000
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

    env = make_atari_env("ALE/Breakout-v5", n_envs=32, seed=None,
                         env_kwargs={"repeat_action_probability": 0.0})
    env = VecFrameStack(env, n_stack=4)

    eval_env = make_atari_env("ALE/Breakout-v5", n_envs=1, seed=None,
                              env_kwargs={"repeat_action_probability": 0.0})
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

    callbacks = CallbackList([eval_callback, checkpoint_callback])

    # Resume from PPO_29 checkpoint if it exists (previously interrupted)
    resume_path = get_latest_checkpoint(CHECKPOINT_PATH)

    if resume_path:
        print(f"Resuming {RUN_NAME} from {resume_path}...")
        model = PPO.load(resume_path, env=env, device="cuda")

    else:
        # First run: load from BASE_MODEL with sticky removed
        base_path = f"./models/{BASE_MODEL}/best_model"
        print(f"Starting {RUN_NAME} from {base_path} with sticky actions removed...")
        model = PPO.load(base_path, env=env, device="cuda",
                         custom_objects={"n_envs": 32})
        # Restart the learning rate schedule from the start
        model.learning_rate = linear_schedule(2.5e-4, 1e-5)
        model.clip_range = linear_schedule(0.2, 0.05)
        model.ent_coef = 0.006

    # Corrected continuation: always train exactly ADDITIONAL_STEPS more
    TARGET = model.num_timesteps + ADDITIONAL_STEPS
    remaining = TARGET - model.num_timesteps

    print(f"{RUN_NAME}: current step {model.num_timesteps}, training {remaining} more steps to reach {TARGET}")

    model.learn(
        total_timesteps=remaining,
        callback=callbacks,
        reset_num_timesteps=False,
        tb_log_name=RUN_NAME,
    )

    model.save(f"./models/{RUN_NAME}/final_model")
    env.close()