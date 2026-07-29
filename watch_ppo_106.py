"""
Watch PPO_106 gameplay — see what the agent is doing.

Usage:
    python watch_ppo_106.py                        # adversarial mode, current defaults
    python watch_ppo_106.py --standard             # standard Breakout (test transfer)
    python watch_ppo_106.py --games 10             # play 10 games
    python watch_ppo_106.py --stoch                # stochastic actions
    python watch_ppo_106.py --max-push 4           # tune push strength
    python watch_ppo_106.py --gain 0.3 --dead 6    # tune all params

Loads the latest checkpoint automatically.
"""
import sys
import time
import numpy as np
import cv2
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, FireResetEnv, EpisodicLifeEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from adversarial_ball_wrapper import AdversarialBallWrapper
import ale_py
gym.register_envs(ale_py)


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


def make_env(adversarial=True, dead_zone=4.0, gain=0.5, max_push=4.0, paddle_zone_y=140):
    """Build the watch environment."""
    env = gym.make("ALE/Breakout-v5", frameskip=1 if adversarial else 4,
                   repeat_action_probability=0, render_mode="human")
    env = FireResetEnv(env)
    if adversarial:
        env = AdversarialBallWrapper(env, dead_zone=dead_zone,
                                      proportional_gain=gain,
                                      paddle_zone_y=paddle_zone_y,
                                      max_push=max_push)
    env = EpisodicLifeEnv(env)
    env = GrayscaleResize(env, width=84, height=84)
    env = ClipRewardEnv(env)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)
    return env


if __name__ == "__main__":
    adversarial = True
    n_games = 5
    deterministic = True
    dead_zone = 4.0
    gain = 0.5
    max_push = 4.0
    paddle_zone_y = 140

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--standard":
            adversarial = False; i += 1
        elif args[i] == "--games":
            n_games = int(args[i + 1]); i += 2
        elif args[i] == "--stoch":
            deterministic = False; i += 1
        elif args[i] == "--det":
            deterministic = True; i += 1
        elif args[i] == "--max-push":
            max_push = float(args[i + 1]); i += 2
        elif args[i] == "--gain":
            gain = float(args[i + 1]); i += 2
        elif args[i] == "--dead":
            dead_zone = float(args[i + 1]); i += 2
        elif args[i] == "--zone-y":
            paddle_zone_y = int(args[i + 1]); i += 2
        else:
            i += 1

    import glob, os
    checkpoints = glob.glob("./models/PPO_106/checkpoint/latest_checkpoint_*_steps.zip")
    if not checkpoints:
        print("No checkpoint found!")
        sys.exit(1)
    model_path = max(checkpoints, key=os.path.getmtime)

    if adversarial:
        mode = (f"ADVERSARIAL (fs=1, dead={dead_zone}, gain={gain}, "
                f"max_push={max_push}, zone_y={paddle_zone_y})")
    else:
        mode = "STANDARD (fs=4, no push)"

    print(f"Loading {model_path}...")
    print(f"Mode: {mode}")
    print(f"Action: {'deterministic' if deterministic else 'stochastic'}")
    print(f"Games: {n_games}")
    print()

    dummy_env = make_env(adversarial=adversarial, dead_zone=dead_zone,
                         gain=gain, max_push=max_push, paddle_zone_y=paddle_zone_y)
    model = PPO.load(model_path, env=dummy_env, device="cuda")
    dummy_env.close()

    print(f"Model: {model.num_timesteps:,} steps")
    print("Press Ctrl+C to stop early.\n")

    env = make_env(adversarial=adversarial, dead_zone=dead_zone,
                   gain=gain, max_push=max_push, paddle_zone_y=paddle_zone_y)
    scores = []

    for game in range(n_games):
        obs = env.reset()
        score = 0.0
        frames = 0
        done = False
        t0 = time.time()

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done_arr, info = env.step(action)
            score += reward[0]
            frames += 1
            done = bool(done_arr[0])

        elapsed = time.time() - t0
        scores.append(score)
        print(f"  Game {game + 1}: {score:.0f} pts, {frames} frames, {elapsed:.1f}s")

    env.close()
    print(f"\nSummary: mean={np.mean(scores):.1f}, "
          f"min={np.min(scores):.0f}, max={np.max(scores):.0f}, "
          f"unique={len(set(round(s, 1) for s in scores))}")
    print(f"Mode: {mode}")
