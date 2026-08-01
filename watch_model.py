"""
Watch a trained model play Breakout with rendering and RAM overlay.

Usage:
    python watch_model.py --model ./models/PPO_115/final_model.zip --games 3
    python watch_model.py --model ./models/PPO_116/best_model.zip --games 5 --fps 15 --stoch
"""
import sys
import time
import re
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import FireResetEnv, NoopResetEnv, EpisodicLifeEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import cv2
import ale_py
gym.register_envs(ale_py)

BALL_X, BALL_Y, PADDLE_X = 99, 101, 72


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


class AutoResetWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            obs, _info = self.env.reset()
        return obs, reward, terminated, truncated, info


if __name__ == "__main__":
    MODEL_PATH = "./models/PPO_115/final_model.zip"
    N_GAMES = 3
    FPS = 30
    MODE = "det"
    SHOW_RAM = True

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == '--model': MODEL_PATH = args[i + 1]; i += 2
        elif args[i] == '--games': N_GAMES = int(args[i + 1]); i += 2
        elif args[i] == '--fps': FPS = int(args[i + 1]); i += 2
        elif args[i] == '--stoch': MODE = "stoch"; i += 1
        elif args[i] == '--det': MODE = "det"; i += 1
        elif args[i] == '--no-ram': SHOW_RAM = False; i += 1
        else: i += 1

    m = re.search(r'PPO_\d+[a-z]?', MODEL_PATH)
    run_name = m.group(0) if m else "model"
    deterministic = MODE == "det"

    env = gym.make("ALE/Breakout-v5", frameskip=4, repeat_action_probability=0,
                    render_mode="human")
    env = NoopResetEnv(env, noop_max=30)
    env = FireResetEnv(env)
    env = EpisodicLifeEnv(env)
    env = GrayscaleResize(env, width=84, height=84)
    env = AutoResetWrapper(env)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)

    def get_ram():
        return env.venv.envs[0].unwrapped.ale.getRAM()

    model = PPO.load(MODEL_PATH, env=env, device="cuda")
    print(f"{run_name} @ {model.num_timesteps:,} steps, {MODE}, {N_GAMES} games, {FPS}fps")
    print("Close the ALE window to skip to next game, Ctrl+C to quit")
    print()

    frame_delay = 1.0 / FPS

    for game in range(N_GAMES):
        obs = env.reset()
        done_flag = False
        score = 0.0
        frame = 0
        t0 = time.time()

        while not done_flag:
            action, _states = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
            score += float(reward[0])
            frame += 1

            if SHOW_RAM and frame % 10 == 0:
                ram = get_ram()
                bx, by, px = int(ram[BALL_X]), int(ram[BALL_Y]), int(ram[PADDLE_X])
                print(f"\r  G{game+1} f={frame} score={int(score):>4}  "
                      f"ball=({bx:>3},{by:>3}) paddle={px:>3} dx={bx-px:+d}   ", end="")

            if done[0]:
                done_flag = True

            time.sleep(frame_delay)

        elapsed = time.time() - t0
        print(f"\r  Game {game+1}: {int(score)} pts, {frame} frames, "
              f"{elapsed:.0f}s ({frame/elapsed:.0f} fps actual)")

    env.close()
    print(f"\nDone.")
