"""
Intervention probe for PPO_107 — test whether the paddle tracks the ball.

Teleports the ball mid-game via setRAM and measures whether the paddle
CHANGES DIRECTION toward the ball's new position.

Key measurement: for each teleport, record paddle direction in the 3 frames
BEFORE the teleport. After teleport, check if the paddle reversed course
toward the ball. A memorized script continues its pattern regardless.
A reactive policy corrects toward the ball.

Dead baseline uses a center-hold strategy that keeps the ball in play
enough to reach the intervention window.

Usage:
    python probe_107_intervention.py
    python probe_107_intervention.py --games 50 --teleport-px 30
"""
import sys
import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import FireResetEnv, NoopResetEnv, EpisodicLifeEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import cv2
import ale_py
gym.register_envs(ale_py)

BALL_X = 99
BALL_Y = 101
PADDLE_X = 72
NOOP, FIRE, RIGHT, LEFT = 0, 1, 2, 3


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


def make_vec_env():
    env = gym.make("ALE/Breakout-v5", frameskip=4, repeat_action_probability=0)
    env = NoopResetEnv(env, noop_max=30)
    env = FireResetEnv(env)
    env = EpisodicLifeEnv(env)
    env = GrayscaleResize(env, width=84, height=84)
    env = AutoResetWrapper(env)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)
    return env


def get_ram(env):
    """Get RAM from a VecEnv by reaching through to the underlying ALE."""
    return env.venv.envs[0].unwrapped.ale.getRAM()


def set_ram(env, addr, val):
    """Set RAM on the underlying ALE."""
    env.venv.envs[0].unwrapped.ale.setRAM(addr, val)


def run_intervention(model, env, teleport_px, n_teleports_target, track_window=30):
    """
    Play games with mid-flight ball teleports. For each teleport, record
    paddle direction before vs after.

    Returns list of dicts with:
      - pre_dir: paddle direction before teleport (-1, 0, +1)
      - post_dir: paddle direction in response window
      - toward_ball: did paddle move toward the teleported ball?
      - reversed: did paddle change direction toward ball (strongest signal)?
      - score
    """
    results = []
    obs = env.reset()
    total_score = 0
    game_frame = 0
    frames_since_teleport = 0
    game = 0

    while len(results) < n_teleports_target:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_score += reward[0]
        game_frame += 1
        frames_since_teleport += 1

        # Conditions for teleport
        if game_frame >= 200 and frames_since_teleport >= 80:
            ram = get_ram(env)
            bx, by, px = int(ram[BALL_X]), int(ram[BALL_Y]), int(ram[PADDLE_X])

            if 40 <= by <= 140:
                # Record paddle direction BEFORE teleport (3-frame window)
                pre_positions = [px]
                for _ in range(3):
                    pre_action, _states = model.predict(obs, deterministic=True)
                    obs, pre_rew, pre_done, info = env.step(pre_action)
                    total_score += pre_rew[0]
                    pre_ram = get_ram(env)
                    pre_positions.append(int(pre_ram[PADDLE_X]))
                    game_frame += 1

                pre_px = pre_positions[-1]
                pre_dir = np.sign(pre_positions[-1] - pre_positions[0])

                # Teleport ball
                bx = int(get_ram(env)[BALL_X])
                direction = np.random.choice([-1, 1])
                new_bx = max(8, min(152, bx + direction * teleport_px))
                set_ram(env, BALL_X, new_bx)

                # Track paddle AFTER teleport
                post_positions = [pre_px]
                for track_frame in range(track_window):
                    track_action, _states = model.predict(obs, deterministic=True)
                    obs, track_rew, track_done, info = env.step(track_action)
                    total_score += track_rew[0]
                    track_ram = get_ram(env)
                    post_positions.append(int(track_ram[PADDLE_X]))
                    game_frame += 1

                post_px = post_positions[-1]
                post_dir = np.sign(post_px - pre_px)

                # Did paddle move toward ball?
                paddle_toward = (direction == 1 and post_px > pre_px) or \
                                (direction == -1 and post_px < pre_px)

                # Did paddle REVERSE toward ball? (pre_dir != 0, post_dir flips toward ball)
                reversed_toward = False
                if pre_dir != 0:
                    # Paddle was moving. Did it reverse toward the ball?
                    if direction == 1:  # ball went right
                        reversed_toward = (pre_dir == -1 and post_dir >= 0)  # was going left, now right/stopped
                    else:  # ball went left
                        reversed_toward = (pre_dir == 1 and post_dir <= 0)  # was going right, now left/stopped

                results.append({
                    'pre_dir': pre_dir,
                    'post_dir': post_dir,
                    'teleport_direction': direction,
                    'paddle_toward': paddle_toward,
                    'reversed_toward': reversed_toward,
                    'pre_positions': pre_positions,
                    'post_positions': post_positions,
                    'score_at_teleport': total_score,
                })

                frames_since_teleport = 0
                # Don't break — continue the game loop

        if done[0]:
            obs = env.reset()
            game += 1
            game_frame = 0
            frames_since_teleport = 0

    return results, total_score, game + 1


def dead_baseline(teleport_px, n_teleports_target, track_window=30):
    """
    Center-hold strategy — keeps paddle at center, fires when ball high.
    This survives long enough to get teleports (unlike pure sweep).
    """
    env = make_vec_env()
    results = []
    obs = env.reset()
    game_frame = 0
    frames_since_teleport = 0

    while len(results) < n_teleports_target:
        # Center-hold: move toward center
        ram = get_ram(env)
        px = int(ram[PADDLE_X])
        by = int(ram[BALL_Y])

        if by > 180:
            action = FIRE
        elif px < 76:
            action = RIGHT
        elif px > 84:
            action = LEFT
        else:
            action = NOOP

        obs, reward, done, info = env.step([action])
        game_frame += 1
        frames_since_teleport += 1

        if game_frame >= 200 and frames_since_teleport >= 80:
            ram = get_ram(env)
            bx, by, px = int(ram[BALL_X]), int(ram[BALL_Y]), int(ram[PADDLE_X])

            if 40 <= by <= 140:
                # Record pre-teleport paddle direction
                pre_positions = [px]
                for _ in range(3):
                    # Continue center-hold for pre-frames
                    pre_ram = get_ram(env)
                    pre_px_ct = int(pre_ram[PADDLE_X])
                    pre_by_ct = int(pre_ram[BALL_Y])
                    if pre_by_ct > 180:
                        pre_act = FIRE
                    elif pre_px_ct < 76:
                        pre_act = RIGHT
                    elif pre_px_ct > 84:
                        pre_act = LEFT
                    else:
                        pre_act = NOOP
                    obs, pre_rew, pre_done, info = env.step([pre_act])
                    pre_ram = get_ram(env)
                    pre_positions.append(int(pre_ram[PADDLE_X]))
                    game_frame += 1

                pre_px = pre_positions[-1]
                pre_dir = np.sign(pre_positions[-1] - pre_positions[0])

                # Teleport
                bx = int(get_ram(env)[BALL_X])
                direction = np.random.choice([-1, 1])
                new_bx = max(8, min(152, bx + direction * teleport_px))
                set_ram(env, BALL_X, new_bx)

                # Post-teleport tracking
                post_positions = [pre_px]
                for track_frame in range(track_window):
                    post_ram = get_ram(env)
                    post_px_ct = int(post_ram[PADDLE_X])
                    post_by_ct = int(post_ram[BALL_Y])
                    if post_by_ct > 180:
                        post_act = FIRE
                    elif post_px_ct < 76:
                        post_act = RIGHT
                    elif post_px_ct > 84:
                        post_act = LEFT
                    else:
                        post_act = NOOP
                    obs, post_rew, post_done, info = env.step([post_act])
                    post_ram = get_ram(env)
                    post_positions.append(int(post_ram[PADDLE_X]))
                    game_frame += 1

                post_px = post_positions[-1]
                post_dir = np.sign(post_px - pre_px)

                paddle_toward = (direction == 1 and post_px > pre_px) or \
                                (direction == -1 and post_px < pre_px)

                reversed_toward = False
                if pre_dir != 0:
                    if direction == 1:
                        reversed_toward = (pre_dir == -1 and post_dir >= 0)
                    else:
                        reversed_toward = (pre_dir == 1 and post_dir <= 0)

                results.append({
                    'pre_dir': pre_dir,
                    'post_dir': post_dir,
                    'teleport_direction': direction,
                    'paddle_toward': paddle_toward,
                    'reversed_toward': reversed_toward,
                })

                frames_since_teleport = 0

        if done[0]:
            obs = env.reset()
            game_frame = 0
            frames_since_teleport = 0

    env.close()
    return results


if __name__ == "__main__":
    MODEL_PATH = "./models/PPO_107/best_model.zip"
    RUN_NAME = "PPO_107"
    N_TELEPORTS = 40
    TELEPORT_PX = 30

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == '--teleports': N_TELEPORTS = int(args[i + 1]); i += 2
        elif args[i] == '--teleport-px': TELEPORT_PX = int(args[i + 1]); i += 2
        elif args[i] == '--model': MODEL_PATH = args[i + 1]; i += 2
        elif args[i] == '--run-name': RUN_NAME = args[i + 1]; i += 2
        else: i += 1

    # Derive run name from model path if not explicitly set
    if RUN_NAME == "PPO_107" and MODEL_PATH != "./models/PPO_107/best_model.zip":
        import re
        m = re.search(r'PPO_\d+[a-z]?', MODEL_PATH)
        if m:
            RUN_NAME = m.group(0)

    print("=" * 70)
    print(f"{RUN_NAME} Intervention Probe — Direction Change Test")
    print("=" * 70)
    print(f"Model: {MODEL_PATH}")
    print(f"Target teleports: {N_TELEPORTS}, Displacement: +/-{TELEPORT_PX}px")
    print(f"Measuring: does paddle REVERSE direction toward teleported ball?")
    print()

    # --- Dead baseline (center-hold) ---
    print("--- Dead baseline (center-hold script) ---")
    t0 = time.time()
    dead = dead_baseline(TELEPORT_PX, N_TELEPORTS)
    dead_toward = sum(1 for r in dead if r['paddle_toward'])
    dead_reversed = sum(1 for r in dead if r['reversed_toward'])
    dead_moving = sum(1 for r in dead if r['pre_dir'] != 0)
    print(f"  Teleports: {len(dead)}")
    print(f"  Pre-teleport moving: {dead_moving}/{len(dead)}")
    print(f"  Toward ball (any): {dead_toward}/{len(dead)} ({dead_toward/len(dead)*100:.1f}%)")
    print(f"  REVERSED toward ball: {dead_reversed}/{dead_moving} ({dead_reversed/dead_moving*100:.1f}% of moving)")
    print(f"  ({time.time() - t0:.0f}s)")
    print()

    # --- PPO_107 ---
    print("--- PPO_107 ---")
    env = make_vec_env()
    print("Loading model...")
    model = PPO.load(MODEL_PATH, env=env, device="cuda")
    print(f"Loaded. Model step count: {model.num_timesteps:,}")
    print()

    t0 = time.time()
    results, total_score, n_games = run_intervention(model, env, TELEPORT_PX, N_TELEPORTS)
    env.close()

    model_toward = sum(1 for r in results if r['paddle_toward'])
    model_reversed = sum(1 for r in results if r['reversed_toward'])
    model_moving = sum(1 for r in results if r['pre_dir'] != 0)

    print(f"  Teleports: {len(results)}")
    print(f"  Games: {n_games}")
    print(f"  Avg score/game: {total_score / n_games:.1f}")
    print(f"  Pre-teleport moving: {model_moving}/{len(results)}")
    print(f"  Toward ball (any): {model_toward}/{len(results)} ({model_toward/len(results)*100:.1f}%)")
    print(f"  REVERSED toward ball: {model_reversed}/{model_moving} ({model_reversed/model_moving*100:.1f}% of moving)")
    print(f"  ({time.time() - t0:.0f}s)")
    print()

    # --- Comparison ---
    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)

    def pct_str(n, d):
        return f"{n}/{d} ({n/d*100:.1f}%)" if d > 0 else "N/A"

    print(f"  {'Metric':<40} {'Dead (center)':>18} {RUN_NAME:>18}")
    print(f"  {'-'*40} {'-'*18} {'-'*18}")
    print(f"  {'Toward ball (any)':<40} {pct_str(dead_toward, len(dead)):>18} {pct_str(model_toward, len(results)):>18}")
    print(f"  {'REVERSED toward ball':<40} {pct_str(dead_reversed, dead_moving):>18} {pct_str(model_reversed, model_moving):>18}")
    print()

    # Interpretation
    dead_pct = dead_reversed / dead_moving * 100 if dead_moving > 0 else 0
    model_pct = model_reversed / model_moving * 100 if model_moving > 0 else 0

    if model_pct > dead_pct + 15:
        print(f"RESULT: {RUN_NAME} reverses toward ball {model_pct:.0f}% vs dead {dead_pct:.0f}%.")
        print("Strong evidence of ball-tracking / reactivity.")
    elif model_pct > dead_pct + 5:
        print(f"RESULT: {RUN_NAME} reverses toward ball {model_pct:.0f}% vs dead {dead_pct:.0f}%.")
        print("Weak/marginal evidence. Run more teleports to increase confidence.")
    else:
        print(f"RESULT: {RUN_NAME} ({model_pct:.0f}%) indistinguishable from dead ({dead_pct:.0f}%).")
        print("No evidence of ball-tracking. Policy is likely a memorized script.")
