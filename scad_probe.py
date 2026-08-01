"""
SCAD Probe -- State-Conditioned Action Distribution.

The first frame-level behavioral metric in this project. Records per-frame
(ball_x, ball_y, paddle_x, action, action_probs) during normal gameplay and
computes conditional action probabilities. Directly measures whether PPO actions
depend on ball position -- replacing interpretive inference (L-012).

Key metrics:
  - Tracking probability: P(moves toward ball | ball not centered under paddle)
  - Mutual information: I(action ; sign(ball_x - paddle_x))
  - Conditional action distributions: P(action | ball left/center/right)

Dead baselines (sweep script, center-hold): MI ~ 0, tracking prob ~ 50%.
Reactive policy: MI > 0.1 bits, tracking prob > 70%.

Tests BOTH det=True and det=False (Critical Rule #9 -- the gap matters).

Usage:
    python scad_probe.py --model ./models/PPO_114/best_model.zip --games 20
    python scad_probe.py --model ./models/PPO_114/best_model.zip --stoch
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

BALL_X, BALL_Y, PADDLE_X = 99, 101, 72
NOOP, FIRE, RIGHT, LEFT = 0, 1, 2, 3
CENTER_THRESHOLD = 4  # px -- ball within +/-4px of paddle center is "centered"
DIRECTION_HISTORY = 3  # frames to look back for paddle movement direction


# -- Wrappers ------------------------------------------------------------

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
    return env.venv.envs[0].unwrapped.ale.getRAM()


# -- Action probability extraction ---------------------------------------

def get_action_probs(model, obs):
    """Get the full action probability distribution for a single observation.

    Args:
        model: SB3 PPO model
        obs: observation batch (1, 4, 84, 84)

    Returns:
        probs: numpy array of shape (4,) -- P(NOOP), P(FIRE), P(RIGHT), P(LEFT)
    """
    import torch as th
    # obs from VecFrameStack is NHWC (1, 84, 84, 4) -- transpose to NCHW (1, 4, 84, 84)
    obs_nchw = np.transpose(obs, (0, 3, 1, 2)).copy()
    obs_tensor = th.tensor(obs_nchw).float().to(model.device)
    with th.no_grad():
        dist = model.policy.get_distribution(obs_tensor)
        probs = dist.distribution.probs.cpu().numpy()[0]
    return probs


# -- Frame-level data collection -----------------------------------------

def collect_frame_data(model, env, n_games, deterministic, device="cuda"):
    """Play games and collect per-frame state + action data.

    Returns list of dicts, each with:
      ball_x, ball_y, paddle_x, action, action_probs[4],
      ball_relative (left/center/right), paddle_moving_toward_ball (bool),
      game_id, frame
    """
    frames = []
    obs = env.reset()

    for game_id in range(n_games):
        game_frame = 0
        prev_px = None
        done_flag = False

        while not done_flag:
            ram = get_ram(env)
            bx, by, px = int(ram[BALL_X]), int(ram[BALL_Y]), int(ram[PADDLE_X])

            # Get action from policy
            action, _states = model.predict(obs, deterministic=deterministic)
            probs = get_action_probs(model, obs)

            # Determine ball position relative to paddle
            dx = bx - px
            if abs(dx) <= CENTER_THRESHOLD:
                ball_relative = "center"
            elif dx < 0:
                ball_relative = "left"
            else:
                ball_relative = "right"

            # Did paddle move toward ball?
            paddle_toward = None
            if prev_px is not None:
                paddle_dx = px - prev_px
                if ball_relative == "left" and paddle_dx < 0:
                    paddle_toward = True
                elif ball_relative == "right" and paddle_dx > 0:
                    paddle_toward = True
                elif paddle_dx != 0:
                    paddle_toward = False  # moved, but away from ball
                # paddle_dx == 0 -> None (didn't move)

            frames.append({
                "ball_x": bx, "ball_y": by, "paddle_x": px,
                "action": int(action[0]),
                "probs": probs.tolist(),
                "ball_relative": ball_relative,
                "paddle_toward": paddle_toward,
                "game_id": game_id,
                "frame": game_frame,
            })

            prev_px = px
            obs, reward, done, info = env.step(action)
            game_frame += 1
            if done[0]:
                done_flag = True

        obs = env.reset()

    return frames


# -- Dead baselines ------------------------------------------------------

def collect_sweep_data(env, n_games):
    """Sweep script: paddle sweeps left-right continuously, fires on serve."""
    frames = []
    obs = env.reset()
    sweep_dir = 1  # start moving right
    sweep_counter = 0

    for game_id in range(n_games):
        game_frame = 0
        prev_px = None
        done_flag = False

        while not done_flag:
            ram = get_ram(env)
            bx, by, px = int(ram[BALL_X]), int(ram[BALL_Y]), int(ram[PADDLE_X])

            # Sweep logic: alternate direction every ~30 frames
            sweep_counter += 1
            if sweep_counter >= 30:
                sweep_dir *= -1
                sweep_counter = 0

            if by > 180:
                action = FIRE
            elif sweep_dir == 1:
                action = RIGHT
            else:
                action = LEFT

            # Uniform probs on the chosen action (dead script has no distribution)
            probs = np.zeros(4)
            probs[action] = 1.0

            dx = bx - px
            if abs(dx) <= CENTER_THRESHOLD:
                ball_relative = "center"
            elif dx < 0:
                ball_relative = "left"
            else:
                ball_relative = "right"

            paddle_toward = None
            if prev_px is not None:
                paddle_dx = px - prev_px
                if ball_relative == "left" and paddle_dx < 0:
                    paddle_toward = True
                elif ball_relative == "right" and paddle_dx > 0:
                    paddle_toward = True
                elif paddle_dx != 0:
                    paddle_toward = False

            frames.append({
                "ball_x": bx, "ball_y": by, "paddle_x": px,
                "action": action,
                "probs": probs.tolist(),
                "ball_relative": ball_relative,
                "paddle_toward": paddle_toward,
                "game_id": game_id,
                "frame": game_frame,
            })

            prev_px = px
            obs, reward, done, info = env.step([action])
            game_frame += 1
            if done[0]:
                done_flag = True

        obs = env.reset()

    return frames


def collect_centerhold_data(env, n_games):
    """Center-hold script: holds center, fires on serve."""
    frames = []
    obs = env.reset()

    for game_id in range(n_games):
        game_frame = 0
        prev_px = None
        done_flag = False

        while not done_flag:
            ram = get_ram(env)
            bx, by, px = int(ram[BALL_X]), int(ram[BALL_Y]), int(ram[PADDLE_X])

            if by > 180:
                action = FIRE
            elif px < 76:
                action = RIGHT
            elif px > 84:
                action = LEFT
            else:
                action = NOOP

            probs = np.zeros(4)
            probs[action] = 1.0

            dx = bx - px
            if abs(dx) <= CENTER_THRESHOLD:
                ball_relative = "center"
            elif dx < 0:
                ball_relative = "left"
            else:
                ball_relative = "right"

            paddle_toward = None
            if prev_px is not None:
                paddle_dx = px - prev_px
                if ball_relative == "left" and paddle_dx < 0:
                    paddle_toward = True
                elif ball_relative == "right" and paddle_dx > 0:
                    paddle_toward = True
                elif paddle_dx != 0:
                    paddle_toward = False

            frames.append({
                "ball_x": bx, "ball_y": by, "paddle_x": px,
                "action": action,
                "probs": probs.tolist(),
                "ball_relative": ball_relative,
                "paddle_toward": paddle_toward,
                "game_id": game_id,
                "frame": game_frame,
            })

            prev_px = px
            obs, reward, done, info = env.step([action])
            game_frame += 1
            if done[0]:
                done_flag = True

        obs = env.reset()

    return frames


# -- Metrics computation -------------------------------------------------

def compute_metrics(frames, label):
    """Compute SCAD metrics from collected frame data."""
    n = len(frames)

    # --- Tracking probability ---
    # Fraction of frames where paddle moved toward ball, excluding "centered"
    off_center = [f for f in frames if f["ball_relative"] != "center" and f["paddle_toward"] is not None]
    if off_center:
        toward_count = sum(1 for f in off_center if f["paddle_toward"])
        tracking_prob = toward_count / len(off_center)
    else:
        tracking_prob = None

    # --- Mutual information: I(action ; ball_relative) ---
    # H(action) = -sum_a P(a) log2 P(a)
    actions = np.array([f["action"] for f in frames])
    action_counts = np.bincount(actions, minlength=4)
    p_action = action_counts / action_counts.sum()
    p_action = np.clip(p_action, 1e-10, None)
    H_action = -np.sum(p_action * np.log2(p_action))

    # H(action | ball_relative) for each region
    regions = ["left", "center", "right"]
    region_frames = {r: [f for f in frames if f["ball_relative"] == r] for r in regions}
    H_action_given_region = 0.0
    for r in regions:
        if region_frames[r]:
            r_actions = np.array([f["action"] for f in region_frames[r]])
            r_counts = np.bincount(r_actions, minlength=4)
            r_p = r_counts / r_counts.sum()
            r_p = np.clip(r_p, 1e-10, None)
            H_action_given_region += (len(region_frames[r]) / n) * (-np.sum(r_p * np.log2(r_p)))

    mutual_info = H_action - H_action_given_region

    # --- Conditional action distributions ---
    cond_dists = {}
    for r in regions:
        if region_frames[r]:
            r_actions = np.array([f["action"] for f in region_frames[r]])
            r_counts = np.bincount(r_actions, minlength=4)
            cond_dists[r] = {
                "count": len(region_frames[r]),
                "pct": len(region_frames[r]) / n * 100,
                "NOOP": r_counts[0] / len(region_frames[r]) * 100,
                "FIRE": r_counts[1] / len(region_frames[r]) * 100,
                "RIGHT": r_counts[2] / len(region_frames[r]) * 100,
                "LEFT": r_counts[3] / len(region_frames[r]) * 100,
            }
        else:
            cond_dists[r] = {"count": 0, "pct": 0, "NOOP": 0, "FIRE": 0, "RIGHT": 0, "LEFT": 0}

    # --- Action probabilities (mean policy confidence) ---
    mean_probs = np.mean([f["probs"] for f in frames], axis=0)

    # --- Score proxy: avg frames per game ---
    games = set(f["game_id"] for f in frames)
    frames_per_game = n / len(games) if games else 0

    return {
        "label": label,
        "n_frames": n,
        "n_games": len(games),
        "tracking_prob": tracking_prob,
        "mutual_info_bits": mutual_info,
        "H_action": H_action,
        "H_action_given_region": H_action_given_region,
        "cond_dists": cond_dists,
        "mean_probs": mean_probs,
        "frames_per_game": frames_per_game,
    }


# -- Display -------------------------------------------------------------

def print_results(metrics_det, metrics_stoch):
    """Print a formatted comparison table."""
    action_names = ["NOOP", "FIRE", "RIGHT", "LEFT"]

    for metrics, mode in [(metrics_det, "det=True"), (metrics_stoch, "det=False")]:
        if metrics is None:
            continue
        print(f"\n{'='*70}")
        print(f"  SCAD Probe -- {metrics['label']} ({mode})")
        print(f"{'='*70}")
        print(f"  Frames: {metrics['n_frames']:,}  Games: {metrics['n_games']}  "
              f"Frames/game: {metrics['frames_per_game']:.0f}")

        tp = metrics["tracking_prob"]
        mi = metrics["mutual_info_bits"]
        print(f"\n  -- Key Metrics --")
        if tp is not None:
            verdict = "REACTIVE" if tp > 0.65 else ("MARGINAL" if tp > 0.55 else "DEAD")
            print(f"  Tracking probability:  {tp*100:.1f}%  ->  {verdict}")
            print(f"    (fraction of off-center frames where paddle moved toward ball)")
        else:
            print(f"  Tracking probability:  N/A (no off-center frames with movement)")
        print(f"  Mutual information:    {mi:.4f} bits")
        if mi < 0.02:
            print(f"    -> Action distribution is nearly independent of ball position (DEAD)")
        elif mi < 0.08:
            print(f"    -> Weak conditioning on ball position (MARGINAL)")
        else:
            print(f"    -> Strong conditioning on ball position (REACTIVE)")

        print(f"\n  -- Conditional Action Distribution --")
        header = f"  {'Ball pos':<12} {'Frames':>8} {'%':>6}"
        for a in action_names:
            header += f" {a:>7}"
        print(header)
        print(f"  {'-'*12} {'-'*8} {'-'*6} " + " ".join(f"{'-'*7}" for _ in action_names))

        for region in ["left", "center", "right"]:
            d = metrics["cond_dists"][region]
            row = f"  {region:<12} {d['count']:>8} {d['pct']:>5.1f}%"
            for a in action_names:
                row += f" {d[a]:>6.1f}%"
            print(row)

        print(f"\n  -- Mean Policy Confidence (avg action probability) --")
        for i, a in enumerate(action_names):
            print(f"  {a}: {metrics['mean_probs'][i]*100:.1f}%")


def print_comparison_header():
    print("=" * 70)
    print("SCAD Probe -- State-Conditioned Action Distribution")
    print("=" * 70)
    print("Measures: does the policy's action distribution depend on ball position?")
    print(f"Center threshold: +/-{CENTER_THRESHOLD}px from paddle")
    print()


# -- Main ----------------------------------------------------------------

if __name__ == "__main__":
    MODEL_PATH = "./models/PPO_107/best_model.zip"
    RUN_NAME = "PPO_107"
    N_GAMES = 20
    DETERMINISTIC = True

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == '--model': MODEL_PATH = args[i + 1]; i += 2
        elif args[i] == '--run-name': RUN_NAME = args[i + 1]; i += 2
        elif args[i] == '--games': N_GAMES = int(args[i + 1]); i += 2
        elif args[i] == '--stoch': DETERMINISTIC = False; i += 1
        elif args[i] == '--det': DETERMINISTIC = True; i += 1
        else: i += 1

    # Derive run name from model path if not explicitly set
    if RUN_NAME == "PPO_107" and MODEL_PATH != "./models/PPO_107/best_model.zip":
        import re
        m = re.search(r'PPO_\d+[a-z]?', MODEL_PATH)
        if m:
            RUN_NAME = m.group(0)

    print_comparison_header()

    env = make_vec_env()

    # --- Dead baselines ---
    print("--- Dead baseline: sweep script ---")
    t0 = time.time()
    sweep_frames = collect_sweep_data(env, N_GAMES)
    sweep_metrics = compute_metrics(sweep_frames, "Sweep Script")
    print(f"  {len(sweep_frames):,} frames, {sweep_metrics['n_games']} games "
          f"({time.time() - t0:.0f}s)")

    print("\n--- Dead baseline: center-hold script ---")
    t0 = time.time()
    ch_frames = collect_centerhold_data(env, N_GAMES)
    ch_metrics = compute_metrics(ch_frames, "Center-Hold Script")
    print(f"  {len(ch_frames):,} frames, {ch_metrics['n_games']} games "
          f"({time.time() - t0:.0f}s)")

    # --- Model ---
    mode_label = "det=True" if DETERMINISTIC else "det=False"
    print(f"\n--- {RUN_NAME} ({mode_label}) ---")
    print("Loading model...")
    model = PPO.load(MODEL_PATH, env=env, device="cuda")
    print(f"Loaded. Model step count: {model.num_timesteps:,}")

    t0 = time.time()
    model_frames = collect_frame_data(model, env, N_GAMES, deterministic=DETERMINISTIC)
    model_metrics = compute_metrics(model_frames, RUN_NAME)
    print(f"  {len(model_frames):,} frames, {model_metrics['n_games']} games "
          f"({time.time() - t0:.0f}s)")

    env.close()

    # --- Print detailed results for each ---
    if DETERMINISTIC:
        print_results(model_metrics, None)
    else:
        print_results(None, model_metrics)

    # Print dead baselines in compact form
    print(f"\n{'='*70}")
    print(f"  DEAD BASELINE COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Metric':<30} {'Sweep':>15} {'Center-Hold':>15} {RUN_NAME:>15}")
    print(f"  {'-'*30} {'-'*15} {'-'*15} {'-'*15}")
    tp_s = f"{sweep_metrics['tracking_prob']*100:.1f}%" if sweep_metrics['tracking_prob'] else "N/A"
    tp_c = f"{ch_metrics['tracking_prob']*100:.1f}%" if ch_metrics['tracking_prob'] else "N/A"
    tp_m = f"{model_metrics['tracking_prob']*100:.1f}%" if model_metrics['tracking_prob'] else "N/A"
    print(f"  {'Tracking probability':<30} {tp_s:>15} {tp_c:>15} {tp_m:>15}")
    print(f"  {'Mutual information (bits)':<30} {sweep_metrics['mutual_info_bits']:>15.4f} "
          f"{ch_metrics['mutual_info_bits']:>15.4f} {model_metrics['mutual_info_bits']:>15.4f}")

    # Interpretation
    mi = model_metrics["mutual_info_bits"]
    tp = model_metrics["tracking_prob"]
    print(f"\n  INTERPRETATION:")
    if mi > 0.08 and (tp is not None and tp > 0.65):
        print(f"  MI={mi:.4f} bits + tracking={tp*100:.0f}% -> STRONG ball-tracking reactivity.")
    elif mi > 0.03 or (tp is not None and tp > 0.55):
        print(f"  MI={mi:.4f} bits, tracking={tp*100:.0f}% -> MARGINAL reactivity.")
        print(f"  Policy may be intermediate -- re-check at later checkpoint.")
    else:
        print(f"  MI={mi:.4f} bits, tracking={tp*100:.0f}% -> DEAD/MEMORIZED.")
        if DETERMINISTIC:
            print(f"  Test with --stoch to check if distribution retains entropy (Critical Rule #9).")
