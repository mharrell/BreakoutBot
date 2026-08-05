"""
Split-watcher using ball teleport instead of brick clearing.

BrickClearWrapper is unreliable — the game engine regenerates brick display
data from internal state every frame. Ball teleport avoids this entirely:
after the ball is launched, teleport it to a different X position on the
ALT side. Both sides see the SAME bricks, but the ball is at a different X.
A reactive policy tracks the diverged ball; a script ignores it.

This measures the same thing as the brick split-watcher: does the argmax
track the ball position? But it's more reliable because ball RAM writes
actually stick (unlike brick RAM which regenerates).

Usage:
    python ball_teleport_split_watcher.py --model ./models/PPO_131/final_model.zip --games 20

Output columns (per game, per frame):
    game, layout, frame, full_act, alt_act,
    full_px, full_bx, full_by,
    alt_px, alt_bx, alt_by,
    full_pb_dist, alt_pb_dist,
    alt_tracks_diverged_ball
"""
import sys
import re
import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import FireResetEnv, EpisodicLifeEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import cv2
import ale_py
gym.register_envs(ale_py)

BALL_X, BALL_Y, PADDLE_X = 99, 101, 72
NOOP, FIRE, RIGHT, LEFT = 0, 1, 2, 3


def make_env(teleport_x=None):
    """Build a raw Breakout env WITHOUT NoopResetEnv."""
    env = gym.make("ALE/Breakout-v5", frameskip=4, repeat_action_probability=0)
    env = FireResetEnv(env)
    env = EpisodicLifeEnv(env)
    return env


def get_ram(env):
    return env.unwrapped.ale.getRAM()


def initial_frame_stack(obs):
    gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return [gray] * 4


def update_frame_stack(fs, obs):
    gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    fs.pop(0)
    fs.append(gray)
    return fs


def apply_teleport(env, offset_x):
    """Teleport the ball X by offset_x pixels. Reads current ball_x, adds offset, writes back."""
    ball_x = int(get_ram(env)[BALL_X])
    new_x = max(10, min(150, ball_x + offset_x))
    env.unwrapped.ale.setRAM(BALL_X, new_x)
    return new_x


if __name__ == "__main__":
    MODEL_PATH = "./models/PPO_131/final_model.zip"
    N_GAMES = 20
    TELEPORT_OFFSET = 30  # pixels to shift ball on ALT side
    MAX_FRAMES = 10000
    OUTPUT_DIR = "recordings/split_watcher_batch"

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--model": MODEL_PATH = args[i + 1]; i += 2
        elif args[i] == "--games": N_GAMES = int(args[i + 1]); i += 2
        elif args[i] == "--offset": TELEPORT_OFFSET = int(args[i + 1]); i += 2
        elif args[i] == "--output-dir": OUTPUT_DIR = args[i + 1]; i += 2
        else: i += 1

    m = re.search(r"PPO_\d+[a-z]?", MODEL_PATH)
    run_name = m.group(0) if m else "model"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load model — no env needed for predict-only usage
    model = PPO.load(MODEL_PATH, device="cuda")

    print(f"Ball-Teleport Split-Watcher — {run_name}")
    print(f"  Model: {MODEL_PATH} @ {model.num_timesteps:,} steps")
    print(f"  Games: {N_GAMES}")
    print(f"  Teleport offset: +/-{TELEPORT_OFFSET}px")
    print()

    # Summary accumulators
    px_correlations = []
    action_divergences = []
    full_scores = []
    alt_scores = []
    tracking_frames_total = 0
    total_frames_all = 0

    for game_idx in range(N_GAMES):
        env_full = make_env()
        env_alt = make_env()

        obs_full, _ = env_full.reset()
        obs_alt, _ = env_alt.reset()

        fs_full = initial_frame_stack(obs_full)
        fs_alt = initial_frame_stack(obs_alt)

        # Both sides: FIRE to launch ball
        obs_full, _, _, _, _ = env_full.step(FIRE)
        obs_alt, _, _, _, _ = env_alt.step(FIRE)
        fs_full = update_frame_stack(fs_full, obs_full)
        fs_alt = update_frame_stack(fs_alt, obs_alt)

        # Wait for ball to be in play, then teleport on ALT side
        teleported = False
        for _ in range(20):
            full_ram = get_ram(env_full)
            alt_ram = get_ram(env_alt)
            if int(full_ram[BALL_Y]) < 180 and not teleported:
                # Ball is in play — teleport ALT ball
                new_bx = apply_teleport(env_alt, TELEPORT_OFFSET)
                teleported = True

            if not teleported:
                obs_full, _, _, _, _ = env_full.step(NOOP)
                obs_alt, _, _, _, _ = env_alt.step(NOOP)
                fs_full = update_frame_stack(fs_full, obs_full)
                fs_alt = update_frame_stack(fs_alt, obs_alt)
            else:
                break

        # Track per-frame data
        full_px_list, alt_px_list = [], []
        full_act_list, alt_act_list = [], []
        full_score, alt_score = 0, 0
        done_full, done_alt = False, False
        frame = 0
        game_tracking = 0
        game_frames = 0

        while not (done_full and done_alt) and frame < MAX_FRAMES:
            frame += 1

            # Predict independently per side
            if not done_full:
                left_obs = np.expand_dims(fs_full, axis=0)
                left_action, _ = model.predict(left_obs, deterministic=True)
                left_act = int(left_action[0])
            else:
                left_act = NOOP

            if not done_alt:
                right_obs = np.expand_dims(fs_alt, axis=0)
                right_action, _ = model.predict(right_obs, deterministic=True)
                right_act = int(right_action[0])
            else:
                right_act = NOOP

            # RAM state
            full_ram = get_ram(env_full) if not done_full else None
            alt_ram = get_ram(env_alt) if not done_alt else None

            full_px = int(full_ram[PADDLE_X]) if full_ram is not None else -1
            full_bx = int(full_ram[BALL_X]) if full_ram is not None else -1
            alt_px = int(alt_ram[PADDLE_X]) if alt_ram is not None else -1
            alt_bx = int(alt_ram[BALL_X]) if alt_ram is not None else -1
            alt_by = int(alt_ram[BALL_Y]) if alt_ram is not None else -1

            if full_px >= 0:
                full_px_list.append(full_px)
                full_act_list.append(left_act)
            if alt_px >= 0:
                alt_px_list.append(alt_px)
                alt_act_list.append(right_act)

            # Tracking signal: is ALT paddle closer to ALT ball than FULL ball?
            if full_ram is not None and alt_ram is not None and alt_by < 180:
                alt_to_alt_ball = abs(alt_px - alt_bx)
                alt_to_full_ball = abs(alt_px - full_bx)
                if alt_to_alt_ball < alt_to_full_ball:
                    game_tracking += 1
                game_frames += 1

            # Step FULL
            if not done_full:
                needs_serve = int(full_ram[BALL_Y]) > 180 if full_ram is not None else False
                act = FIRE if needs_serve else left_act
                obs, reward, terminated, truncated, info = env_full.step(act)
                full_score += reward
                if terminated or truncated:
                    try:
                        is_game_over = env_full.unwrapped.ale.lives() == 0
                    except Exception:
                        is_game_over = True
                    if is_game_over:
                        done_full = True
                    else:
                        obs, info = env_full.reset()
                        fs_full = initial_frame_stack(obs)
                        # Re-FIRE
                        obs, _, _, _, _ = env_full.step(FIRE)
                        fs_full = update_frame_stack(fs_full, obs)
                        continue
                else:
                    update_frame_stack(fs_full, obs)

            # Step ALT
            if not done_alt:
                needs_serve = int(alt_ram[BALL_Y]) > 180 if alt_ram is not None else False
                act = FIRE if needs_serve else right_act
                obs, reward, terminated, truncated, info = env_alt.step(act)
                alt_score += reward
                if terminated or truncated:
                    try:
                        is_game_over = env_alt.unwrapped.ale.lives() == 0
                    except Exception:
                        is_game_over = True
                    if is_game_over:
                        done_alt = True
                    else:
                        obs, info = env_alt.reset()
                        fs_alt = initial_frame_stack(obs)
                        obs, _, _, _, _ = env_alt.step(FIRE)
                        fs_alt = update_frame_stack(fs_alt, obs)
                        # Re-teleport on life loss
                        for _ in range(10):
                            ram = get_ram(env_alt)
                            if int(ram[BALL_Y]) < 180:
                                apply_teleport(env_alt, TELEPORT_OFFSET)
                                break
                            obs, _, _, _, _ = env_alt.step(NOOP)
                            fs_alt = update_frame_stack(fs_alt, obs)
                        continue
                else:
                    update_frame_stack(fs_alt, obs)

        env_full.close()
        env_alt.close()

        # Compute metrics for this game
        min_len = min(len(full_px_list), len(alt_px_list))
        if min_len > 1:
            full_arr = np.array(full_px_list[:min_len])
            alt_arr = np.array(alt_px_list[:min_len])
            corr = np.corrcoef(full_arr, alt_arr)[0, 1]
            px_correlations.append(corr)

            act_div = sum(1 for a, b in zip(full_act_list[:min_len], alt_act_list[:min_len]) if a != b)
            action_divergences.append(act_div / min_len * 100)
        else:
            px_correlations.append(1.0)
            action_divergences.append(0.0)

        full_scores.append(full_score)
        alt_scores.append(alt_score)
        tracking_frames_total += game_tracking
        total_frames_all += game_frames

        pct_tracking = game_tracking / game_frames * 100 if game_frames > 0 else 0
        print(f"  Game {game_idx + 1:2d}: FULL={int(full_score)} ALT={int(alt_score)} | "
              f"px_corr={px_correlations[-1]:.4f} | div={action_divergences[-1]:.1f}% | "
              f"tracks={pct_tracking:.0f}%")

    # Overall verdict
    print()
    print("=" * 70)
    print("OVERALL VERDICT")
    print("=" * 70)
    n = len(px_correlations)
    perfect_transfers = sum(1 for c in px_correlations if c > 0.99)
    avg_div = np.mean(action_divergences) if action_divergences else 0
    avg_px_corr = np.mean(px_correlations) if px_correlations else 1.0
    avg_tracking = tracking_frames_total / total_frames_all * 100 if total_frames_all > 0 else 0

    print(f"  Games run: {n}")
    print(f"  Perfect paddle correlation (px_corr > 0.99): {perfect_transfers}/{n}")
    print(f"  Avg px_corr: {avg_px_corr:.4f}")
    print(f"  Avg action divergence: {avg_div:.1f}%")
    print(f"  Avg tracking (ALT paddle closer to ALT ball): {avg_tracking:.1f}%")
    print(f"  Avg FULL score: {np.mean(full_scores):.1f}")
    print(f"  Avg ALT score:  {np.mean(alt_scores):.1f}")
    print()

    if perfect_transfers == n:
        print("  VERDICT: MEMORIZED")
        print("  Paddle positions are identical on every game despite ball teleport.")
        print("  This is a memorized script — the policy ignores the ball position.")
    elif avg_tracking > 60:
        print("  VERDICT: REACTIVE")
        print(f"  ALT paddle tracks the teleported ball {avg_tracking:.0f}% of the time.")
        print("  The argmax genuinely responds to ball position.")
    elif perfect_transfers == 0 and avg_div > 30:
        print("  VERDICT: LIKELY REACTIVE (moderate divergence)")
        print(f"  No perfect transfers, {avg_div:.0f}% divergence, {avg_tracking:.0f}% tracking.")
        print("  Run more games or verify with full analysis.")
    else:
        print("  VERDICT: INCONCLUSIVE")
        print(f"  {perfect_transfers}/{n} perfect transfers, {avg_div:.0f}% divergence.")
        print("  Mixed signal — run more games or use intervention probe.")
