"""
Headless split-watcher for BeamRider — quantitative ship position comparison.

Runs the same PPO model on two independent BeamRider instances.
Different noop RNG seeds produce different enemy patterns naturally.
Same model, independent argmax predictions per side.

Memorized: ship moves identically on both sides despite different enemies
Reactive:  ship positions diverge in response to different enemy threats

Usage:
    python verify_beamrider_split.py
    python verify_beamrider_split.py --model ./models/BEAMRIDER_baseline/final_model.zip
"""
import sys
import re
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import FireResetEnv, NoopResetEnv, EpisodicLifeEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import cv2
import ale_py
gym.register_envs(ale_py)

SHIP_X = 41          # player ship X position (verified: responds to LEFT/RIGHT)
GAME_STATUS = 16      # 1=neutral, 2=fighting, 3=sentinel, 4=transition
LIVES = 5             # lives remaining

# ---------------------------------------------------------------------------
# Wrappers (mirror Breakout split-watcher pattern)
# ---------------------------------------------------------------------------

class AutoResetWrapper(gym.Wrapper):
    """Auto-reset on episode end (for vec env model loading only)."""
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            obs, info = self.env.reset()
        return obs, reward, terminated, truncated, info


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_raw_env():
    """Build a raw (non-vec) BeamRider env matching the training pipeline."""
    env = gym.make("ALE/BeamRider-v5", frameskip=4, repeat_action_probability=0)
    env = NoopResetEnv(env, noop_max=30)
    env = FireResetEnv(env)
    env = EpisodicLifeEnv(env)
    return env


def get_ram(env):
    return env.unwrapped.ale.getRAM()


def initial_frame_stack(obs):
    """Build initial 4-frame grayscale stack from first observation."""
    gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return [gray] * 4


def update_frame_stack(fs, obs):
    """Push a new grayscale frame onto the stack."""
    gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    fs.pop(0)
    fs.append(gray)
    return fs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    MODEL_PATH = "./models/BEAMRIDER_baseline/final_model.zip"
    N_GAMES = 10
    MAX_FRAMES = 12000

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--model":
            MODEL_PATH = args[i + 1]; i += 2
        elif args[i] == "--games":
            N_GAMES = int(args[i + 1]); i += 2
        else:
            i += 1

    m = re.search(r"BEAMRIDER_\w+", MODEL_PATH)
    run_name = m.group(0) if m else "beamrider_model"

    # Load model with vec env (training-pipeline-compatible)
    def _make_dummy():
        e = gym.make("ALE/BeamRider-v5", frameskip=4, repeat_action_probability=0)
        e = NoopResetEnv(e, noop_max=30)
        e = FireResetEnv(e)
        e = EpisodicLifeEnv(e)
        e = GrayscaleResize(e, width=84, height=84)
        e = AutoResetWrapper(e)
        return e

    dummy_env = DummyVecEnv([_make_dummy])
    dummy_env = VecFrameStack(dummy_env, n_stack=4)
    model = PPO.load(MODEL_PATH, env=dummy_env, device="cuda")
    dummy_env.close()

    print(f"{'='*70}")
    print(f"BeamRider Split-Watcher Verification -- {run_name}")
    print(f"{'='*70}")
    print(f"Model: {MODEL_PATH} @ {model.num_timesteps:,} steps")
    print(f"Inference: deterministic (argmax)")
    print(f"Games: {N_GAMES}")
    print()
    print("Principle: Two independent BeamRider instances, different noop RNG seeds.")
    print("  Memorized: ship moves IDENTICALLY on both sides (ignores enemy state)")
    print("  Reactive:  ship positions DIVERGE (responds to different enemy patterns)")
    print()

    full_scores_all = []
    alt_scores_all = []
    px_corrs_all = []
    divergences_all = []
    perfect_transfers_all = []

    for g in range(N_GAMES):
        # Two independent envs — same config, different RNG state
        env_a = make_raw_env()
        env_b = make_raw_env()

        obs_a, _info = env_a.reset()
        obs_b, _info = env_b.reset()

        fs_a = initial_frame_stack(obs_a)
        fs_b = initial_frame_stack(obs_b)

        a_ship_x = []
        b_ship_x = []
        a_actions = []
        b_actions = []
        a_score = 0.0
        b_score = 0.0
        diverged_frames = 0
        compared_frames = 0

        done_a = False
        done_b = False
        step = 0

        while not (done_a and done_b) and step < MAX_FRAMES:
            step += 1

            # --- Independent predictions per side ---
            if not done_a:
                obs_a_stacked = np.expand_dims(fs_a, axis=0)
                action_a, _ = model.predict(obs_a_stacked, deterministic=True)
                act_a = int(action_a[0])
            else:
                act_a = 0  # NOOP

            if not done_b:
                obs_b_stacked = np.expand_dims(fs_b, axis=0)
                action_b, _ = model.predict(obs_b_stacked, deterministic=True)
                act_b = int(action_b[0])
            else:
                act_b = 0

            # Compare actions (both sides alive = valid comparison)
            if not done_a and not done_b:
                a_actions.append(act_a)
                b_actions.append(act_b)
                if act_a != act_b:
                    diverged_frames += 1
                compared_frames += 1

            # --- Step side A ---
            if not done_a:
                obs, reward, terminated, truncated, info = env_a.step(act_a)
                a_score += float(reward)
                try:
                    px = int(get_ram(env_a)[SHIP_X])
                except Exception:
                    px = -1
                a_ship_x.append(px)

                if terminated or truncated:
                    try:
                        is_game_over = env_a.unwrapped.ale.lives() == 0
                    except Exception:
                        is_game_over = True
                    if is_game_over:
                        done_a = True
                    else:
                        obs, info = env_a.reset()
                        fs_a = [cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)] * 4
                        fs_a = [cv2.resize(g, (84, 84), interpolation=cv2.INTER_AREA) for g in fs_a]
                        continue
                else:
                    update_frame_stack(fs_a, obs)

            # --- Step side B ---
            if not done_b:
                obs, reward, terminated, truncated, info = env_b.step(act_b)
                b_score += float(reward)
                try:
                    px = int(get_ram(env_b)[SHIP_X])
                except Exception:
                    px = -1
                b_ship_x.append(px)

                if terminated or truncated:
                    try:
                        is_game_over = env_b.unwrapped.ale.lives() == 0
                    except Exception:
                        is_game_over = True
                    if is_game_over:
                        done_b = True
                    else:
                        obs, info = env_b.reset()
                        fs_b = [cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)] * 4
                        fs_b = [cv2.resize(g, (84, 84), interpolation=cv2.INTER_AREA) for g in fs_b]
                        continue
                else:
                    update_frame_stack(fs_b, obs)

        env_a.close()
        env_b.close()

        # Metrics
        div_pct = (diverged_frames / compared_frames * 100) if compared_frames > 0 else 0.0

        min_len = min(len(a_ship_x), len(b_ship_x))
        if min_len > 2:
            a_px = np.array(a_ship_x[:min_len])
            b_px = np.array(b_ship_x[:min_len])
            px_corr = np.corrcoef(a_px, b_px)[0, 1]
        else:
            px_corr = 0.0

        score_ret = (b_score / a_score * 100) if a_score > 0 else 0.0

        # Perfect transfer: ship moves identically despite different enemies
        is_perfect_transfer = (px_corr > 0.99 and score_ret > 80)

        full_scores_all.append(a_score)
        alt_scores_all.append(b_score)
        px_corrs_all.append(px_corr)
        divergences_all.append(div_pct)
        perfect_transfers_all.append(is_perfect_transfer)

        marker = " *** PERFECT TRANSFER ***" if is_perfect_transfer else ""
        print(f"  Game {g+1:2d}: {compared_frames:5d}f  |  "
              f"A={a_score:.0f}  B={b_score:.0f} ({score_ret:.0f}%)  |  "
              f"actions diverged: {diverged_frames}/{compared_frames} ({div_pct:.1f}%)  "
              f"ship_x_corr={px_corr:.4f}{marker}")

    # -----------------------------------------------------------------------
    # Overall verdict
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("OVERALL VERDICT")
    print("=" * 70)

    n_perfect = sum(perfect_transfers_all)
    n_total = len(perfect_transfers_all)
    avg_div = np.mean(divergences_all) if divergences_all else 0
    avg_full = np.mean(full_scores_all) if full_scores_all else 0
    avg_retention = np.mean([b / a * 100 for a, b in zip(alt_scores_all, full_scores_all)
                             if a > 0]) if full_scores_all else 0

    # Also show score std for side A (same as noop=30, det=True condition)
    unique_a = len(set(round(s) for s in full_scores_all))
    std_a = np.std(full_scores_all) if len(full_scores_all) > 1 else 0

    print(f"  Side A (FULL):  mean={avg_full:.1f}  std={std_a:.1f}  unique={unique_a}")
    print(f"  Games with perfect transfer (ship_x_corr>0.99, B~=A): {n_perfect}/{n_total}")
    print(f"  Avg action divergence: {avg_div:.1f}%")
    print(f"  Avg score retention (B/A): {avg_retention:.0f}%")
    print()

    # Verdict logic:
    #   1. DEAD: FULL score near zero
    #   2. SINGLE_SCRIPT on side A: std=0, unique=1 -> MEMORIZED (definitive)
    #      Different noop offsets produce identical scores = the argmax is a
    #      fixed sequence regardless of game state.
    #   3. Perfect transfer: ship_x_corr>0.99, B~=A -> MEMORIZED (definitive)
    #   4. Otherwise: check retention for INCONCLUSIVE vs scrambled cues
    if avg_full < 5:
        print(f"  VERDICT: DEAD")
        print(f"  Side A score ({avg_full:.0f} pts) near zero. Policy never learned")
        print(f"  to play BeamRider.")
    elif unique_a == 1 and std_a < 1.0:
        print(f"  VERDICT: MEMORIZED SCRIPT (SINGLE_SCRIPT)")
        print(f"  Side A produced exactly ONE score ({avg_full:.0f} pts) across all")
        print(f"  {n_total} games with different noop offsets. A reactive policy would")
        print(f"  produce different scores from different starting states.")
        print(f"  std={std_a:.1f}, unique={unique_a} -- this is the definition of a")
        print(f"  memorized argmax sequence.")
    elif n_perfect > 0:
        print(f"  VERDICT: MEMORIZED SCRIPT (perfect transfer)")
        print(f"  {n_perfect}/{n_total} games showed PERFECT TRANSFER -- identical ship")
        print(f"  movement on a DIFFERENT enemy pattern. A reactive policy cannot do")
        print(f"  this: different enemy positions -> different threats -> different")
        print(f"  evasion responses -> different ship positions.")
    elif avg_retention > 60:
        print(f"  VERDICT: INCONCLUSIVE (promising)")
        print(f"  No perfect-transfer games and no SINGLE_SCRIPT detected. Ship positions")
        print(f"  diverge ({avg_div:.0f}% action divergence) and score retention is")
        print(f"  {avg_retention:.0f}%. Side A: {unique_a} unique scores, std={std_a:.1f}.")
        print(f"  Could be reactive -- run more games or verify with other diagnostics.")
    else:
        print(f"  VERDICT: MEMORIZED (scrambled cues)")
        print(f"  No perfect-transfer games, but ALT retention is only {avg_retention:.0f}%.")
        print(f"  Different visual inputs scramble the CNN features that trigger the")
        print(f"  memorized sequence. Action divergence ({avg_div:.0f}%) is from unfamiliar")
        print(f"  pixels, not reactivity to enemy state.")

    print()
    print("How to read this:")
    print("  Perfect transfer (ship_x_corr>0.99, B~=A) = DEFINITIVE memorization")
    print("  A reactive policy CANNOT move identically on different enemy patterns.")
    print("  Action divergence alone is a confound -- scrambled visual cues also")
    print("  cause divergence in memorized policies.")
