"""
Headless split-watcher verification — quantitative paddle position comparison.

Runs the same model on FULL vs ALTERED brick layout side-by-side.
Same argmax action drives both sides. Compares paddle positions frame by frame.

A memorized script: identical paddle positions on both sides.
A reactive policy:  paddle positions diverge because the ball bounces
                    differently and the policy tracks it.

Usage:
    python verify_split_watcher.py
    python verify_split_watcher.py --model ./models/PPO_116/best_model.zip
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

BALL_X, BALL_Y, PADDLE_X = 99, 101, 72
NOOP, FIRE, RIGHT, LEFT = 0, 1, 2, 3

# ---------------------------------------------------------------------------
# Wrappers (mirror watch_model_split.py exactly)
# ---------------------------------------------------------------------------

class BrickClearWrapper(gym.Wrapper):
    def __init__(self, env, clear_addrs=None):
        super().__init__(env)
        self._static_addrs = None
        self._random_pct = None
        self._rng = np.random.default_rng()
        if isinstance(clear_addrs, str) and clear_addrs.startswith("random_"):
            self._random_pct = int(clear_addrs.split("_")[1]) / 100.0
        else:
            self._static_addrs = list(clear_addrs or [])

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self._random_pct is not None:
            n_clear = max(1, int(36 * self._random_pct))
            addrs = list(self._rng.choice(36, size=n_clear, replace=False))
        else:
            addrs = self._static_addrs
        for addr in addrs:
            self.unwrapped.ale.setRAM(addr, 0)
        # Take one NOOP step to refresh the observation — otherwise the
        # returned obs shows the FULL brick wall, not the cleared layout.
        # The model would see identical first frames on both sides, masking
        # genuine early reactivity.
        obs, _, _, _, _ = self.env.step(0)
        return obs, info


class AutoResetWrapper(gym.Wrapper):
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

def make_raw_env(brick_addrs=None):
    """Build a raw (non-vec) Breakout env matching watch_model_split."""
    env = gym.make("ALE/Breakout-v5", frameskip=4, repeat_action_probability=0)
    env = NoopResetEnv(env, noop_max=30)
    env = FireResetEnv(env)
    if brick_addrs is not None:
        env = BrickClearWrapper(env, clear_addrs=brick_addrs)
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
    MODEL_PATH = "./models/PPO_115/final_model.zip"
    N_GAMES = 4
    MAX_FRAMES = 6000

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--model":
            MODEL_PATH = args[i + 1]; i += 2
        elif args[i] == "--games":
            N_GAMES = int(args[i + 1]); i += 2
        else:
            i += 1

    m = re.search(r"PPO_\d+[a-z]?", MODEL_PATH)
    run_name = m.group(0) if m else "model"

    # Load model with vec env (training-pipeline-compatible)
    def _make_dummy():
        e = gym.make("ALE/Breakout-v5", frameskip=4, repeat_action_probability=0)
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
    print(f"Split-Watcher Verification — {run_name}")
    print(f"{'='*70}")
    print(f"Model: {MODEL_PATH} @ {model.num_timesteps:,} steps")
    print(f"Inference: deterministic (argmax)")
    print(f"Games per layout pair: {N_GAMES}")
    print()
    print("Principle: INDEPENDENT predictions per side.")
    print("  Memorized: actions IDENTICAL on both sides (same sequence regardless of bricks)")
    print("  Reactive:  actions DIVERGE when ball positions differ")
    print()

    LAYOUTS = [
        ("RIGHT_HALF", list(range(0, 18))),
        ("LEFT_HALF", list(range(18, 36))),
        ("RANDOM_50", "random_50"),
    ]

    full_scores_all = []
    alt_scores_all = []
    px_corrs_all = []
    divergences_all = []
    perfect_transfers_all = []

    for layout_name, layout_addrs in LAYOUTS:
        for g in range(N_GAMES):
            env_full = make_raw_env(brick_addrs=None)
            env_alt = make_raw_env(brick_addrs=layout_addrs)

            obs_full, _info = env_full.reset()
            obs_alt, _info = env_alt.reset()

            fs_full = initial_frame_stack(obs_full)
            fs_alt = initial_frame_stack(obs_alt)

            full_paddle = []
            alt_paddle = []
            full_actions = []
            alt_actions = []
            full_score = 0.0
            alt_score = 0.0
            diverged_frames = 0
            compared_frames = 0

            done_full = False
            done_alt = False
            step = 0

            while not (done_full and done_alt) and step < MAX_FRAMES:
                step += 1

                # --- Predict INDEPENDENTLY for each side ---
                # Left (FULL) prediction
                if not done_full:
                    left_obs = np.expand_dims(fs_full, axis=0)
                    left_action, _ = model.predict(left_obs, deterministic=True)
                    left_act = int(left_action[0])
                else:
                    left_act = NOOP

                # Right (ALT) prediction
                if not done_alt:
                    right_obs = np.expand_dims(fs_alt, axis=0)
                    right_action, _ = model.predict(right_obs, deterministic=True)
                    right_act = int(right_action[0])
                else:
                    right_act = NOOP

                # Compare actions (both sides alive = valid comparison)
                if not done_full and not done_alt:
                    full_actions.append(left_act)
                    alt_actions.append(right_act)
                    if left_act != right_act:
                        diverged_frames += 1
                    compared_frames += 1

                # --- Step FULL side ---
                if not done_full:
                    try:
                        ram = get_ram(env_full)
                        needs_serve = int(ram[BALL_Y]) > 180
                    except Exception:
                        needs_serve = False
                    act = FIRE if needs_serve else left_act
                    obs, reward, terminated, truncated, info = env_full.step(act)
                    full_score += float(reward)
                    try:
                        px = int(get_ram(env_full)[PADDLE_X])
                    except Exception:
                        px = -1
                    full_paddle.append(px)

                    if terminated or truncated:
                        try:
                            is_game_over = env_full.unwrapped.ale.lives() == 0
                        except Exception:
                            is_game_over = True
                        if is_game_over:
                            done_full = True
                        else:
                            obs, info = env_full.reset()
                            fs_full = [cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)] * 4
                            fs_full = [cv2.resize(g, (84, 84), interpolation=cv2.INTER_AREA) for g in fs_full]
                            continue
                    else:
                        update_frame_stack(fs_full, obs)

                # --- Step ALT side ---
                if not done_alt:
                    try:
                        ram = get_ram(env_alt)
                        needs_serve = int(ram[BALL_Y]) > 180
                    except Exception:
                        needs_serve = False
                    act = FIRE if needs_serve else right_act
                    obs, reward, terminated, truncated, info = env_alt.step(act)
                    alt_score += float(reward)
                    try:
                        px = int(get_ram(env_alt)[PADDLE_X])
                    except Exception:
                        px = -1
                    alt_paddle.append(px)

                    if terminated or truncated:
                        try:
                            is_game_over = env_alt.unwrapped.ale.lives() == 0
                        except Exception:
                            is_game_over = True
                        if is_game_over:
                            done_alt = True
                        else:
                            obs, info = env_alt.reset()
                            fs_alt = [cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)] * 4
                            fs_alt = [cv2.resize(g, (84, 84), interpolation=cv2.INTER_AREA) for g in fs_alt]
                            continue
                    else:
                        update_frame_stack(fs_alt, obs)

            env_full.close()
            env_alt.close()

            # The KEY signal: px_corr ~= 1.0 on a DIFFERENT layout.
            # A reactive policy CANNOT produce identical paddle positions on
            # different brick layouts — different bricks -> different ball
            # bounces -> different tracking responses -> different positions.
            # px_corr=1.0 + ALT~=FULL = memorized script that happens to work
            # on this layout. This is definitive.
            div_pct = (diverged_frames / compared_frames * 100) if compared_frames > 0 else 0.0

            # Also show paddle correlation for reference
            min_len = min(len(full_paddle), len(alt_paddle))
            if min_len > 2:
                full_px = np.array(full_paddle[:min_len])
                alt_px = np.array(alt_paddle[:min_len])
                px_corr = np.corrcoef(full_px, alt_px)[0, 1]
            else:
                px_corr = 0.0

            # Score retention for this game
            score_ret = (alt_score / full_score * 100) if full_score > 0 else 0.0

            # Perfect transfer detection: px_corr > 0.99 AND ALT ~= FULL
            is_perfect_transfer = (px_corr > 0.99 and score_ret > 80)

            full_scores_all.append(full_score)
            alt_scores_all.append(alt_score)
            px_corrs_all.append(px_corr)
            divergences_all.append(div_pct)
            perfect_transfers_all.append(is_perfect_transfer)

            marker = " *** PERFECT TRANSFER ***" if is_perfect_transfer else ""
            print(f"  {layout_name} game {g+1}: {compared_frames}f  |  "
                  f"FULL={full_score:.0f}  ALT={alt_score:.0f} ({score_ret:.0f}%)  |  "
                  f"actions diverged: {diverged_frames}/{compared_frames} ({div_pct:.1f}%)  "
                  f"px_corr={px_corr:.4f}{marker}")

    # -----------------------------------------------------------------------
    # Overall verdict
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("OVERALL VERDICT")
    print("=" * 70)

    n_perfect = sum(perfect_transfers_all)
    n_total = len(perfect_transfers_all)
    avg_div = np.mean(divergences_all) if divergences_all else 0
    avg_retention = np.mean([a / f * 100 for a, f in zip(alt_scores_all, full_scores_all)
                             if f > 0]) if full_scores_all else 0

    print(f"  Games with perfect transfer (px_corr>0.99, ALT~=FULL): {n_perfect}/{n_total}")
    print(f"  Avg action divergence: {avg_div:.1f}%")
    print(f"  Avg ALT score retention: {avg_retention:.0f}%")
    print()

    # Verdict logic:
    #   Perfect transfer = px_corr>0.99 AND ALT~=FULL on a DIFFERENT layout.
    #   This is physically impossible for a reactive policy — different bricks
    #   cause different ball bounces, which force different paddle movements.
    #   Only a memorized sequence that ignores ball position can do this.
    #
    #   Memorized:     any perfect-transfer game detected -> script confirmed
    #   Dead:          FULL score ~= 0 -> never learned anything
    #   Inconclusive:  no perfect transfers, but some ALT retention -> need
    #                  more games or intervention probe to determine

    avg_full = np.mean(full_scores_all) if full_scores_all else 0

    if avg_full < 5:
        print(f"  VERDICT: DEAD")
        print(f"  FULL score ({avg_full:.0f} pts) near zero. Policy never learned")
        print(f"  to play Breakout on ANY layout.")
    elif n_perfect > 0:
        print(f"  VERDICT: MEMORIZED SCRIPT")
        print(f"  {n_perfect}/{n_total} games showed PERFECT TRANSFER — identical paddle")
        print(f"  movement on a DIFFERENT brick layout. A reactive policy cannot do")
        print(f"  this: different bricks -> different ball bounces -> different")
        print(f"  tracking responses -> different paddle positions.")
        print(f"  px_corr>0.99 on an altered layout is definitive evidence of a")
        print(f"  memorized action sequence that ignores ball position.")
    elif avg_retention > 60:
        print(f"  VERDICT: INCONCLUSIVE (promising)")
        print(f"  No perfect-transfer games detected, and ALT score retention")
        print(f"  is {avg_retention:.0f}%. Could be reactive — run more games or")
        print(f"  verify with intervention probe.")
    else:
        print(f"  VERDICT: MEMORIZED (scrambled cues)")
        print(f"  No perfect-transfer games, but ALT retention is only {avg_retention:.0f}%.")
        print(f"  Different brick layouts scramble the visual cues that trigger the")
        print(f"  memorized sequence. Actions diverge ({avg_div:.0f}%) because the CNN")
        print(f"  sees unfamiliar pixels, not because it tracks the ball.")

    print()
    print("How to read this:")
    print("  Perfect transfer (px_corr>0.99, ALT~=FULL) = DEFINITIVE memorization")
    print("  A reactive policy CANNOT move identically on different brick layouts.")
    print("  Action divergence alone is a confound — scrambled visual cues also")
    print("  cause divergence in memorized policies.")
