"""
Intervention Gradient -- Dose-Response Curve for Ball-Tracking Reactivity.

Extends the intervention probe to test MULTIPLE teleport magnitudes in one run,
producing a reversal-rate vs. displacement curve. Fills the gap where the
binary probe can only track one number -- the gradient lets us track partial
progress over training and compare models on a continuum.

Key metrics:
  - Reversal AUC: area under reversal-rate vs displacement curve (0 to 1)
  - Half-max displacement: teleport distance where reversal drops to 50% of max
  - Per-magnitude reversal rates at +/-0, +/-8, +/-15, +/-30, +/-45, +/-60px

Dead baseline: center-hold produces ~0% reversal at ALL magnitudes.
Reactive policy: sigmoid-like falloff with increasing displacement.

Usage:
    python intervention_gradient.py
    python intervention_gradient.py --model ./models/PPO_114/best_model.zip --teleports 20
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

# Default magnitudes to sweep
DEFAULT_MAGNITUDES = [0, 8, 15, 30, 45, 60]


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


def set_ram(env, addr, val):
    env.venv.envs[0].unwrapped.ale.setRAM(addr, val)


# -- Intervention at a single magnitude ----------------------------------

def run_intervention_at_magnitude(model, env, teleport_px, n_teleports_target,
                                   track_window=30, rng=None):
    """Run the intervention probe at one teleport magnitude.

    Returns results list (same format as probe_107_intervention.py).
    """
    if rng is None:
        rng = np.random.default_rng()
    results = []
    obs = env.reset()
    game_frame = 0
    frames_since_teleport = 0
    game = 0

    while len(results) < n_teleports_target:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        game_frame += 1
        frames_since_teleport += 1

        if game_frame >= 200 and frames_since_teleport >= 80:
            ram = get_ram(env)
            bx, by, px = int(ram[BALL_X]), int(ram[BALL_Y]), int(ram[PADDLE_X])

            if 40 <= by <= 140:
                # Record pre-teleport positions
                pre_positions = [px]
                for _ in range(3):
                    pre_action, _states = model.predict(obs, deterministic=True)
                    obs, pre_rew, pre_done, info = env.step(pre_action)
                    pre_ram = get_ram(env)
                    pre_positions.append(int(pre_ram[PADDLE_X]))
                    game_frame += 1

                pre_px = pre_positions[-1]
                pre_dir = np.sign(pre_positions[-1] - pre_positions[0])

                # Teleport (or control at px=0)
                bx = int(get_ram(env)[BALL_X])
                direction = rng.choice([-1, 1])
                new_bx = max(8, min(152, bx + direction * teleport_px))
                set_ram(env, BALL_X, new_bx)

                # Track post-teleport
                post_positions = [pre_px]
                for _ in range(track_window):
                    track_action, _states = model.predict(obs, deterministic=True)
                    obs, track_rew, track_done, info = env.step(track_action)
                    track_ram = get_ram(env)
                    post_positions.append(int(track_ram[PADDLE_X]))
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
                    'magnitude': teleport_px,
                })

                frames_since_teleport = 0

        if done[0]:
            obs = env.reset()
            game += 1
            game_frame = 0
            frames_since_teleport = 0

    return results


# -- Dead baseline at one magnitude --------------------------------------

def dead_baseline_at_magnitude(env, teleport_px, n_teleports_target,
                                 track_window=30, rng=None):
    """Center-hold dead baseline at one teleport magnitude."""
    if rng is None:
        rng = np.random.default_rng()
    results = []
    obs = env.reset()
    game_frame = 0
    frames_since_teleport = 0

    while len(results) < n_teleports_target:
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
                pre_positions = [px]
                for _ in range(3):
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

                bx = int(get_ram(env)[BALL_X])
                direction = rng.choice([-1, 1])
                new_bx = max(8, min(152, bx + direction * teleport_px))
                set_ram(env, BALL_X, new_bx)

                post_positions = [pre_px]
                for _ in range(track_window):
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

                reversed_toward = False
                if pre_dir != 0:
                    if direction == 1:
                        reversed_toward = (pre_dir == -1 and post_dir >= 0)
                    else:
                        reversed_toward = (pre_dir == 1 and post_dir <= 0)

                results.append({
                    'pre_dir': pre_dir,
                    'post_dir': post_dir,
                    'reversed_toward': reversed_toward,
                    'magnitude': teleport_px,
                })

                frames_since_teleport = 0

        if done[0]:
            obs = env.reset()
            game_frame = 0
            frames_since_teleport = 0

    return results


# -- AUC computation -----------------------------------------------------

def compute_auc(magnitudes, reversal_rates):
    """Compute area under the reversal-rate curve using trapezoidal rule.

    Normalized to [0, 1] by dividing by max_magnitude.
    """
    if len(magnitudes) < 2:
        return None
    # Sort by magnitude
    order = np.argsort(magnitudes)
    mags = np.array(magnitudes)[order]
    rates = np.array(reversal_rates)[order]
    # Normalize to 0-1 range: convert rates to fractions, x to [0, 1]
    rates_frac = np.array(reversal_rates)[order] / 100.0
    x_norm = mags / mags[-1] if mags[-1] > 0 else mags
    # Manual trapezoidal integration
    auc = float(np.sum((rates_frac[1:] + rates_frac[:-1]) / 2 * (x_norm[1:] - x_norm[:-1])))
    return auc


def compute_half_max(magnitudes, reversal_rates):
    """Find the displacement at which reversal drops to 50% of its max value.

    Interpolates between data points. Returns None if never drops below 50%.
    """
    if len(magnitudes) < 2:
        return None
    order = np.argsort(magnitudes)
    mags = np.array(magnitudes)[order]
    rates = np.array(reversal_rates)[order]
    max_rate = rates[0]  # should be the control (+/-0px) rate
    if max_rate <= 0:
        return None
    half = max_rate / 2
    # Find where rate crosses below half
    for i in range(len(mags) - 1):
        if rates[i] >= half and rates[i+1] <= half:
            # Linear interpolation
            frac = (half - rates[i]) / (rates[i+1] - rates[i])
            return float(mags[i] + frac * (mags[i+1] - mags[i]))
    return None


# -- Display -------------------------------------------------------------

def print_gradient_table(magnitudes, model_rates, dead_rates, model_moving, dead_moving):
    """Print per-magnitude reversal rates in a comparison table."""
    order = np.argsort(magnitudes)
    print(f"\n{'='*70}")
    print(f"  INTERVENTION GRADIENT -- Per-Magnitude Reversal Rates")
    print(f"{'='*70}")
    print(f"  {'Magnitude':>12}  {'Dead rev':>10}  {'Model rev':>10}  {'Delta':>10}  {'Interpretation':>20}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*20}")

    for i in order:
        mag = magnitudes[i]
        dr = dead_rates[i]
        mr = model_rates[i]
        delta = mr - dr
        if delta > 15:
            interp = "STRONG reactivity"
        elif delta > 5:
            interp = "WEAK reactivity"
        else:
            interp = "indistinguishable"

        print(f"  +/-{mag:>3}px          {dr:>5.1f}%       {mr:>5.1f}%       {delta:>+6.1f}%       {interp:<20}")


# -- Main ----------------------------------------------------------------

if __name__ == "__main__":
    MODEL_PATH = "./models/PPO_107/best_model.zip"
    RUN_NAME = "PPO_107"
    N_TELEPORTS = 20  # per magnitude
    MAGNITUDES = DEFAULT_MAGNITUDES

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == '--model': MODEL_PATH = args[i + 1]; i += 2
        elif args[i] == '--run-name': RUN_NAME = args[i + 1]; i += 2
        elif args[i] == '--teleports': N_TELEPORTS = int(args[i + 1]); i += 2
        elif args[i] == '--magnitudes':
            MAGNITUDES = [int(x) for x in args[i + 1].split(',')]; i += 2
        else: i += 1

    if RUN_NAME == "PPO_107" and MODEL_PATH != "./models/PPO_107/best_model.zip":
        import re
        m = re.search(r'PPO_\d+[a-z]?', MODEL_PATH)
        if m:
            RUN_NAME = m.group(0)

    print("=" * 70)
    print(f"{RUN_NAME} Intervention Gradient -- Dose-Response Curve")
    print("=" * 70)
    print(f"Model: {MODEL_PATH}")
    print(f"Magnitudes: {MAGNITUDES}")
    print(f"Teleports per magnitude: {N_TELEPORTS}")
    print(f"Total teleports: {len(MAGNITUDES) * N_TELEPORTS}")
    print()

    rng = np.random.default_rng(42)

    # --- Dead baseline (all magnitudes) ---
    print("--- Dead baseline (center-hold) ---")
    dead_env = make_vec_env()
    dead_rates = []
    dead_moving = []
    t0 = time.time()

    for mag in MAGNITUDES:
        dead_results = dead_baseline_at_magnitude(dead_env, mag, N_TELEPORTS, rng=rng)
        moving = sum(1 for r in dead_results if r['pre_dir'] != 0)
        rev = sum(1 for r in dead_results if r['reversed_toward'])
        rate = rev / moving * 100 if moving > 0 else 0
        dead_rates.append(rate)
        dead_moving.append(moving)
        print(f"  +/-{mag:>3}px: {rev}/{moving} reversed = {rate:.1f}%")

    dead_env.close()
    print(f"  ({time.time() - t0:.0f}s total)")

    # --- Model (all magnitudes) ---
    print(f"\n--- {RUN_NAME} ---")
    print("Loading model...")
    env = make_vec_env()
    model = PPO.load(MODEL_PATH, env=env, device="cuda")
    print(f"Loaded. Model step count: {model.num_timesteps:,}")

    model_rates = []
    model_moving = []
    t0 = time.time()

    for mag in MAGNITUDES:
        results = run_intervention_at_magnitude(model, env, mag, N_TELEPORTS, rng=rng)
        moving = sum(1 for r in results if r['pre_dir'] != 0)
        rev = sum(1 for r in results if r['reversed_toward'])
        rate = rev / moving * 100 if moving > 0 else 0
        model_rates.append(rate)
        model_moving.append(moving)
        print(f"  +/-{mag:>3}px: {rev}/{moving} reversed = {rate:.1f}%")

    env.close()
    print(f"  ({time.time() - t0:.0f}s total)")

    # --- Summary ---
    print_gradient_table(MAGNITUDES, model_rates, dead_rates, model_moving, dead_moving)

    # --- AUC ---
    auc = compute_auc(MAGNITUDES, model_rates)
    dead_auc = compute_auc(MAGNITUDES, dead_rates)
    half_max = compute_half_max(MAGNITUDES, model_rates)

    print(f"\n{'='*70}")
    print(f"  SUMMARY METRICS")
    print(f"{'='*70}")
    print(f"  Reversal AUC:        {auc:.4f}" if auc is not None else "  Reversal AUC:        N/A")
    if dead_auc is not None:
        print(f"  Dead AUC:            {dead_auc:.4f}")
    print(f"  Half-max displacement: {half_max:.0f}px" if half_max is not None
          else "  Half-max displacement: N/A (never drops below 50%)")
    print(f"  Control (+/-0px):       {model_rates[0]:.1f}% reversal"
          f" (should be ~50% -- pure chance, no information)")

    print(f"\n  INTERPRETATION:")
    if auc is not None:
        if auc > 0.25:
            print(f"  AUC={auc:.3f} -> STRONG dose-response curve. Policy tracks ball across wide displacements.")
        elif auc > 0.10:
            print(f"  AUC={auc:.3f} -> MODERATE dose-response. Ball tracking degrades with displacement.")
        elif auc > 0.03:
            print(f"  AUC={auc:.3f} -> WEAK dose-response. Marginal ball tracking at small displacements only.")
        else:
            print(f"  AUC={auc:.3f} -> FLAT. No dose-response -- policy is a memorized script.")

    if half_max is not None:
        print(f"  Reversal halves at +/-{half_max:.0f}px displacement.")
        print(f"  Larger half-max = wider tracking window = more robust reactivity.")

    # --- ASCII curve ---
    print(f"\n{'='*70}")
    print(f"  REVERSAL RATE vs DISPLACEMENT")
    print(f"{'='*70}")
    max_rate = max(max(model_rates), max(dead_rates), 1)
    for row in range(20, -1, -1):
        y = row / 20 * max_rate
        line = f"  {y:>5.1f}% |"
        for i in np.argsort(MAGNITUDES):
            mr = model_rates[i]
            dr = dead_rates[i]
            if mr >= y:
                line += " #"
            elif dr >= y:
                line += " ."
            else:
                line += "  "
        print(line)
    print(f"  {'':>6} +" + "--" * len(MAGNITUDES))
    mag_labels = "".join(f"{m:>3}" for m in np.sort(MAGNITUDES))
    print(f"  {'':>6}  {mag_labels} px")
    print(f"  # = {RUN_NAME}   . = dead baseline")
