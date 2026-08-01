"""
Feature-Attribution Probe -- tests whether a trained PPO CNN encodes ball position.

Freezes the CNN from a PPO checkpoint, extracts 512-dim feature vectors for
thousands of frames with known ball positions, then trains a linear regressor
to predict (ball_x, ball_y) from those features. MAE tells us whether the CNN
preserved ball-position information through training.

Key question: is the perception-policy gap "sees the ball but ignores it"
(MAE ~2px) or "stopped seeing the ball" (MAE > 20px)?

Baselines:
  - Supervised POC: 1.9px MAE (gold standard -- a dedicated ball-tracking CNN)
  - Random features (untrained CNN): ~40px MAE (chance-level for 160px range)
  - A PPO model that unlearned ball features: >20px MAE (perception collapse)

Usage:
    python feature_attribution_probe.py --model ./models/PPO_114/best_model.zip
    python feature_attribution_probe.py --checkpoints ./models/PPO_114/checkpoint/

Collects ~2000 frames using center-hold script + model's own gameplay for
both in-distribution and out-of-distribution coverage.
"""
import sys
import time
import os
import glob
import numpy as np
import gymnasium as gym
import cv2
import torch as th
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import FireResetEnv, NoopResetEnv, EpisodicLifeEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import ale_py
gym.register_envs(ale_py)

BALL_X, BALL_Y, PADDLE_X = 99, 101, 72
NOOP, FIRE, RIGHT, LEFT = 0, 1, 2, 3
SUPERVISED_MAE_BASELINE = 1.9  # px -- from perception POC


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


# -- Data collection ----------------------------------------------------

def collect_frames_with_labels(env, model, n_frames, collect_mode="mixed"):
    """Collect (observation, ball_x, ball_y) pairs.

    Args:
        env: VecFrameStack env
        model: SB3 PPO model (can be None for script-only collection)
        n_frames: target number of frames
        collect_mode: "center_hold", "model", or "mixed"
    """
    frames = []
    labels = []  # (ball_x, ball_y) per frame
    obs = env.reset()
    game_frame = 0

    while len(frames) < n_frames:
        ram = get_ram(env)
        bx, by = int(ram[BALL_X]), int(ram[BALL_Y])
        px = int(ram[PADDLE_X])

        # Record this frame
        # Observation shape is (1, 4, 84, 84) -- store the full frame stack
        frames.append(obs.copy())
        labels.append((bx, by))

        # Choose action
        if collect_mode == "center_hold" or model is None:
            if by > 180:
                action = FIRE
            elif px < 76:
                action = RIGHT
            elif px > 84:
                action = LEFT
            else:
                action = NOOP
            action_arr = [action]
        elif collect_mode == "model":
            action_arr, _states = model.predict(obs, deterministic=True)
        else:  # mixed: alternate
            if game_frame % 3 == 0:
                action_arr, _states = model.predict(obs, deterministic=True)
            else:
                if by > 180:
                    action = FIRE
                elif px < 76:
                    action = RIGHT
                elif px > 84:
                    action = LEFT
                else:
                    action = NOOP
                action_arr = [action]

        obs, reward, done, info = env.step(action_arr)
        game_frame += 1

        if done[0]:
            obs = env.reset()
            game_frame = 0

    return frames, labels


# -- Feature extraction -------------------------------------------------

def extract_features(model, frames, device="cuda", batch_size=128):
    """Extract 512-dim feature vectors from frozen PPO CNN.

    Args:
        model: SB3 PPO model
        frames: list of observations, each (1, 4, 84, 84)
        device: torch device

    Returns:
        features: np.array (n_frames, 512)
    """
    features_list = []
    features_extractor = model.policy.features_extractor
    features_extractor.eval()

    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i + batch_size]
        # VecFrameStack returns NHWC (1, 84, 84, 4) -> NCHW (batch, 4, 84, 84)
        batch_nhwc = np.concatenate(batch_frames, axis=0)
        batch_nchw = np.transpose(batch_nhwc, (0, 3, 1, 2)).copy()
        batch_tensor = th.tensor(batch_nchw).float().to(device)
        with th.no_grad():
            feats = features_extractor(batch_tensor)
        features_list.append(feats.cpu().numpy())

    return np.concatenate(features_list, axis=0)


def extract_features_random_baseline(n_frames, feature_dim=512, device="cuda"):
    """Generate random features for the untrained-CNN baseline.

    A randomly-initialized NatureCNN produces near-random features.
    We approximate this with Gaussian noise at the scale of ReLU outputs
    (non-negative, mean ~0.5-1.0 for 512-dim after ReLU).
    """
    rng = np.random.default_rng(42)
    # ReLU outputs are non-negative; approximate with folded normal
    return np.abs(rng.normal(loc=1.0, scale=0.8, size=(n_frames, feature_dim)))


# -- Regression ---------------------------------------------------------

def train_and_evaluate(features, labels, test_size=0.2, random_state=42):
    """Train linear regression on features -> (ball_x, ball_y), report MAE.

    Returns dict with train/test MAE for x and y.
    """
    n = len(features)
    n_train = int(n * (1 - test_size))
    indices = np.random.default_rng(random_state).permutation(n)
    train_idx, test_idx = indices[:n_train], indices[n_train:]

    X_train, X_test = features[train_idx], features[test_idx]
    y_train, y_test = np.array(labels)[train_idx], np.array(labels)[test_idx]

    # Train separate regressors for x and y
    reg_x = LinearRegression().fit(X_train, y_train[:, 0])
    reg_y = LinearRegression().fit(X_train, y_train[:, 1])

    # Predictions
    pred_x_train, pred_x_test = reg_x.predict(X_train), reg_x.predict(X_test)
    pred_y_train, pred_y_test = reg_y.predict(X_train), reg_y.predict(X_test)

    # MAE
    mae_x_train = mean_absolute_error(y_train[:, 0], pred_x_train)
    mae_x_test = mean_absolute_error(y_test[:, 0], pred_x_test)
    mae_y_train = mean_absolute_error(y_train[:, 1], pred_y_train)
    mae_y_test = mean_absolute_error(y_test[:, 1], pred_y_test)

    # Combined Euclidean error
    euclidean_train = np.sqrt((y_train[:, 0] - pred_x_train)**2 + (y_train[:, 1] - pred_y_train)**2)
    euclidean_test = np.sqrt((y_test[:, 0] - pred_x_test)**2 + (y_test[:, 1] - pred_y_test)**2)

    # R² scores
    r2_x = reg_x.score(X_test, y_test[:, 0])
    r2_y = reg_y.score(X_test, y_test[:, 1])

    # Chance-level baseline: always predict mean
    mean_x, mean_y = y_train[:, 0].mean(), y_train[:, 1].mean()
    chance_mae_x = mean_absolute_error(y_test[:, 0], np.full_like(y_test[:, 0], mean_x))
    chance_mae_y = mean_absolute_error(y_test[:, 1], np.full_like(y_test[:, 1], mean_y))
    chance_euclidean = np.sqrt((y_test[:, 0] - mean_x)**2 + (y_test[:, 1] - mean_y)**2)

    return {
        "mae_x_train": float(mae_x_train), "mae_x_test": float(mae_x_test),
        "mae_y_train": float(mae_y_train), "mae_y_test": float(mae_y_test),
        "euclidean_train": float(euclidean_train.mean()),
        "euclidean_test": float(euclidean_test.mean()),
        "r2_x": float(r2_x), "r2_y": float(r2_y),
        "chance_mae_x": float(chance_mae_x), "chance_mae_y": float(chance_mae_y),
        "chance_euclidean": float(chance_euclidean.mean()),
        "n_train": n_train, "n_test": n - n_train,
    }


# -- Random-feature baseline --------------------------------------------

def random_feature_baseline(labels, n_trials=5, feature_dim=512):
    """Run multiple trials with random features to establish chance-level MAE."""
    results = []
    for trial in range(n_trials):
        random_feats = extract_features_random_baseline(len(labels), feature_dim)
        res = train_and_evaluate(random_feats, labels, random_state=42 + trial)
        results.append(res)
    # Average
    avg = {}
    for key in results[0]:
        avg[key] = float(np.mean([r[key] for r in results]))
    avg["n_trials"] = n_trials
    return avg


# -- Display -------------------------------------------------------------

def print_results(model_name, model_step, results, chance_results):
    """Print formatted feature-attribution results."""
    print(f"\n{'='*70}")
    print(f"  Feature-Attribution Probe -- {model_name}")
    print(f"{'='*70}")
    if model_step:
        print(f"  Model step count: {model_step:,}")
    print(f"  Training frames: {results['n_train']:,}  Test frames: {results['n_test']:,}")
    print()

    # Per-axis MAE
    print(f"  -- Ball Position MAE (px) --")
    print(f"  {'Axis':<10} {'Train':>8} {'Test':>8} {'Chance':>8} {'Supervised':>12}")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*12}")
    print(f"  {'ball_x':<10} {results['mae_x_train']:>7.1f}px {results['mae_x_test']:>7.1f}px "
          f"{chance_results['mae_x_test']:>7.1f}px {SUPERVISED_MAE_BASELINE:>11.1f}px")
    print(f"  {'ball_y':<10} {results['mae_y_train']:>7.1f}px {results['mae_y_test']:>7.1f}px "
          f"{chance_results['mae_y_test']:>7.1f}px")
    print(f"  {'euclidean':<10} {results['euclidean_train']:>7.1f}px {results['euclidean_test']:>7.1f}px "
          f"{chance_results['euclidean_test']:>7.1f}px")

    # R²
    print(f"\n  -- R² (test) --")
    print(f"  ball_x: {results['r2_x']:.4f}    ball_y: {results['r2_y']:.4f}")
    print(f"  R²=0 = no better than predicting mean. R²=1 = perfect.")

    # Interpretation
    mae = results['euclidean_test']
    print(f"\n  -- INTERPRETATION --")
    if mae < 5:
        print(f"  Euclidean MAE = {mae:.1f}px -> CNN ENCODES ball position PRECISELY.")
        print(f"  This is a PURE policy-optimization problem: the CNN sees the ball,")
        print(f"  the value/policy heads just don't use it. (Perception-policy gap CONFIRMED.)")
    elif mae < 12:
        print(f"  Euclidean MAE = {mae:.1f}px -> CNN ENCODES ball position PARTIALLY.")
        print(f"  Some ball information survives in the features, but it's degraded.")
        print(f"  The policy head has a weaker signal to work with.")
    elif mae < 25:
        print(f"  Euclidean MAE = {mae:.1f}px -> CNN weakly encodes ball position.")
        print(f"  Significant feature degradation during training.")
    else:
        print(f"  Euclidean MAE = {mae:.1f}px -> CNN DOES NOT encode ball position.")
        print(f"  Features are indistinguishable from random. Full perception collapse.")
        print(f"  -> Need to protect representations (aux supervision, layer freezing).")


# -- Main ----------------------------------------------------------------

if __name__ == "__main__":
    MODEL_PATH = "./models/PPO_107/best_model.zip"
    RUN_NAME = "PPO_107"
    N_FRAMES = 3000
    CHECKPOINTS_DIR = None

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == '--model': MODEL_PATH = args[i + 1]; i += 2
        elif args[i] == '--run-name': RUN_NAME = args[i + 1]; i += 2
        elif args[i] == '--frames': N_FRAMES = int(args[i + 1]); i += 2
        elif args[i] == '--checkpoints': CHECKPOINTS_DIR = args[i + 1]; i += 2
        else: i += 1

    if RUN_NAME == "PPO_107" and MODEL_PATH != "./models/PPO_107/best_model.zip":
        import re
        m = re.search(r'PPO_\d+[a-z]?', MODEL_PATH)
        if m:
            RUN_NAME = m.group(0)

    # Determine models to test
    models_to_test = []
    if CHECKPOINTS_DIR and os.path.isdir(CHECKPOINTS_DIR):
        # Find all checkpoints
        ckpts = glob.glob(os.path.join(CHECKPOINTS_DIR, "latest_checkpoint_*_steps.zip"))
        ckpts = sorted(ckpts, key=os.path.getmtime)
        run_name = os.path.basename(os.path.dirname(CHECKPOINTS_DIR))
        for ckpt in ckpts:
            # Extract step count from filename
            m = re.search(r'latest_checkpoint_(\d+)_steps\.zip', os.path.basename(ckpt))
            step = int(m.group(1)) if m else 0
            models_to_test.append((run_name, ckpt, step))
        print(f"Found {len(models_to_test)} checkpoints in {CHECKPOINTS_DIR}")
    else:
        models_to_test.append((RUN_NAME, MODEL_PATH, None))

    print("=" * 70)
    print("Feature-Attribution Probe -- Does the CNN Encode Ball Position?")
    print("=" * 70)
    print(f"Target frames per model: {N_FRAMES:,}")
    print(f"Supervised baseline (perception POC): {SUPERVISED_MAE_BASELINE}px MAE")
    print()

    # Collect frames once using center-hold script (shared across checkpoints)
    print("--- Collecting frames with center-hold script ---")
    env_collect = make_vec_env()
    t0 = time.time()
    frames, labels = collect_frames_with_labels(env_collect, model=None,
                                                  n_frames=N_FRAMES, collect_mode="center_hold")
    env_collect.close()
    print(f"  Collected {len(frames):,} frames in {time.time() - t0:.0f}s")
    print(f"  Ball x range: [{min(l[0] for l in labels)}, {max(l[0] for l in labels)}]")
    print(f"  Ball y range: [{min(l[1] for l in labels)}, {max(l[1] for l in labels)}]")

    # Random-feature baseline (chance level)
    print(f"\n--- Random-feature baseline (chance-level MAE) ---")
    chance_results = random_feature_baseline(labels, n_trials=5)
    print(f"  Chance Euclidean MAE: {chance_results['euclidean_test']:.1f}px "
          f"(x: {chance_results['mae_x_test']:.1f}, y: {chance_results['mae_y_test']:.1f})")

    # Test each model/checkpoint
    all_results = []
    for run_name, model_path, step in models_to_test:
        print(f"\n--- {run_name}" + (f" @ {step:,} steps ---" if step else " ---"))

        env = make_vec_env()
        model = PPO.load(model_path, env=env, device="cuda")
        actual_step = model.num_timesteps
        env.close()

        t0 = time.time()
        features = extract_features(model, frames, device="cuda")
        print(f"  Feature extraction: {time.time() - t0:.0f}s "
              f"(shape: {features.shape})")

        t0 = time.time()
        results = train_and_evaluate(features, labels)
        print(f"  Regression: {time.time() - t0:.1f}s")

        print_results(run_name, actual_step, results, chance_results)
        all_results.append((run_name, actual_step, results))

    # Multi-checkpoint summary if applicable
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print(f"  CHECKPOINT TRAJECTORY -- Ball-Position Encoding Over Training")
        print(f"{'='*70}")
        print(f"  {'Steps':>12}  {'MAE x':>8}  {'MAE y':>8}  {'Euclid':>8}  {'R² x':>8}  {'R² y':>8}")
        print(f"  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
        for name, step, res in all_results:
            print(f"  {step:>12,}  {res['mae_x_test']:>7.1f}px {res['mae_y_test']:>7.1f}px "
                  f"{res['euclidean_test']:>7.1f}px {res['r2_x']:>7.3f}  {res['r2_y']:>7.3f}")

        # Trend detection
        maes = [r['euclidean_test'] for _, _, r in all_results]
        if len(maes) >= 3:
            trend = np.polyfit(range(len(maes)), maes, 1)[0]
            if trend > 0.5:
                print(f"\n  TREND: MAE INCREASING (+{trend:.2f}px/checkpoint)")
                print(f"  CNN is UNLEARNING ball position over training.")
            elif trend < -0.5:
                print(f"\n  TREND: MAE DECREASING ({trend:.2f}px/checkpoint)")
                print(f"  CNN is IMPROVING ball encoding over training.")
            else:
                print(f"\n  TREND: MAE STABLE -- ball encoding is maintained throughout training.")
