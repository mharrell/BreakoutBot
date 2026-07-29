"""
Calibrate AdversarialBallWrapper params — test paddle strategies.

Runs scripted paddle strategies against the adversarial wrapper to find
parameter settings where:
  - Perfect tracking → high score (env is learnable)
  - Sweep/static scripts → low score (scripts aren't viable)
  - Gap between them is large enough for PPO to discover

Usage:
    python calibrate_adv_wrapper.py                        # default params
    python calibrate_adv_wrapper.py --max-push 3 --gain 0.3
    python calibrate_adv_wrapper.py --max-push 2,3,4,5    # sweep max_push
    python calibrate_adv_wrapper.py --games 50             # more games per test
"""
import sys
import time
import numpy as np
import gymnasium as gym
from stable_baselines3.common.atari_wrappers import FireResetEnv
import ale_py
gym.register_envs(ale_py)

from adversarial_ball_wrapper import AdversarialBallWrapper

# ALE Breakout paddle actions
NOOP  = 0
FIRE  = 1
RIGHT = 2
LEFT  = 3

BALL_X_ADDR = 99
BALL_Y_ADDR = 101
PADDLE_X_ADDR = 72


class Strategy:
    """Base class for paddle strategies."""
    def act(self, ball_x, paddle_x, frame):
        raise NotImplementedError

    @property
    def label(self):
        raise NotImplementedError


class PerfectTracking(Strategy):
    """Paddle matches ball_x exactly — reactivity ceiling."""
    label = "perfect_track"

    def act(self, ball_x, paddle_x, frame):
        if ball_x is None:
            return FIRE
        if paddle_x < ball_x:
            return RIGHT
        elif paddle_x > ball_x:
            return LEFT
        return NOOP


class CenterHold(Strategy):
    """Paddle stays at center — simplest static script."""
    label = "center_hold"

    def act(self, ball_x, paddle_x, frame):
        target = 80
        if paddle_x < target:
            return RIGHT
        elif paddle_x > target:
            return LEFT
        return NOOP


class Sweep(Strategy):
    """Paddle sweeps back and forth — classic memorized script."""
    def __init__(self, period=40):
        self.period = period

    @property
    def label(self):
        return f"sweep_p{self.period}"

    def act(self, ball_x, paddle_x, frame):
        half = self.period // 2
        phase = frame % self.period
        if phase < half:
            return RIGHT
        else:
            return LEFT


class EdgeCamp(Strategy):
    """Paddle camps at one edge."""
    def __init__(self, side='left'):
        self.side = side

    @property
    def label(self):
        return f"edge_{self.side}"

    def act(self, ball_x, paddle_x, frame):
        if self.side == 'left':
            return LEFT if paddle_x > 20 else NOOP
        else:
            return RIGHT if paddle_x < 140 else NOOP


class RandomActs(Strategy):
    """Random actions — noise floor."""
    label = "random"

    def act(self, ball_x, paddle_x, frame):
        return np.random.choice([NOOP, RIGHT, LEFT], p=[0.3, 0.35, 0.35])


# All strategies to test
STRATEGIES = [
    PerfectTracking(),
    CenterHold(),
    Sweep(period=40),
    Sweep(period=80),
    EdgeCamp('left'),
    EdgeCamp('right'),
    RandomActs(),
]


def run_game(env, strategy, max_frames=5000):
    """Play one game with a scripted strategy, return score and push stats."""
    obs, info = env.reset()
    total_score = 0.0
    total_push = 0.0
    push_count = 0
    frames_since_fire = 0

    for frame in range(max_frames):
        # Read RAM for strategy
        ram = env.unwrapped.ale.getRAM()
        ball_x = int(ram[BALL_X_ADDR])
        ball_y = int(ram[BALL_Y_ADDR])
        paddle_x = int(ram[PADDLE_X_ADDR])

        action = strategy.act(ball_x, paddle_x, frame)

        # FIRE periodically if ball is in serve zone (high y)
        if ball_y > 180 and frames_since_fire > 10:
            action = FIRE
            frames_since_fire = 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_score += reward
        frames_since_fire += 1

        if info and 'adv_push' in info and abs(info['adv_push']) > 0.01:
            total_push += abs(info['adv_push'])
            push_count += 1

        if terminated or truncated:
            break

    return total_score, total_push, push_count, frame


def run_calibration(dead_zone=4.0, gain=0.5, max_push=4.0, zone_y=140,
                    n_games=20, frameskip=1):
    """Run all strategies against one wrapper config."""
    results = {}

    for strategy in STRATEGIES:
        scores = []
        push_mags = []
        push_rates = []
        durations = []

        for game in range(n_games):
            env = gym.make("ALE/Breakout-v5", frameskip=frameskip,
                           repeat_action_probability=0)
            env = FireResetEnv(env)
            env = AdversarialBallWrapper(env, dead_zone=dead_zone,
                                          proportional_gain=gain,
                                          paddle_zone_y=zone_y,
                                          max_push=max_push)
            score, total_push, push_count, frames = run_game(env, strategy)
            env.close()

            scores.append(score)
            if push_count > 0:
                push_mags.append(total_push / push_count)
            push_rates.append(push_count / max(frames, 1) * 100)
            durations.append(frames)

        results[strategy.label] = {
            'scores': np.array(scores),
            'mean_push': np.mean(push_mags) if push_mags else 0,
            'push_rate': np.mean(push_rates),
            'mean_frames': np.mean(durations),
        }

    return results


def print_results(results, params):
    """Print formatted results table."""
    print(f"\n{'='*80}")
    print(f"Params: dead_zone={params['dead_zone']}, gain={params['gain']}, "
          f"max_push={params['max_push']}, zone_y={params['zone_y']}, "
          f"fs={params['frameskip']}")
    print(f"{'='*80}")
    print(f"{'Strategy':<18} {'Score':>7} {'±':>5} {'Min':>5} {'Max':>5} "
          f"{'Push%':>6} {'PushMag':>7} {'Frames':>7}")
    print(f"{'-'*18} {'-'*7} {'-'*5} {'-'*5} {'-'*5} {'-'*6} {'-'*7} {'-'*7}")

    for strategy in STRATEGIES:
        r = results[strategy.label]
        s = r['scores']
        print(f"{strategy.label:<18} {s.mean():7.1f} {s.std():5.1f} "
              f"{s.min():5.0f} {s.max():5.0f} "
              f"{r['push_rate']:5.1f}% {r['mean_push']:6.1f}px "
              f"{r['mean_frames']:7.0f}")

    # Score gap: perfect tracking vs best script
    perfect = results['perfect_track']['scores'].mean()
    scripts = [r['scores'].mean() for label, r in results.items()
               if label not in ('perfect_track', 'random')]
    best_script = max(scripts)
    gap = perfect - best_script
    print(f"\n  Perfect tracking: {perfect:.1f}  |  Best script: {best_script:.1f}  "
          f"|  Gap: {gap:.1f}  |  Random: {results['random']['scores'].mean():.1f}")


if __name__ == "__main__":
    # Default params
    dead_zone = 4.0
    gain = 0.5
    max_pushes = [4.0]
    zone_y = 140
    n_games = 20
    frameskip = 1

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--max-push":
            # Can be comma-separated list for sweep
            val = args[i + 1]
            max_pushes = [float(v) for v in val.split(",")]
            i += 2
        elif args[i] == "--gain":
            gain = float(args[i + 1]); i += 2
        elif args[i] == "--dead":
            dead_zone = float(args[i + 1]); i += 2
        elif args[i] == "--zone-y":
            zone_y = int(args[i + 1]); i += 2
        elif args[i] == "--games":
            n_games = int(args[i + 1]); i += 2
        else:
            i += 1

    print(f"Calibrating AdversarialBallWrapper")
    print(f"  Games per strategy: {n_games}")
    print(f"  Strategies: {len(STRATEGIES)} ({', '.join(s.label for s in STRATEGIES)})")
    print(f"  Testing {len(max_pushes)} max_push values: {max_pushes}")
    print()

    for mp in max_pushes:
        t0 = time.time()
        params = {'dead_zone': dead_zone, 'gain': gain, 'max_push': mp,
                  'zone_y': zone_y, 'frameskip': frameskip}
        results = run_calibration(dead_zone=dead_zone, gain=gain, max_push=mp,
                                  zone_y=zone_y, n_games=n_games,
                                  frameskip=frameskip)
        elapsed = time.time() - t0
        print_results(results, params)
        print(f"  ({elapsed:.0f}s)")

    if len(max_pushes) > 1:
        print(f"\n{'='*80}")
        print("SUMMARY: Perfect tracking vs best script gap")
        print(f"{'='*80}")
        # (re-run is cheap, just summarizing — in a real sweep we'd cache)
