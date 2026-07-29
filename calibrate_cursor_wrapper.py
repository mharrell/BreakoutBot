"""
Calibrate AdversarialCursorWrapper params — test paddle strategies.

Runs scripted paddle strategies against the cursor wrapper to find parameter
settings where:
  - Perfect tracking → high score (env is learnable)
  - Sweep/static scripts → low score (scripts aren't viable)
  - Gap between them is large enough for PPO's reward gradient to favor tracking

Usage:
    python calibrate_cursor_wrapper.py
    python calibrate_cursor_wrapper.py --approach-speed 1,2,3
    python calibrate_cursor_wrapper.py --push 4 --warning 3
    python calibrate_cursor_wrapper.py --games 50
"""
import sys
import time
import numpy as np
import gymnasium as gym
from stable_baselines3.common.atari_wrappers import FireResetEnv
import ale_py
gym.register_envs(ale_py)

from adversarial_cursor_wrapper import AdversarialCursorWrapper

NOOP, FIRE, RIGHT, LEFT = 0, 1, 2, 3
BALL_X_ADDR = 99
BALL_Y_ADDR = 101
PADDLE_X_ADDR = 72


class Strategy:
    def act(self, ball_x, ball_y, paddle_x, frame):
        raise NotImplementedError
    @property
    def label(self):
        raise NotImplementedError


class PerfectTracking(Strategy):
    label = "perfect_track"
    def act(self, ball_x, ball_y, paddle_x, frame):
        if ball_y > 180:
            return FIRE
        if paddle_x < ball_x:
            return RIGHT
        elif paddle_x > ball_x:
            return LEFT
        return NOOP


class Sweep(Strategy):
    def __init__(self, period=40):
        self.period = period
    @property
    def label(self):
        return f"sweep_p{self.period}"
    def act(self, ball_x, ball_y, paddle_x, frame):
        if ball_y > 180:
            return FIRE
        return RIGHT if (frame % self.period) < (self.period // 2) else LEFT


class CenterHold(Strategy):
    label = "center_hold"
    def act(self, ball_x, ball_y, paddle_x, frame):
        target = 80
        if paddle_x < target: return RIGHT
        elif paddle_x > target: return LEFT
        return NOOP


class EdgeCamp(Strategy):
    def __init__(self, side='left'):
        self.side = side
    @property
    def label(self):
        return f"edge_{self.side}"
    def act(self, ball_x, ball_y, paddle_x, frame):
        if self.side == 'left':
            return LEFT if paddle_x > 20 else NOOP
        else:
            return RIGHT if paddle_x < 140 else NOOP


class RandomActs(Strategy):
    label = "random"
    def act(self, ball_x, ball_y, paddle_x, frame):
        return np.random.choice([NOOP, RIGHT, LEFT], p=[0.3, 0.35, 0.35])


STRATEGIES = [
    PerfectTracking(),
    Sweep(period=40),
    Sweep(period=80),
    CenterHold(),
    EdgeCamp('left'),
    EdgeCamp('right'),
    RandomActs(),
]


def run_game(env, strategy, max_frames=5000):
    obs, info = env.reset()
    total_score = 0.0
    attacks = 0
    total_push = 0.0

    for frame in range(max_frames):
        ram = env.unwrapped.ale.getRAM()
        bx, by, px = int(ram[BALL_X_ADDR]), int(ram[BALL_Y_ADDR]), int(ram[PADDLE_X_ADDR])

        action = strategy.act(bx, by, px, frame)
        obs, reward, terminated, truncated, info = env.step(action)
        total_score += reward

        if info and info.get('adv_push', 0) != 0:
            attacks += 1
            total_push += abs(info['adv_push'])

        if terminated or truncated:
            break

    return total_score, attacks, total_push, frame


def run_calibration(params, n_games=20, frameskip=4):
    """Run all strategies against one wrapper config."""
    results = {}
    for strategy in STRATEGIES:
        scores, attack_counts, push_totals, durations = [], [], [], []
        for _ in range(n_games):
            env = gym.make("ALE/Breakout-v5", frameskip=frameskip,
                           repeat_action_probability=0)
            env = FireResetEnv(env)
            env = AdversarialCursorWrapper(env, **params)
            score, attacks, push, frames = run_game(env, strategy)
            env.close()
            scores.append(score)
            attack_counts.append(attacks)
            push_totals.append(push)
            durations.append(frames)
        results[strategy.label] = {
            'scores': np.array(scores),
            'attacks': np.mean(attack_counts),
            'mean_push': np.mean(push_totals),
            'mean_frames': np.mean(durations),
        }
    return results


def print_results(results, params):
    print(f"\n{'='*80}")
    pstr = ', '.join(f'{k}={v}' for k, v in params.items())
    print(f"Params: {pstr}")
    print(f"{'='*80}")
    print(f"{'Strategy':<18} {'Score':>7} {'±':>5} {'Min':>5} {'Max':>5} "
          f"{'Attacks':>8} {'Push':>7} {'Frames':>7}")
    print(f"{'-'*18} {'-'*7} {'-'*5} {'-'*5} {'-'*5} {'-'*8} {'-'*7} {'-'*7}")

    for s in STRATEGIES:
        r = results[s.label]
        sc = r['scores']
        print(f"{s.label:<18} {sc.mean():7.1f} {sc.std():5.1f} "
              f"{sc.min():5.0f} {sc.max():5.0f} "
              f"{r['attacks']:7.1f} {r['mean_push']:6.1f}px "
              f"{r['mean_frames']:7.0f}")

    perfect = results['perfect_track']['scores'].mean()
    script_scores = [r['scores'].mean() for label, r in results.items()
                     if label not in ('perfect_track', 'random')]
    best_script = max(script_scores)
    gap = perfect - best_script
    print(f"\n  Perfect tracking: {perfect:.1f}  |  Best script: {best_script:.1f}  "
          f"|  Gap: {gap:.1f}")

    # Per-strategy attack rate
    print(f"  Attacks/game:")
    for s in STRATEGIES:
        r = results[s.label]
        print(f"    {s.label:<18} {r['attacks']:.1f}")


if __name__ == "__main__":
    # Default params
    defaults = dict(
        approach_speed=2.0,
        tracking_threshold=8,
        threat_radius=8,
        warning_frames=5,
        push_magnitude=4.0,
        cooldown_frames=60,
        cursor_size=4,
    )
    n_games = 20
    frameskip = 4

    # Parse CLI overrides
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == '--games':
            n_games = int(args[i + 1]); i += 2
        elif args[i] == '--fs':
            frameskip = int(args[i + 1]); i += 2
        elif args[i] in ('--approach-speed', '--tracking-threshold', '--threat-radius',
                         '--warning-frames', '--push', '--cooldown-frames', '--cursor-size'):
            key = args[i][2:].replace('-', '_')
            if key == 'push':
                key = 'push_magnitude'
            val = args[i + 1]
            # Support comma-separated sweeps
            if ',' in val:
                defaults[key] = [float(v) if '.' in v else int(v) for v in val.split(',')]
            else:
                defaults[key] = float(val) if '.' in val else int(val)
            i += 2
        else:
            i += 1

    # Check which params have list values (sweep params)
    sweep_keys = [k for k, v in defaults.items() if isinstance(v, list)]

    if not sweep_keys:
        # Single config
        t0 = time.time()
        results = run_calibration(defaults, n_games=n_games, frameskip=frameskip)
        print_results(results, defaults)
        print(f"\n  ({time.time() - t0:.0f}s)")
    else:
        # Sweep one param
        key = sweep_keys[0]
        values = defaults[key]
        print(f"Calibrating cursor wrapper — sweeping {key} over {values}")
        print(f"  Games per strategy: {n_games}, frameskip={frameskip}")
        print(f"  Fixed params: {', '.join(f'{k}={v}' for k, v in defaults.items() if k != key)}")

        for val in values:
            params = dict(defaults)
            params[key] = val
            t0 = time.time()
            results = run_calibration(params, n_games=n_games, frameskip=frameskip)
            print_results(results, params)
            print(f"  ({time.time() - t0:.0f}s)")

    print(f"\nGoal: perfect_track >> best script. Gap > 5 is workable, > 10 is strong.")
