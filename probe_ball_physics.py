"""
Find ball velocity/physics RAM addresses in Breakout.

Instead of teleporting ball_x via setRAM, we want to modify the ball's
horizontal velocity so it curves naturally away from the paddle.

This probes RAM to find:
1. Ball horizontal velocity/delta
2. Ball vertical velocity/delta
3. Any other physics-related addresses
"""
import numpy as np
import gymnasium as gym
from stable_baselines3.common.atari_wrappers import FireResetEnv
import ale_py
gym.register_envs(ale_py)

BALL_X = 99
BALL_Y = 101
PADDLE_X = 72

NOOP, FIRE, RIGHT, LEFT = 0, 1, 2, 3


def probe():
    env = gym.make("ALE/Breakout-v5", frameskip=1, repeat_action_probability=0)
    env = FireResetEnv(env)

    # Fire to launch ball, then track positions
    env.reset()

    # Fire to launch
    for _ in range(20):
        env.step(FIRE)

    # Now track ball_x frame-by-frame to compute velocity
    positions = []
    rams = []
    for i in range(60):
        ram = env.unwrapped.ale.getRAM()
        bx = int(ram[BALL_X])
        by = int(ram[BALL_Y])
        px = int(ram[PADDLE_X])
        positions.append((bx, by, px))
        rams.append(np.array(ram, dtype=int))
        obs, reward, terminated, truncated, info = env.step(NOOP)

    env.close()

    # Compute ball velocity from position changes
    print("=== Ball position tracking (60 frames, fs=1, NOOP) ===")
    print(f"{'Frame':>5} {'BallX':>6} {'BallY':>6} {'dX':>5} {'dY':>5} {'PdlX':>6}")
    for i in range(len(positions) - 1):
        bx, by, px = positions[i]
        bx2, by2, px2 = positions[i + 1]
        dx = bx2 - bx
        dy = by2 - by
        marker = " <--" if abs(dx) > 1 else ""
        print(f"{i:5d} {bx:6d} {by:6d} {dx:5d} {dy:5d} {px:6d}{marker}")

    # Find RAM addresses that correlate with ball velocity
    print("\n=== RAM correlation with ball delta_x ===")
    deltas_x = []
    for i in range(len(positions) - 1):
        deltas_x.append(positions[i+1][0] - positions[i][0])

    # Check each RAM address for correlation with dx
    all_rams = np.array(rams)
    matches = []
    for addr in range(128):
        ram_vals = all_rams[:-1, addr]  # values before movement
        # Check if this address predicts the next frame's delta_x
        if len(set(ram_vals)) > 1:  # skip constant addresses
            corr = np.corrcoef(ram_vals.astype(float), deltas_x)[0, 1]
            if not np.isnan(corr) and abs(corr) > 0.3:
                matches.append((addr, abs(corr), corr, list(set(ram_vals))[:5]))

    matches.sort(key=lambda x: -x[1])
    for addr, abs_corr, corr, sample_vals in matches[:10]:
        print(f"  RAM[{addr:3d}]: |r|={abs_corr:.3f} (r={corr:+.3f}), samples={sample_vals}")

    # Print all RAM values that changed in the last frame vs first frame
    print("\n=== RAM addresses that changed during tracking ===")
    first = all_rams[0]
    last = all_rams[-1]
    changed = []
    for addr in range(128):
        if first[addr] != last[addr]:
            changed.append((addr, first[addr], last[addr]))
    for addr, fv, lv in changed[:30]:
        print(f"  RAM[{addr:3d}]: {fv:3d} -> {lv:3d}")


if __name__ == "__main__":
    probe()
