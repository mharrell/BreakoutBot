"""
Proper ball speed probe — clear bricks, watch for acceleration.
"""
import numpy as np
import gymnasium as gym
from stable_baselines3.common.atari_wrappers import FireResetEnv
import ale_py
gym.register_envs(ale_py)

BALL_X, BALL_Y, PADDLE_X = 99, 101, 72
NOOP, FIRE, RIGHT, LEFT = 0, 1, 2, 3

env = gym.make("ALE/Breakout-v5", frameskip=1, repeat_action_probability=0)
env = FireResetEnv(env)
env.reset()

# Fire to launch
for _ in range(20):
    env.step(FIRE)

# Play with perfect tracking for many steps, note speed changes
print("=== Ball speed over extended play (perfect tracking, fs=1) ===")
print(f"{'Frame':>6} {'BallX':>6} {'BallY':>6} {'dX':>4} {'dY':>4} {'Speed':>5}")

positions = []
all_rams = []
prev_bx = prev_by = None
prev_ram = None

for i in range(3000):
    ram = np.array(env.unwrapped.ale.getRAM(), dtype=int)
    bx = int(ram[BALL_X])
    by = int(ram[BALL_Y])
    px = int(ram[PADDLE_X])

    # Perfect tracking
    if by > 180:
        action = FIRE
    elif px < bx:
        action = RIGHT
    elif px > bx:
        action = LEFT
    else:
        action = NOOP

    obs, reward, terminated, truncated, info = env.step(action)

    if prev_bx is not None:
        dx = bx - prev_bx
        dy = by - prev_by
        speed = abs(dx) + abs(dy)
        if speed > 1:  # Only log speed changes
            positions.append((i, bx, by, dx, dy, speed))
            all_rams.append((i, ram.copy(), prev_ram.copy() if prev_ram is not None else None))

    prev_bx, prev_by = bx, by
    prev_ram = ram.copy()

    if terminated or truncated:
        break

# Show speed transitions
print(f"\nSpeed changes detected ({len(positions)} non-1px movements):")
for i, bx, by, dx, dy, speed in positions[:30]:
    print(f"  frame {i:5d}: ({bx:3d},{by:3d}) dX={dx:3d} dY={dy:3d} speed={speed}")

# Look for RAM changes at speed transitions
if len(all_rams) >= 2:
    print("\n=== RAM changes at speed transitions ===")
    for i, ram_now, ram_before in all_rams[:5]:
        if ram_before is not None:
            changed = []
            for addr in range(128):
                if ram_now[addr] != ram_before[addr]:
                    changed.append((addr, int(ram_before[addr]), int(ram_now[addr])))
            if changed:
                print(f"  frame {i}: {changed[:10]}")

# Show all RAM values at a speed-2+ frame
if all_rams:
    print(f"\n=== Full RAM at first speed transition (frame {all_rams[0][0]}) ===")
    ram = all_rams[0][1]
    for row in range(0, 128, 16):
        vals = ' '.join(f'{int(ram[a]):3d}' for a in range(row, min(row+16, 128)))
        print(f"  {row:3d}-{min(row+15,127):3d}: {vals}")

env.close()
