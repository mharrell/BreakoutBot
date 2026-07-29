"""
Find the RAM address that controls ball speed in Breakout.

The ball starts slow (moves every N frames) and speeds up as bricks
are cleared. There should be a timer/counter in RAM that controls this.
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

# Track ball movement frame by frame — look for frames where ball DOESN'T move
print("=== Ball movement per frame (fs=1, NOOP) ===")
print(f"{'Frame':>5} {'BallX':>6} {'BallY':>6} {'dX':>4} {'dY':>4}")
positions = []
all_rams = []
for i in range(120):
    ram = np.array(env.unwrapped.ale.getRAM(), dtype=int)
    bx = int(ram[BALL_X])
    by = int(ram[BALL_Y])
    positions.append((bx, by))
    all_rams.append(ram)
    obs, reward, terminated, truncated, info = env.step(NOOP)

for i in range(len(positions) - 1):
    bx, by = positions[i]
    bx2, by2 = positions[i+1]
    dx, dy = bx2 - bx, by2 - by
    marker = " <-- STALL" if (dx == 0 and dy == 0) else ""
    if marker or i < 30:
        print(f"{i:5d} {bx:6d} {by:6d} {dx:4d} {dy:4d}{marker}")

# Find addresses that are CONSTANT when ball moves, DIFFERENT when ball stalls
print("\n=== Looking for speed-control RAM addresses ===")
# Frames where ball stalled (dx=0, dy=0)
stall_frames = []
move_frames = []
for i in range(len(positions) - 1):
    dx = positions[i+1][0] - positions[i][0]
    dy = positions[i+1][1] - positions[i][1]
    if dx == 0 and dy == 0:
        stall_frames.append(i)
    else:
        move_frames.append(i)

print(f"Stall frames: {len(stall_frames)}, Move frames: {len(move_frames)}")

if stall_frames:
    # For each RAM address, check if value differs between stall and move frames
    candidates = []
    for addr in range(128):
        stall_vals = set(int(all_rams[i][addr]) for i in stall_frames)
        move_vals = set(int(all_rams[i][addr]) for i in move_frames[:len(stall_frames)])
        # Address that is DIFFERENT during stalls vs moves
        if stall_vals != move_vals and len(stall_vals | move_vals) <= 4:
            candidates.append((addr, stall_vals, move_vals))

    print(f"\nAddresses that differ between stall and move frames:")
    for addr, sv, mv in candidates[:20]:
        print(f"  RAM[{addr:3d}]: stall={sv}, move={mv}")

# Also check: addresses that correlate with overall game speed
# In Breakout, speed increases as more bricks are cleared
# RAM[??] might be a speed counter or level counter
print("\n=== All non-constant low-variance RAM addresses ===")
for addr in range(128):
    vals = set(int(r[addr]) for r in all_rams)
    if 2 <= len(vals) <= 4:
        print(f"  RAM[{addr:3d}]: values={sorted(vals)}")

env.close()
