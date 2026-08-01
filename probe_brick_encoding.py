"""
Probe: brick state encoding in ALE Breakout RAM.

Breakout has 36 bricks visible on screen. According to OCAtari,
brick state is stored in RAM bytes 0-35. Each byte probably encodes
whether a brick is present, and if so, how many hits remaining.

This script:
  1. Starts a game with standard layout, reads brick RAM → baseline
  2. Resets with a custom layout (write specific bricks to RAM before first fire)
  3. Confirms the observation changes as expected

Run with: python probe_brick_encoding.py
"""
import time
import numpy as np
import gymnasium as gym
from stable_baselines3.common.atari_wrappers import FireResetEnv, MaxAndSkipEnv
import cv2
import ale_py
gym.register_envs(ale_py)

BALL_X, BALL_Y, PADDLE_X = 99, 101, 72
NOOP, FIRE, RIGHT, LEFT = 0, 1, 2, 3


def read_full_ram(env, label=""):
    """Print all 128 RAM bytes in structured format."""
    ram = np.array(env.unwrapped.ale.getRAM(), dtype=int)
    print(f"\n=== Full RAM dump {label} ===")
    for addr in range(0, 128, 16):
        vals = "  ".join(f"{ram[a]:3d}" for a in range(addr, min(addr+16, 128)))
        row = f"  [{addr:3d}-{min(addr+15,127):3d}] {vals}"
        print(row)
    # Brick bytes 0-35 specifically
    print(f"\n--- Brick bytes (0-35) {label} ---")
    brick_ram = ram[0:36]
    for row in range(6):
        start = row * 6
        vals = "  ".join(f"{b:4d}" for b in brick_ram[start:start+6])
        print(f"  row {row}: {vals}")
    non_zero = sum(1 for b in brick_ram if b > 0)
    print(f"  Non-zero bricks: {non_zero}")
    return ram


def write_brick_layout(env, layout):
    """Write a specific brick layout to RAM bytes 0-35.

    layout is a list of 36 bytes. Standard layout has each byte
    representing brick hits remaining (0 = gone, 1 = one hit left,
    2+ = two hits left for the orange/red rows).
    """
    for addr, val in enumerate(layout):
        env.unwrapped.ale.setRAM(addr, int(val))
    print(f"  Written custom layout to RAM[0-35]")


# --- Step 1: Standard game to see default brick RAM ---
print("=== Step 1: Read standard brick layout ===")
env = gym.make("ALE/Breakout-v5", frameskip=4, repeat_action_probability=0)
env = MaxAndSkipEnv(env, skip=4)  # needed for FireResetEnv
env = FireResetEnv(env)
env.reset()

# Wait for ball to be in play and bricks visible
print("Firing to launch ball...")
for _ in range(30):
    ram = env.unwrapped.ale.getRAM()
    by = ram[BALL_Y]
    if 30 <= by <= 180:  # ball in play
        break
    env.step(FIRE)

# Let a few bounces happen to see bricks getting hit
for _ in range(200):
    ram = env.unwrapped.ale.getRAM()
    bx, by, px = int(ram[BALL_X]), int(ram[BALL_Y]), int(ram[PADDLE_X])
    if by > 180:
        action = FIRE
    elif px < bx:
        action = RIGHT
    elif px > bx:
        action = LEFT
    else:
        action = NOOP
    env.step(action)

read_full_ram(env, "(standard layout, after a few bounces)")
env.close()

# --- Step 2: Custom layout ---
print("\n=== Step 2: Write custom brick layout ===")
env2 = gym.make("ALE/Breakout-v5", frameskip=4, repeat_action_probability=0)
env2 = MaxAndSkipEnv(env2, skip=4)
obs = env2.reset()

# Clear all bricks
all_cleared = [0] * 36
write_brick_layout(env2, all_cleared)

# Then fire and verify bricks are gone
env2.step(FIRE)
for _ in range(50):
    env2.step(NOOP)

read_full_ram(env2, "(all bricks cleared)")
env2.close()

# --- Step 3: Single-column layout ---
print("\n=== Step 3: Only left column ===")
env3 = gym.make("ALE/Breakout-v5", frameskip=4, repeat_action_probability=0)
env3 = MaxAndSkipEnv(env3, skip=4)
env3.reset()

# Only column 0 of each row — one-brick-tall pillars
# Standard Breakout brick map: 6 rows × 6 columns, 18 columns on screen?
# Let's probe — set every address in 0-35 that could be column 0
# Actually the screen has 18 brick columns across, not 6.
# The OCAtari note might be misleading about the mapping.

# Let's just clear everything and add back a few known-position bricks to see what they are
all_cleared = [0] * 36
write_brick_layout(env3, all_cleared)

# Write a single known brick pattern
TEST_BRICKS = [0] * 36
TEST_BRICKS[0] = 1   # first brick
TEST_BRICKS[5] = 2  # 6th brick
TEST_BRICKS[17] = 3  # 18th brick
TEST_BRICKS[35] = 1  # last brick
write_brick_layout(env3, TEST_BRICKS)

# Fire and observe
env3.step(FIRE)
for _ in range(5):
    env3.step(NOOP)
read_full_ram(env3, "(test pattern — 4 bricks set)")
env3.close()

# --- Step 4: Final definitive encoding check ---
print("\n=== Step 4: Encoding confirmation ===")
# Try to figure out how many bricks per byte
# The ALE Breakout screen shows approximately:
#   - 6 rows (yellow-green-orange-red-blue-magenta, top to bottom)
#   - Number of brick columns varies — need to check

env4 = gym.make("ALE/Breakout-v5", frameskip=1, repeat_action_probability=0)
env4.reset()
env4.step(FIRE)

# Let the game progress naturally for a couple bounces
for _ in range(500):
    ram = np.array(env4.unwrapped.ale.getRAM(), dtype=int)
    bx, by, px = int(ram[BALL_X]), int(ram[BALL_Y]), int(ram[PADDLE_X])
    if by > 180:
        action = FIRE
    elif px < bx:
        action = RIGHT
    elif px > bx:
        action = LEFT
    else:
        action = NOOP
    obs, reward, terminated, truncated, info = env4.step(action)
    if reward > 0:
        # A brick got hit — read state immediately
        post_ram = np.array(env4.unwrapped.ale.getRAM(), dtype=int)
        brick_bytes = post_ram[0:36]
        score_bytes = post_ram[76:78]  # 16-bit BCD score
        lives_byte = post_ram[57]
        print(f"Hit! reward={reward:.0f}  bricks non-zero={sum(1 for b in brick_bytes if b>0)}  "
              f"score={score_bytes}  lives={lives_byte}")
        brick_rows = "\n    ".join(
            "  ".join(f"{b:3d}" for b in brick_bytes[r*6:(r+1)*6])
            for r in range(6)
        )
        print(f"    brick bytes (0-35) rows:\n    {brick_rows}")
    if terminated or truncated:
        break

env4.close()

print("\n=== Done. Based on results we can interpret brick encoding and write custom layouts. ===")
