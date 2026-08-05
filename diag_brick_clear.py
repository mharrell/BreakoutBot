"""
Diagnostic: test whether setRAM() actually clears Breakout bricks.

The BrickClearWrapper sets RAM addresses 0-35 to 0 to clear bricks,
but the split-watcher shows identical scores across all layouts,
suggesting the bricks aren't actually being cleared.

This script tests: write to RAM, step, read back — did it stick?
"""
import numpy as np
import gymnasium as gym
from stable_baselines3.common.atari_wrappers import FireResetEnv
import ale_py
gym.register_envs(ale_py)

BRICK_ADDRS = list(range(36))  # addresses 0-35

def make_env():
    env = gym.make("ALE/Breakout-v5", frameskip=4, repeat_action_probability=0)
    env = FireResetEnv(env)
    return env


def test_setram_persistence(env, addrs, n_steps=5):
    """Set RAM, then step n times and check if values persist."""
    env.reset()
    ram_before = env.unwrapped.ale.getRAM()

    # Clear bricks
    for addr in addrs:
        env.unwrapped.ale.setRAM(addr, 0)

    results = []
    for i in range(n_steps):
        obs, reward, terminated, truncated, info = env.step(0)  # NOOP
        ram_after = env.unwrapped.ale.getRAM()
        cleared_count = sum(1 for addr in addrs if ram_after[addr] == 0)
        results.append((i + 1, cleared_count, len(addrs)))
        if terminated or truncated:
            break

    return ram_before, results


def test_setram_before_reset(env, addrs):
    """Set RAM BEFORE reset — does ALE overwrite?"""
    # Try setting RAM before the first reset
    for addr in addrs:
        env.unwrapped.ale.setRAM(addr, 0)

    obs, info = env.reset()
    ram_after = env.unwrapped.ale.getRAM()
    cleared = sum(1 for addr in addrs if ram_after[addr] == 0)
    return cleared, len(addrs)


def test_multiframe_step(env, addrs):
    """After clearing, step with FIRE to launch ball, then check."""
    obs, info = env.reset()

    for addr in addrs:
        env.unwrapped.ale.setRAM(addr, 0)

    # Step with FIRE to launch
    obs, reward, terminated, truncated, info = env.step(1)  # FIRE

    ram = env.unwrapped.ale.getRAM()
    cleared = sum(1 for addr in addrs if ram[addr] == 0)

    # Check a few more steps
    for i in range(3):
        obs, reward, terminated, truncated, info = env.step(0)
        ram = env.unwrapped.ale.getRAM()
        cleared_now = sum(1 for addr in addrs if ram[addr] == 0)
        if cleared_now != cleared:
            print(f"  Step {i+2}: cleared count changed from {cleared} to {cleared_now}")

    return cleared, len(addrs)


if __name__ == "__main__":
    print("=== BrickClearWrapper Diagnostic ===\n")

    # Read raw brick values
    env = make_env()
    env.reset()
    ram = env.unwrapped.ale.getRAM()
    brick_vals = [ram[addr] for addr in BRICK_ADDRS]
    unique_vals = sorted(set(brick_vals))
    nonzero = [(addr, ram[addr]) for addr in BRICK_ADDRS if ram[addr] != 0]

    print(f"Brick RAM addresses (0-35) after reset:")
    print(f"  Values: {brick_vals}")
    print(f"  Unique: {unique_vals}")
    print(f"  Nonzero addresses: {nonzero}")
    print()

    # Test 1: setRAM persistence across NOOP steps
    print("--- Test 1: setRAM persistence across NOOP steps ---")
    env2 = make_env()
    ram_before, results = test_setram_persistence(env2, BRICK_ADDRS)
    print(f"  Before: {sum(1 for a in BRICK_ADDRS if ram_before[a] != 0)} nonzero bricks")
    for step, cleared, total in results:
        print(f"  After {step} NOOP(s): {cleared}/{total} cleared")
    print()

    # Test 2: setRAM before reset
    print("--- Test 2: setRAM BEFORE reset ---")
    env3 = make_env()
    cleared, total = test_setram_before_reset(env3, BRICK_ADDRS)
    print(f"  After reset: {cleared}/{total} cleared")
    print()

    # Test 3: with FIRE step
    print("--- Test 3: Clear + FIRE step ---")
    env4 = make_env()
    cleared, total = test_multiframe_step(env4, BRICK_ADDRS)
    print(f"  After FIRE: {cleared}/{total} cleared")
    print()

    # Test 4: write different pattern and verify visually
    print("--- Test 4: Specific addresses ---")
    env5 = make_env()
    env5.reset()
    test_addrs = [0, 1, 2, 18, 19, 20]  # mix of top rows
    print(f"  Setting addresses {test_addrs} to 0...")
    for addr in test_addrs:
        env5.unwrapped.ale.setRAM(addr, 0)
    obs, _, _, _, _ = env5.step(0)
    ram_after = env5.unwrapped.ale.getRAM()
    for addr in test_addrs:
        print(f"  addr {addr}: {ram_after[addr]}")

    # Check screen pixels — are bricks actually gone?
    # The top-left area of the screen should show cleared bricks
    print()
    print(f"  Screen shape: {obs.shape}")
    print(f"  Top-left 5x5 pixel block mean: {obs[50:55, 20:25].mean():.1f}")

    env5.close()
    env2.close()
    env3.close()
    env4.close()
    env.close()

    print("\nDone.")
