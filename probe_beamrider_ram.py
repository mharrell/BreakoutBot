"""
Quick probe: verify BeamRider RAM tracks player ship X position.

OCAtari says: x_screen = int(ram[41]*1.5) - 115, ship y fixed at ~167.
We verify by pressing LEFT/RIGHT and watching RAM bytes change.
Uses standard wrappers (NoopResetEnv, FireResetEnv) matching training pipeline.
"""
import gymnasium as gym
import numpy as np
from stable_baselines3.common.atari_wrappers import FireResetEnv
import ale_py
gym.register_envs(ale_py)


def main():
    env = gym.make("ALE/BeamRider-v5", frameskip=4, repeat_action_probability=0)
    env = FireResetEnv(env)                # handles FIRE-to-start
    # No NoopResetEnv — noop_max=0 crashes (range 1..1 invalid), and we want
    # deterministic start for controlled test anyway

    obs, info = env.reset()
    print("After reset (FireResetEnv handled launch). Now pressing LEFT then RIGHT.")
    print("Frame | RAM[41] | RAM[16] | RAM[5] | Action")
    print("-" * 55)

    for frame in range(120):
        # LEFT for 60 frames, RIGHT for 60 frames
        if frame < 60:
            action = 3  # LEFT
        else:
            action = 2  # RIGHT

        obs, reward, terminated, truncated, info = env.step(action)
        ram = env.unwrapped.ale.getRAM()
        action_name = {2: 'RIGHT', 3: 'LEFT'}.get(action, str(action))
        print(f"  {frame:4d} | {int(ram[41]):7d} | {int(ram[16]):7d} | {int(ram[5]):5d} | {action_name}")

        if terminated or truncated:
            print("  -> Episode ended (life loss or game over)")
            obs, info = env.reset()

    env.close()

    print()
    print("If RAM[41] changes with LEFT/RIGHT, it tracks ship X position.")
    print("RAM[5] = lives, RAM[16] = game status (2=fighting)")

    # Quick scan: which RAM bytes change during LEFT vs RIGHT?
    print()
    print("Scanning for RAM bytes that respond to LEFT vs RIGHT...")
    env = gym.make("ALE/BeamRider-v5", frameskip=4, repeat_action_probability=0)
    env = FireResetEnv(env)
    obs, info = env.reset()

    # Wait for game to be in fighting state, take snapshot
    for _ in range(20):
        obs, _, _, _, _ = env.step(0)  # NOOP until game starts

    ram_before = env.unwrapped.ale.getRAM().copy()

    # Press LEFT 20 times
    for _ in range(20):
        obs, _, terminated, truncated, _ = env.step(3)
        if terminated or truncated:
            obs, _ = env.reset()

    ram_left = env.unwrapped.ale.getRAM().copy()

    # Press RIGHT 20 times
    for _ in range(20):
        obs, _, terminated, truncated, _ = env.step(2)
        if terminated or truncated:
            obs, _ = env.reset()

    ram_right = env.unwrapped.ale.getRAM().copy()

    changed = []
    for addr in range(128):
        if ram_before[addr] != ram_left[addr] or ram_left[addr] != ram_right[addr]:
            changed.append((addr, int(ram_before[addr]), int(ram_left[addr]), int(ram_right[addr])))

    print(f"  {len(changed)} RAM bytes changed:")
    for addr, before, left, right in changed:
        print(f"    RAM[{addr:3d}]: before={before:3d}  LEFT={left:3d}  RIGHT={right:3d}")

    env.close()


if __name__ == "__main__":
    main()
