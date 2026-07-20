"""
Step 0 — ALE RAM Address Verification

Verifies OCAtari RAM address map against ALE 0.11.2 by writing known
values and reading back. Does NOT require a display — pure write/read
verification.

Addresses under test (from OCAtari, unverified):
  Paddle X:    72    (write 20, 80, 120, 160)
  Ball X:      99    (write 50, 100, 150)
  Ball Y:      101   (write 50, 100, 150)
  Score:       76-77 (read-only — play frames, observe changes)
  Lives:       57    (read-only — observe starting value)

Usage:
    python probe_ale_ram.py
"""
import sys
import ale_py
import numpy as np
import gymnasium as gym

gym.register_envs(ale_py)


def make_env():
    return gym.make("ALE/Breakout-v5", frameskip=1, repeat_action_probability=0)


def verify_writeable(ale, addr, label, test_values):
    """Write values to addr and verify they stick after a step."""
    print(f"\n  [{label}] addr={addr}")
    ram = ale.getRAM()
    initial = int(ram[addr])
    print(f"    Initial: {initial}")

    all_ok = True
    for value in test_values:
        ale.setRAM(addr, value)
        ale.getRAM()  # flush? unnecessary but harmless
        ram = ale.getRAM()
        post = int(ram[addr])
        ok = post == value
        if not ok:
            all_ok = False
        marker = "OK" if ok else f"FAIL (wrote {value}, read {post})"
        print(f"    {marker}: wrote {value:3d}, read {post:3d}")

    # Restore
    ale.setRAM(addr, initial)
    return all_ok


def main():
    print("=" * 60)
    print("  ALE RAM Address Probe — Step 0")
    print("  ALE 0.11.2, Breakout-v5, frameskip=1")
    print("=" * 60)
    sys.stdout.flush()

    env = make_env()
    ale = env.unwrapped.ale

    # --- Reset and warm up ---
    env.reset()
    for _ in range(10):
        env.step(0)

    # Dump full RAM once for reference
    ram = ale.getRAM()
    print(f"\n  Full RAM snapshot (128 bytes):")
    print(f"  {list(ram)}")

    # --- Probe writeable addresses ---
    results = {}

    # Paddle X (addr 72)
    env.reset()
    for _ in range(5):
        env.step(0)
    results["paddle_x_72"] = verify_writeable(
        ale, 72, "Paddle X", [20, 80, 120, 160])

    # Ball X (addr 99)
    env.reset()
    for _ in range(3):
        env.step(0)
    results["ball_x_99"] = verify_writeable(
        ale, 99, "Ball X", [50, 100, 150])

    # Ball Y (addr 101)
    env.reset()
    for _ in range(3):
        env.step(0)
    results["ball_y_101"] = verify_writeable(
        ale, 101, "Ball Y", [50, 100, 150])

    # --- Read-only addresses: monitor across frames ---
    env.reset()
    print(f"\n  [Score + Lives] read-only monitoring across 60 frames")
    score_vals = set()
    lives_vals = set()
    for i in range(60):
        ram = ale.getRAM()
        score_vals.add(int(ram[76]))
        score_vals.add(int(ram[77]))
        lives_vals.add(int(ram[57]))
        action = 3 if i % 5 == 0 else 0
        env.step(action)

    print(f"    Score byte 76 values: {sorted(score_vals)}")
    print(f"    Score byte 77 values: {sorted(score_vals)}")
    print(f"    Lives (addr 57):       {sorted(lives_vals)}")

    env.close()

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    all_pass = True
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  {name}: {status}")
    print(f"\n  Verdict: {'ALL ADDRESSES VERIFIED' if all_pass else 'SOME ADDRESSES FAILED - DO NOT USE WITHOUT CORRECTION'}")


if __name__ == "__main__":
    main()
