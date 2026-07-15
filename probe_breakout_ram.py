"""
Breakout RAM probe — records all 128 Atari 2600 RAM bytes during structured
gameplay and identifies which addresses correspond to which game parameters.

Prerequisite for Experiment 5B (RAM-parameterized physics): we need to know
what's pokeable via ALE.setRAM() before designing parameter randomization.

Usage:
    python probe_breakout_ram.py          # record and analyze
    python probe_breakout_ram.py --test   # run unit tests only

Design decisions (2026-07-14):
  - Structured action sequences, not random — systematic sweeps make
    parameter identification unambiguous.
  - frameskip=1 (single-frame mode) — captures every RAM change.
  - Analysis functions operate on numpy arrays — testable without ALE.
"""

import numpy as np
import sys
import os

# ---------------------------------------------------------------------------
# Analysis functions (unit-testable — no ALE dependency)
# ---------------------------------------------------------------------------

def classify_byte(time_series: np.ndarray) -> dict:
    """Classify a single RAM byte's behavior from its frame-by-frame values.

    Args:
        time_series: 1D numpy array of byte values (uint8) over N frames.

    Returns dict with keys:
        address: int, address in RAM (0-127)
        min_val, max_val: range observed
        unique_count: number of distinct values
        changed: bool, did the value change at all during recording
        classification: str, one of: 'constant', 'smooth_position', 'counter',
                       'discrete_state', 'volatile'
        confidence: str, 'high', 'medium', 'low'
        notes: str, human-readable interpretation
    """
    result = {
        "min_val": int(np.min(time_series)),
        "max_val": int(np.max(time_series)),
        "unique_count": len(np.unique(time_series)),
        "changed": False,
        "classification": "constant",
        "confidence": "low",
        "notes": "",
    }
    if result["unique_count"] <= 1:
        result["notes"] = "never changed during recording"
        return result

    result["changed"] = True
    vals = time_series.astype(np.int32)
    diffs = np.diff(vals)
    nonzero_diffs = diffs[diffs != 0]

    # Smooth position: value changes by small amounts (±1 to ±4 per frame),
    # stays within a bounded range, covers many values.
    step_sizes = np.abs(nonzero_diffs)
    if (len(nonzero_diffs) > 20
            and np.median(step_sizes) <= 4
            and np.max(step_sizes) <= 8
            and result["unique_count"] > 20
            and result["max_val"] - result["min_val"] >= 30):
        result["classification"] = "smooth_position"
        result["confidence"] = "high"
        result["notes"] = "smooth position — likely paddle/ball coordinate"
        return result

    # Discrete state (few transitions): check BEFORE counter, since a step
    # function like lives (5→4→3→2) has all-decrements but only a handful of
    # transition points — not a continuously-decrementing counter.
    if result["unique_count"] <= 12 and len(nonzero_diffs) <= 10:
        result["classification"] = "discrete_state"
        result["confidence"] = "high" if result["unique_count"] <= 6 else "medium"
        result["notes"] = (f"step function: {result['unique_count']} states, "
                           f"{len(nonzero_diffs)} transitions — likely lives, mode, or game phase")
        return result

    # Counter: mostly decrements by 1 (or increments slowly), many events.
    decrements = np.sum(nonzero_diffs < 0)
    increments = np.sum(nonzero_diffs > 0)
    if (decrements > 0
            and decrements >= 3 * increments
            and result["unique_count"] <= 80
            and np.median(step_sizes) <= 2):
        result["classification"] = "counter"
        result["confidence"] = "medium"
        result["notes"] = "mostly decrements — likely brick count, lives, or timer"
        return result

    # Volatile: many unique values, large jumps, erratic pattern.
    result["classification"] = "volatile"
    result["confidence"] = "low"
    result["notes"] = (f"erratic pattern: {len(nonzero_diffs)} changes, "
                       f"steps range {int(np.min(step_sizes))}-{int(np.max(step_sizes))}")
    return result


def label_known_addresses(address: int, classification: dict) -> str:
    """Return a human label for known RAM addresses (validated in this project)."""
    known = {
        70: "Paddle X position (0-191)",
        72: "Ball X position (0-191)",
        90: "Ball Y position (increments downward)",
    }
    return known.get(address, "")


def analyze_all_ram(ram_log: np.ndarray) -> list:
    """Analyze a full RAM recording.

    Args:
        ram_log: 2D numpy array of shape (num_frames, 128), dtype uint8.
                 ram_log[t, a] = value of address a at frame t.

    Returns:
        List of dicts, one per address (0-127), sorted by likely interestingness.
    """
    results = []
    for addr in range(128):
        ts = ram_log[:, addr]
        result = classify_byte(ts)
        result["address"] = addr
        known_label = label_known_addresses(addr, result)
        if known_label:
            result["known_label"] = known_label
            if result["confidence"] != "high":
                result["confidence"] = "high"  # confirmed by prior work
        results.append(result)

    # Sort: changed first, then by classification priority
    priority = {
        "smooth_position": 0,
        "counter": 1,
        "discrete_state": 2,
        "volatile": 3,
        "constant": 4,
    }
    results.sort(key=lambda r: (0 if r["changed"] else 1,
                                 priority.get(r["classification"], 5),
                                 r["address"]))
    return results


# ---------------------------------------------------------------------------
# Recording (requires ALE)
# ---------------------------------------------------------------------------

def record_ram_frames(rom="ALE/Breakout-v5", max_frames=3000) -> np.ndarray:
    """Play structured sequences and record all 128 RAM bytes each frame.

    Structured sequence:
      1. Noop (serve ball)
      2. Alternate LEFT/RIGHT sweeps (identify paddle position)
      3. Idle — let ball bounce naturally (identify ball position)
      4. (Brick hits and score changes happen naturally)

    Args:
        rom: ALE environment ID.
        max_frames: Maximum frames to record (~3,000 = ~50 seconds at 60fps).

    Returns:
        2D numpy array of shape (num_frames, 128), dtype uint8.
    """
    import ale_py
    import gymnasium as gym
    gym.register_envs(ale_py)

    env = gym.make(rom, frameskip=1, repeat_action_probability=0.0)
    env.reset()
    ale = env.unwrapped.ale

    frames = []
    action_sequence = []

    # Phase 1: Serve the ball (~180 frames: NOOP until FIRE, then back to NOOP)
    action_sequence.extend([0] * 120)   # NOOP
    action_sequence.extend([1] * 10)    # FIRE to serve
    action_sequence.extend([0] * 50)    # NOOP — let ball travel

    # Phase 2: Paddle sweeps (alternate LEFT/RIGHT to hit full range)
    for _ in range(4):
        action_sequence.extend([3] * 60)   # LEFT
        action_sequence.extend([2] * 60)   # RIGHT

    # Phase 3: Idle — let ball bounce naturally, score, die
    action_sequence.extend([0] * 1500)

    # Phase 4: More sweeps (after some bricks cleared)
    for _ in range(3):
        action_sequence.extend([3] * 40)
        action_sequence.extend([2] * 40)

    # Phase 5: Idle until death / recording limit
    action_sequence.extend([0] * 1000)

    n_frames = min(len(action_sequence), max_frames)
    print(f"Recording {n_frames} frames of RAM data...")

    for i in range(n_frames):
        action = action_sequence[i]
        ale.act(action)
        ram = ale.getRAM()
        frames.append(np.array(ram, dtype=np.uint8))

    env.close()
    ram_log = np.stack(frames)
    print(f"Recorded: {ram_log.shape[0]} frames × {ram_log.shape[1]} bytes")
    return ram_log


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(results: list):
    """Print human-readable report of RAM analysis."""
    print()
    print("=" * 70)
    print("BREAKOUT RAM PROBE — ANALYSIS REPORT")
    print("=" * 70)

    changed = [r for r in results if r["changed"]]
    constant = [r for r in results if not r["changed"]]
    print(f"\nChanged bytes: {len(changed)} / 128")
    print(f"Constant bytes: {len(constant)} / 128")

    print("\n--- Likely Game Parameters ---")
    for r in results:
        if r["classification"] == "constant" and not r.get("known_label"):
            continue
        known = r.get("known_label", "")
        label = f"  [{known}]" if known else ""
        print(f"  ${r['address']:02x} ({r['address']:3d}): "
              f"class={r['classification']:20s} "
              f"range=[{r['min_val']:3d}, {r['max_val']:3d}] "
              f"unique={r['unique_count']:3d} "
              f"conf={r['confidence']:6s}"
              f"{label}")
        if r["notes"]:
            print(f"       {r['notes']}")

    print(f"\n--- All Changed Bytes ({len(changed)}) ---")
    for r in results:
        if not r["changed"]:
            continue
        known = r.get("known_label", "")
        label = f"  [{known}]" if known else ""
        print(f"  ${r['address']:02x} ({r['address']:3d}): "
              f"{r['classification']:20s} "
              f"[{r['min_val']:3d}-{r['max_val']:3d}] "
              f"n={r['unique_count']:3d}"
              f"{label}")

    print(f"\n--- Constant Bytes ({len(constant)}) ---")
    constant_addrs = [f"${r['address']:02x}" for r in constant]
    for i in range(0, len(constant_addrs), 16):
        print(f"  {' '.join(constant_addrs[i:i+16])}")


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def run_tests():
    """Test analysis functions with synthetic RAM data."""
    import traceback
    passed = 0
    failed = 0

    def check(name, condition, detail=""):
        nonlocal passed, failed
        if condition:
            passed += 1
        else:
            failed += 1
            print(f"  FAIL: {name} — {detail}")
            traceback.print_stack()

    print("Running RAM probe unit tests...")

    # Test 1: constant byte
    ts = np.full(1000, 42, dtype=np.uint8)
    r = classify_byte(ts)
    check("constant: changed=False", not r["changed"])
    check("constant: classification", r["classification"] == "constant")
    check("constant: unique=1", r["unique_count"] == 1)

    # Test 2: smooth position (simulated paddle sweep)
    ts = np.zeros(1000, dtype=np.uint8)
    pos = 95
    direction = 1
    for i in range(1000):
        pos += direction
        if pos >= 190:
            direction = -1
        if pos <= 5:
            direction = 1
        ts[i] = pos
    r = classify_byte(ts)
    check("position: changed=True", r["changed"])
    check("position: classification", r["classification"] == "smooth_position")
    check("position: confidence", r["confidence"] == "high")
    check("position: range", r["max_val"] - r["min_val"] >= 30)

    # Test 3: decrementing counter (simulated brick count)
    ts = np.full(1000, 110, dtype=np.uint8)
    bricks = 110
    for i in range(1000):
        if i % 50 == 0 and bricks > 0:
            bricks -= 1
        ts[i] = bricks
    r = classify_byte(ts)
    check("counter: changed=True", r["changed"])
    check("counter: classification", r["classification"] == "counter")
    check("counter: confidence in [medium,high]", r["confidence"] in ("medium", "high"))

    # Test 4: discrete state (simulated lives counter)
    ts = np.full(500, 5, dtype=np.uint8)
    ts[100:200] = 4
    ts[200:350] = 3
    ts[350:500] = 2
    r = classify_byte(ts)
    check("state: changed=True", r["changed"])
    check("state: classification", r["classification"] == "discrete_state")
    check("state: unique <= 5", r["unique_count"] <= 5)

    # Test 5: volatile (random-looking data)
    rng = np.random.default_rng(42)
    ts = rng.integers(0, 256, 500, dtype=np.uint8)
    r = classify_byte(ts)
    check("volatile: changed=True", r["changed"])
    check("volatile: unique many", r["unique_count"] > 50)

    # Test 6: known address labeling
    r = {"changed": True, "classification": "smooth_position", "confidence": "high",
         "min_val": 0, "max_val": 191, "unique_count": 100, "notes": ""}
    label = label_known_addresses(70, r)
    check("label: address 70 is paddle", "Paddle" in label)
    label = label_known_addresses(99, r)
    check("label: unknown address empty", label == "")

    # Test 7: analyze_all_ram returns 128 results
    ram_log = np.zeros((100, 128), dtype=np.uint8)
    ram_log[:, 70] = np.linspace(0, 191, 100).astype(np.uint8)  # paddle sweep
    results = analyze_all_ram(ram_log)
    check("analyze_all: 128 results", len(results) == 128)
    check("analyze_all: first result is changed",
          results[0]["changed"])  # address 70 should be first
    check("analyze_all: paddle labeled",
          results[0].get("known_label", "") != "")

    print(f"\n  {passed} passed, {failed} failed")
    return failed == 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if "--test" in sys.argv:
        ok = run_tests()
        sys.exit(0 if ok else 1)
    else:
        # Run tests first as a smoke check
        if not run_tests():
            print("\nTests failed — aborting recording.")
            sys.exit(1)

        ram_log = record_ram_frames(max_frames=3000)
        results = analyze_all_ram(ram_log)
        print_report(results)

        # Save raw data for offline analysis
        np.save("recordings/breakout_ram_probe.npy", ram_log)
        print(f"\nRaw RAM data saved to recordings/breakout_ram_probe.npy")
