"""
Probe: map brick RAM bits to screen positions.

Clears all bricks, sets one bit at a time, captures screen pixels,
and builds a (byte, bit) -> (x, y) mapping table.

Usage:
    python probe_brick_grid.py
"""
import numpy as np
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)


def get_screen(env):
    """Get full RGB screen array from ALE."""
    return env.unwrapped.ale.getScreenRGB()  # (210, 160, 3) uint8


def find_brick_pixels(screen, bg_screen):
    """Return list of (x, y) pixel coords that differ from background."""
    diff = np.abs(screen.astype(np.int16) - bg_screen.astype(np.int16))
    changed = diff.max(axis=2) > 30  # threshold for "different from bg"
    ys, xs = np.where(changed)
    return list(zip(xs, ys))


def main():
    env = gym.make("ALE/Breakout-v5", frameskip=1, repeat_action_probability=0,
                   render_mode="rgb_array")

    print("Probing brick RAM bit-to-position mapping...")
    print("=" * 60)

    # Fire to start game
    env.reset()
    bg = get_screen(env)  # black screen before fire

    for _ in range(10):
        env.step(1)  # FIRE
    env.step(0)  # NOOP to render

    # Capture background (no bricks)
    # First, clear ALL bricks
    ale = env.unwrapped.ale
    for addr in range(36):
        ale.setRAM(addr, 0)

    # Step to render cleared bricks
    for _ in range(5):
        env.step(0)
    cleared_screen = get_screen(env)

    # Now probe each byte, each bit
    mapping = []
    for byte_addr in range(36):
        for bit in range(8):
            # Set only this bit in this byte
            value = 1 << bit
            ale.setRAM(byte_addr, value)

            # Step to render
            for _ in range(3):
                env.step(0)
            screen = get_screen(env)

            # Find brick pixels
            bricks = find_brick_pixels(screen, cleared_screen)

            if bricks:
                # Compute bounding box
                xs = [x for x, y in bricks]
                ys = [y for x, y in bricks]
                cx = int(np.mean(xs))
                cy = int(np.mean(ys))
                w = max(xs) - min(xs) + 1
                h = max(ys) - min(ys) + 1

                # Classify row based on Y
                # Breakout bricks start around Y=30-40 and go to ~Y=140
                row = int((cy - 30) / 6) if cy > 30 else -1

                mapping.append({
                    "byte": byte_addr,
                    "bit": bit,
                    "value": value,
                    "cx": cx, "cy": cy,
                    "width": w, "height": h,
                    "side": "right" if byte_addr < 18 else "left",
                    "est_row": row,
                })

            # Clear the byte
            ale.setRAM(byte_addr, 0)

    # Print results sorted by Y (row)
    mapping.sort(key=lambda m: (m["cy"], m["cx"]))

    print(f"Found {len(mapping)} byte:bit -> brick position mappings\n")
    print(f"{'Byte':>4} {'Bit':>3} {'Side':>6} {'Row':>4} {'cx':>5} {'cy':>5} {'w':>3} {'h':>3}")
    print("-" * 50)
    for m in mapping:
        print(f"{m['byte']:4d} {m['bit']:3d} {m['side']:>6s} {m['est_row']:4d} "
              f"{m['cx']:5d} {m['cy']:5d} {m['width']:3d} {m['height']:3d}")

    # Build row-indexed lookup
    print(f"\n--- ROW LAYOUT (byte:bit -> x position) ---")
    rows = {}
    for m in mapping:
        side = m["side"]
        row = m["est_row"]
        key = (side, row)
        if key not in rows:
            rows[key] = []
        rows[key].append(m)

    for (side, row), bricks in sorted(rows.items()):
        bricks.sort(key=lambda m: m["cx"])
        bits_str = ", ".join(f"b{b['bit']}(x={b['cx']})" for b in bricks)
        print(f"  {side:>6s} row {row:2d}: {bits_str}")

    env.close()


if __name__ == "__main__":
    main()
