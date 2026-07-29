"""
Test: flip ball horizontal direction instead of teleporting position.

If the ball's velocity is hardcoded (±1px/frame), we can't change magnitude.
But we CAN flip direction via RAM[105]. The ball curves naturally because
normal physics handles the movement — we just change the heading.

This tests whether direction flipping creates the "dodge" look without
the teleport/zig-zag artifacts.
"""
import numpy as np
import gymnasium as gym
from stable_baselines3.common.atari_wrappers import FireResetEnv
import ale_py
gym.register_envs(ale_py)

BALL_X = 99
BALL_Y = 101
PADDLE_X = 72
BALL_DIR = 105  # 1=left, 255=right (unsigned)

NOOP, FIRE, RIGHT, LEFT = 0, 1, 2, 3


def perfect_track(bx, px, by, frame):
    if by > 180:
        return FIRE
    if px < bx:
        return RIGHT
    elif px > bx:
        return LEFT
    return NOOP


def sweep(bx, px, by, frame, period=40):
    if by > 180:
        return FIRE
    return RIGHT if (frame % period) < (period // 2) else LEFT


print("=== Direction control test: push ball AWAY from paddle ===")
print("Mode: flip (toggle dir) vs dodge (set dir away from paddle)")
print()

for strategy_name, strategy_fn in [("perfect_track", perfect_track), ("sweep_p40", sweep)]:
    for mode in ["off", "flip", "dodge"]:
        env = gym.make("ALE/Breakout-v5", frameskip=4, repeat_action_probability=0)
        env = FireResetEnv(env)

        scores = []
        interventions_per_game = []

        for game in range(5):
            obs, info = env.reset()
            score = 0.0
            interventions = 0

            for frame in range(5000):
                ram = env.unwrapped.ale.getRAM()
                bx, by, px = int(ram[BALL_X]), int(ram[BALL_Y]), int(ram[PADDLE_X])

                action = strategy_fn(bx, px, by, frame)
                obs, reward, terminated, truncated, info = env.step(action)
                score += reward

                if mode != "off":
                    new_ram = env.unwrapped.ale.getRAM()
                    new_bx = int(new_ram[BALL_X])
                    new_by = int(new_ram[BALL_Y])
                    heading_down = new_by > by

                    if heading_down and new_by > 140:
                        error = new_bx - px
                        if abs(error) > 4:
                            new_dir = int(new_ram[BALL_DIR])
                            if mode == "flip":
                                # Toggle: 1↔255
                                env.unwrapped.ale.setRAM(BALL_DIR, 1 if new_dir == 255 else 255)
                            else:  # dodge
                                # Force AWAY from paddle: error>0 → ball right(255), error<0 → ball left(1)
                                env.unwrapped.ale.setRAM(BALL_DIR, 255 if error > 0 else 1)
                            interventions += 1

                if terminated or truncated:
                    break

            scores.append(score)
            interventions_per_game.append(interventions)

        label = f"{strategy_name} ({mode})"
        print(f"  {label:<35} scores={scores}  mean={np.mean(scores):.1f}  intv/game={np.mean(interventions_per_game):.1f}")
        env.close()

print()
print("Goal: 'dodge' kills scripts while perfect tracking survives (low interventions, high score).")
