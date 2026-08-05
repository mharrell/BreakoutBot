"""
Visual ball-teleport split-watcher — watch the model track (or ignore) the teleported ball.

Left: FULL (ball at normal position)
Right: ALT (ball teleported +30px right)

Usage:
    python watch_ball_teleport.py --model ./models/PPO_131/final_model.zip
"""
import sys
import re
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import FireResetEnv, EpisodicLifeEnv
import cv2
import ale_py
gym.register_envs(ale_py)

BALL_X, BALL_Y, PADDLE_X = 99, 101, 72
NOOP, FIRE, RIGHT, LEFT = 0, 1, 2, 3


def make_env():
    env = gym.make("ALE/Breakout-v5", frameskip=4, repeat_action_probability=0,
                   render_mode="rgb_array")
    env = FireResetEnv(env)
    env = EpisodicLifeEnv(env)
    return env


def get_ram(env):
    return env.unwrapped.ale.getRAM()


def initial_frame_stack(obs):
    gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return [gray] * 4


def update_frame_stack(fs, obs):
    gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    fs.pop(0)
    fs.append(gray)
    return fs


if __name__ == "__main__":
    MODEL_PATH = "./models/PPO_131/final_model.zip"
    TELEPORT_OFFSET = 30
    FPS = 30

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--model": MODEL_PATH = args[i + 1]; i += 2
        elif args[i] == "--offset": TELEPORT_OFFSET = int(args[i + 1]); i += 2
        else: i += 1

    m = re.search(r"PPO_\d+[a-z]?", MODEL_PATH)
    run_name = m.group(0) if m else "model"

    model = PPO.load(MODEL_PATH, device="cuda")
    print(f"Ball-Teleport Watch — {run_name}")
    print(f"  LEFT:  FULL (normal ball)")
    print(f"  RIGHT: ALT (ball teleported +{TELEPORT_OFFSET}px)")
    print(f"  Press Q to quit, any other key for next game")
    print()

    while True:
        env_full = make_env()
        env_alt = make_env()

        obs_full, _ = env_full.reset()
        obs_alt, _ = env_alt.reset()
        fs_full = initial_frame_stack(obs_full)
        fs_alt = initial_frame_stack(obs_alt)

        # FIRE
        obs_full, _, _, _, _ = env_full.step(FIRE)
        obs_alt, _, _, _, _ = env_alt.step(FIRE)
        fs_full = update_frame_stack(fs_full, obs_full)
        fs_alt = update_frame_stack(fs_alt, obs_alt)

        # Wait for ball descent, then teleport ALT
        teleported = False
        for _ in range(25):
            full_ram = get_ram(env_full)
            alt_ram = get_ram(env_alt)
            if int(full_ram[BALL_Y]) < 180 and not teleported:
                bx = int(alt_ram[BALL_X])
                env_alt.unwrapped.ale.setRAM(BALL_X, max(10, min(150, bx + TELEPORT_OFFSET)))
                teleported = True
            if not teleported:
                obs_full, _, _, _, _ = env_full.step(NOOP)
                obs_alt, _, _, _, _ = env_alt.step(NOOP)
                fs_full = update_frame_stack(fs_full, obs_full)
                fs_alt = update_frame_stack(fs_alt, obs_alt)
            else:
                break

        done_full, done_alt = False, False
        full_score, alt_score = 0, 0
        frame = 0
        game_tracking, game_frames = 0, 0

        while not (done_full and done_alt) and frame < 15000:
            frame += 1

            # Predict
            if not done_full:
                act_full, _ = model.predict(np.expand_dims(fs_full, axis=0), deterministic=True)
                left_act = int(act_full[0])
            else:
                left_act = NOOP
            if not done_alt:
                act_alt, _ = model.predict(np.expand_dims(fs_alt, axis=0), deterministic=True)
                right_act = int(act_alt[0])
            else:
                right_act = NOOP

            # RAM
            full_ram = get_ram(env_full) if not done_full else None
            alt_ram = get_ram(env_alt) if not done_alt else None

            # Step envs
            if not done_full:
                if full_ram is not None and int(full_ram[BALL_Y]) > 180:
                    obs_full, r, term, trunc, _ = env_full.step(FIRE)
                else:
                    obs_full, r, term, trunc, _ = env_full.step(left_act)
                full_score += r
                if term or trunc:
                    try:
                        game_over = env_full.unwrapped.ale.lives() == 0
                    except:
                        game_over = True
                    if game_over:
                        done_full = True
                    else:
                        obs_full, _ = env_full.reset()
                        fs_full = initial_frame_stack(obs_full)
                        obs_full, _, _, _, _ = env_full.step(FIRE)
                        fs_full = update_frame_stack(fs_full, obs_full)
                        continue
                else:
                    update_frame_stack(fs_full, obs_full)

            if not done_alt:
                if alt_ram is not None and int(alt_ram[BALL_Y]) > 180:
                    obs_alt, r, term, trunc, _ = env_alt.step(FIRE)
                else:
                    obs_alt, r, term, trunc, _ = env_alt.step(right_act)
                alt_score += r
                if term or trunc:
                    try:
                        game_over = env_alt.unwrapped.ale.lives() == 0
                    except:
                        game_over = True
                    if game_over:
                        done_alt = True
                    else:
                        obs_alt, _ = env_alt.reset()
                        fs_alt = initial_frame_stack(obs_alt)
                        obs_alt, _, _, _, _ = env_alt.step(FIRE)
                        fs_alt = update_frame_stack(fs_alt, obs_alt)
                        # Re-teleport
                        for _ in range(15):
                            ram = get_ram(env_alt)
                            if int(ram[BALL_Y]) < 180:
                                bx = int(ram[BALL_X])
                                env_alt.unwrapped.ale.setRAM(BALL_X, max(10, min(150, bx + TELEPORT_OFFSET)))
                                break
                            obs_alt, _, _, _, _ = env_alt.step(NOOP)
                            fs_alt = update_frame_stack(fs_alt, obs_alt)
                        continue
                else:
                    update_frame_stack(fs_alt, obs_alt)

            # Tracking metric
            if full_ram is not None and alt_ram is not None:
                fby = int(full_ram[BALL_Y])
                aby = int(alt_ram[BALL_Y])
                if fby < 180 and aby < 180:
                    apx = int(alt_ram[PADDLE_X])
                    abx = int(alt_ram[BALL_X])
                    fbx = int(full_ram[BALL_X])
                    if abs(apx - abx) < abs(apx - fbx):
                        game_tracking += 1
                    game_frames += 1

            # Render — side by side (match watch_model_split.py style: 480x320, NEAREST)
            DISPLAY_W, DISPLAY_H = 480, 320

            if not done_full and obs_full is not None:
                left_rgb = cv2.cvtColor(obs_full, cv2.COLOR_RGB2BGR)
            else:
                left_rgb = np.zeros((210, 160, 3), dtype=np.uint8)
            if not done_alt and obs_alt is not None:
                right_rgb = cv2.cvtColor(obs_alt, cv2.COLOR_RGB2BGR)
            else:
                right_rgb = np.zeros((210, 160, 3), dtype=np.uint8)

            lf = cv2.resize(left_rgb, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_NEAREST)
            rf = cv2.resize(right_rgb, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_NEAREST)

            # Side labels — subtle, Atari-authentic (black bg, light text)
            cv2.putText(lf, "FULL", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
            cv2.putText(lf, f"Score: {int(full_score)}", (5, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1)

            cv2.putText(rf, "ALT (+30px)", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
            cv2.putText(rf, f"Score: {int(alt_score)}", (5, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1)

            # Tracking indicator
            if full_ram is not None and alt_ram is not None and not done_alt:
                apx = int(alt_ram[PADDLE_X])
                abx = int(alt_ram[BALL_X])
                fbx = int(full_ram[BALL_X])
                tracking = abs(apx - abx) < abs(apx - fbx)
                color = (0, 200, 0) if tracking else (200, 0, 0)
                cv2.putText(rf, "TRACKING" if tracking else "MISSING", (5, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

            # Divider
            divider = np.full((DISPLAY_H, 2, 3), 60, dtype=np.uint8)
            combined = np.hstack([lf, divider, rf])

            # Top bar
            left_act_name = {0: "NOOP", 1: "FIRE", 2: "RIGHT", 3: "LEFT"}.get(left_act, "?")
            right_act_name = {0: "NOOP", 1: "FIRE", 2: "RIGHT", 3: "LEFT"}.get(right_act, "?")
            div_text = " ***DIVERGE***" if left_act != right_act and not done_full and not done_alt else ""
            action_str = f"L:{left_act_name} R:{right_act_name}{div_text}"

            top_bar = np.full((24, combined.shape[1], 3), 0, dtype=np.uint8)
            cv2.putText(top_bar, f"Frame: {frame}  Action: {action_str}",
                        (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            display = np.vstack([top_bar, combined])
            cv2.imshow(f"{run_name} — Ball Teleport: FULL (L) vs TELEPORTED (R)", display)

            key = cv2.waitKey(1000 // FPS) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
                env_full.close()
                env_alt.close()
                exit()

        env_full.close()
        env_alt.close()
        pct = game_tracking / game_frames * 100 if game_frames > 0 else 0
        print(f"Game: FULL={int(full_score)} ALT={int(alt_score)} | tracking={pct:.0f}% | frames={frame}")
        print("Press any key for next game, Q to quit...")
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            break
