"""
MemorizationCheckCallback — an SB3 training callback that periodically plays
a handful of games using the live in-memory model and checks for behavioral
collapse to a fixed action sequence (see EXPERIMENTS.md Experiment 2/3).

Runs entirely in-memory against self.model — no disk reload, no risk of
catching best_model.zip mid-write. Uses a dedicated single-env instance
separate from the training and eval envs so it doesn't interfere with
either's episode tracking.

Appends one row per check to {output_dir}/{run_name}_memorization_track.csv
so the trajectory across the whole run is visible later, not just the
latest snapshot.

Usage:
    from memorization_check_callback import MemorizationCheckCallback

    memorization_callback = MemorizationCheckCallback(
        run_name=RUN_NAME,
        sticky_actions=False,   # match whatever this script's training env uses
        check_freq=10_000_000,
        n_games=20,
    )
    callbacks = CallbackList([eval_callback, checkpoint_callback, memorization_callback])
"""
import os
import csv
import time
from datetime import datetime

import ale_py
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

gym.register_envs(ale_py)


class MemorizationCheckCallback(BaseCallback):
    def __init__(self, run_name, sticky_actions, check_freq=10_000_000,
                 n_games=20, output_dir="./recordings",
                 max_check_steps=200_000, verbose=1, summary_lines=None):
        super().__init__(verbose)
        self.run_name = run_name
        self.sticky_actions = sticky_actions
        self.check_freq = check_freq
        self.n_games = n_games
        self.output_dir = output_dir
        self.max_check_steps = max_check_steps  # safety cap, avoids hangs
        self.summary_lines = summary_lines or []
        self.last_check_step = 0
        self.track_log_path = os.path.join(
            output_dir, f"{run_name}_memorization_track.csv"
        )

    def _init_callback(self):
        os.makedirs(self.output_dir, exist_ok=True)
        is_new = not os.path.exists(self.track_log_path)
        self.track_log_file = open(self.track_log_path, "a", newline="", encoding="utf-8")
        self.track_log_writer = csv.writer(self.track_log_file)
        if is_new:
            for line in self.summary_lines:
                self.track_log_file.write(f"# {line}\n")
            self.track_log_writer.writerow([
                "check_timestamp", "training_step", "games_played",
                "unique_scores", "avg_score", "best_score",
                "worst_score", "verdict"
            ])
            self.track_log_file.flush()

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_check_step >= self.check_freq:
            self.last_check_step = self.num_timesteps
            self._run_check()
        return True  # never halts training

    def _run_check(self):
        if self.verbose:
            print(f"\n[MemorizationCheck] Running {self.n_games}-game check at "
                  f"step {self.num_timesteps:,}...")

        env_kwargs = {"repeat_action_probability": 0.25 if self.sticky_actions else 0.0}
        check_env = make_atari_env("ALE/Breakout-v5", n_envs=1, seed=None,
                                   env_kwargs=env_kwargs)
        check_env = VecFrameStack(check_env, n_stack=4)

        obs = check_env.reset()
        scores = []
        episode = 0
        steps_taken = 0
        start_time = time.time()

        while episode < self.n_games and steps_taken < self.max_check_steps:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = check_env.step(action)
            steps_taken += 1

            if done[0]:
                lives = info[0].get("lives", -1)
                if lives == 0:
                    score = float(info[0].get("episode", {}).get("r", 0))
                    scores.append(score)
                    episode += 1
                    obs = check_env.reset()
                else:
                    obs, _, _, _ = check_env.step([0])

        check_env.close()
        elapsed = time.time() - start_time

        if not scores:
            verdict = "INCOMPLETE"
            unique, avg, best, worst = 0, 0.0, 0.0, 0.0
            if self.verbose:
                print(f"[MemorizationCheck] No games completed within "
                      f"{self.max_check_steps:,} steps — skipping this check.")
        else:
            unique = len(set(scores))
            avg = sum(scores) / len(scores)
            best = max(scores)
            worst = min(scores)
            verdict = "MEMORIZED" if unique <= 2 else "GENERALIZING"

            if self.verbose:
                tag = "*** MEMORIZED ***" if verdict == "MEMORIZED" else "generalizing"
                print(f"[MemorizationCheck] Step {self.num_timesteps:,} | "
                      f"{len(scores)} games in {elapsed:.0f}s | "
                      f"unique={unique} | avg={avg:.1f} | best={best:.0f} | "
                      f"worst={worst:.0f} | {tag}")
                if verdict == "MEMORIZED":
                    print("[MemorizationCheck] Model appears to be playing a "
                          "fixed sequence at this step count. Training continues "
                          "— this is informational, not a stop condition.")

        self.track_log_writer.writerow([
            datetime.now().isoformat(), self.num_timesteps, len(scores),
            unique, round(avg, 1), best, worst, verdict
        ])
        self.track_log_file.flush()

    def _on_training_end(self) -> None:
        if hasattr(self, "track_log_file") and not self.track_log_file.closed:
            self.track_log_file.close()
