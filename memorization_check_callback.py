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

Environment selection:
  - Default (no make_env_fn): uses ALE/Breakout-v5 via make_atari_env.
    Backward-compatible with all existing training scripts.
  - With make_env_fn: uses the provided callable to create the check
    environment. The callable must return a VecFrameStack-wrapped env
    ready for inference. Use this for GymBreakout-trained models so
    the check tests the correct environment.

Stochastic check (check_deterministic_false=True):
  - After the standard det=True check, runs a second batch with det=False.
  - Appends additional columns: stoch_unique, stoch_avg, stoch_best,
    stoch_worst, stoch_verdict.
  - This is the reactivity signal — a SINGLE_SCRIPT det=True result with a
    MULTIPLE_SCRIPTS stoch result means the argmax is a script but the policy
    has real entropy.

IMPORTANT — verdict labels are descriptive, not diagnostic:
  SINGLE_SCRIPT    = 0-2 unique scores in n_games → policy plays a fixed
                     sequence (or near-fixed). Always a single argmax script.
  MULTIPLE_SCRIPTS = 3+ unique scores → policy is NOT playing one script.
                     Could mean: genuine ball-tracking reactivity, script-
                     switching between several memorized sequences, or
                     environment-noise-driven variation masking a dead policy.
                     Use eval_reactivity.py to determine which.

Never interpret MULTIPLE_SCRIPTS as "generalizes" or "reactive" without
further testing. Every false positive in this project's history came from
confusing score diversity with generalization.

Usage (ALE, backward-compatible):
    from memorization_check_callback import MemorizationCheckCallback

    memorization_callback = MemorizationCheckCallback(
        run_name=RUN_NAME,
        sticky_actions=False,
        check_freq=10_000_000,
        n_games=20,
    )

Usage (GymBreakout with stochastic check):
    from memorization_check_callback import MemorizationCheckCallback
    from gym_breakout import GymBreakout

    def make_check_env():
        env = GymBreakout(fixed=True)
        env = GrayscaleResize(env)
        env = ClipRewardEnv(env)
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])  # wrap for VecFrameStack
        env = VecFrameStack(env, n_stack=4)
        return env

    memorization_callback = MemorizationCheckCallback(
        run_name=RUN_NAME,
        sticky_actions=False,
        check_freq=1_000_000,
        n_games=20,
        make_env_fn=make_check_env,
        check_deterministic_false=True,
    )
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
                 max_check_steps=200_000, verbose=1, summary_lines=None,
                 make_env_fn=None, check_deterministic_false=False):
        """
        Args:
            run_name: Name prefix for output CSV files.
            sticky_actions: If True, set repeat_action_probability=0.25
                            (only used when make_env_fn is None / ALE default).
            check_freq: Steps between checks.
            n_games: Number of complete games to play per check.
            output_dir: Directory for CSV output.
            max_check_steps: Safety cap — aborts check if this many total
                             steps are taken without completing n_games.
            verbose: Print check results to console.
            summary_lines: Lines to write as comments at the top of a new CSV.
            make_env_fn: Optional callable that returns a VecFrameStack-wrapped
                         single env. If None, uses ALE/Breakout-v5 (backward
                         compatible). The callable must handle all env creation
                         and wrapping — the callback calls it as-is.
            check_deterministic_false: If True, run a second batch of n_games
                         with deterministic=False after the standard det=True
                         check. Adds stoch_* columns to the CSV.
        """
        super().__init__(verbose)
        self.run_name = run_name
        self.sticky_actions = sticky_actions
        self.check_freq = check_freq
        self.n_games = n_games
        self.output_dir = output_dir
        self.max_check_steps = max_check_steps
        self.summary_lines = summary_lines or []
        self.make_env_fn = make_env_fn
        self.check_deterministic_false = check_deterministic_false
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
            headers = [
                "check_timestamp", "training_step", "games_played",
                "unique_scores", "avg_score", "best_score",
                "worst_score", "verdict"
            ]
            if self.check_deterministic_false:
                headers.extend([
                    "stoch_unique_scores", "stoch_avg_score",
                    "stoch_best_score", "stoch_worst_score",
                    "stoch_verdict"
                ])
            self.track_log_writer.writerow(headers)
            self.track_log_file.flush()

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_check_step >= self.check_freq:
            self.last_check_step = self.num_timesteps
            self._run_check()
        return True  # never halts training

    def _make_default_env(self):
        """Create the default ALE Breakout env (backward-compatible)."""
        env_kwargs = {"repeat_action_probability": 0.25 if self.sticky_actions else 0.0}
        check_env = make_atari_env("ALE/Breakout-v5", n_envs=1, seed=None,
                                    env_kwargs=env_kwargs)
        check_env = VecFrameStack(check_env, n_stack=4)
        return check_env

    def _run_episodes(self, env, deterministic, label_suffix=""):
        """Run self.n_games episodes and return (scores, elapsed_seconds)."""
        obs = env.reset()
        scores = []
        episode = 0
        steps_taken = 0
        start_time = time.time()

        mode_label = "det=True" if deterministic else "det=False"
        if self.verbose:
            print(f"[MemorizationCheck{label_suffix}] Running {self.n_games}-game "
                  f"{mode_label} check at step {self.num_timesteps:,}...")

        while episode < self.n_games and steps_taken < self.max_check_steps:
            action, _ = self.model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
            steps_taken += 1

            if done[0]:
                lives = info[0].get("lives", -1)
                if lives == 0:
                    score = float(info[0].get("episode", {}).get("r", 0))
                    scores.append(score)
                    episode += 1
                    obs = env.reset()
                else:
                    # Life lost — fire to respawn
                    obs, _, _, _ = env.step([1])

        elapsed = time.time() - start_time
        return scores, elapsed

    @staticmethod
    def _compute_stats(scores):
        """Return (unique, avg, best, worst, verdict) from a list of scores."""
        if not scores:
            return 0, 0.0, 0.0, 0.0, "INCOMPLETE"
        unique = len(set(scores))
        avg = sum(scores) / len(scores)
        best = max(scores)
        worst = min(scores)
        verdict = "SINGLE_SCRIPT" if unique <= 2 else "MULTIPLE_SCRIPTS"
        return unique, round(avg, 1), best, worst, verdict

    def _log_results(self, scores, elapsed, label):
        """Print results to console. Returns verdict string."""
        unique, avg, best, worst, verdict = self._compute_stats(scores)
        if self.verbose and scores:
            tag = "*** SINGLE_SCRIPT ***" if verdict == "SINGLE_SCRIPT" else "MULTIPLE_SCRIPTS"
            print(f"[MemorizationCheck{label}] Step {self.num_timesteps:,} | "
                  f"{len(scores)} games in {elapsed:.0f}s | "
                  f"unique={unique} | avg={avg:.1f} | best={best:.0f} | "
                  f"worst={worst:.0f} | {tag}")
            if verdict == "SINGLE_SCRIPT":
                print(f"[MemorizationCheck{label}] Model plays a fixed sequence "
                      f"at this step count. Training continues — this is "
                      f"informational, not a stop condition.")
        return unique, avg, best, worst, verdict

    def _run_check(self):
        # ---- Build env ----
        if self.make_env_fn is not None:
            check_env = self.make_env_fn()
        else:
            check_env = self._make_default_env()

        # ---- det=True check (always runs) ----
        det_scores, det_elapsed = self._run_episodes(check_env, deterministic=True)
        det_unique, det_avg, det_best, det_worst, det_verdict = \
            self._log_results(det_scores, det_elapsed, "")

        check_env.close()

        # ---- det=False check (optional) ----
        stoch_unique = stoch_avg = stoch_best = stoch_worst = 0.0
        stoch_verdict = ""
        stoch_scores = []

        if self.check_deterministic_false:
            if self.make_env_fn is not None:
                check_env = self.make_env_fn()
            else:
                check_env = self._make_default_env()

            stoch_scores, stoch_elapsed = self._run_episodes(
                check_env, deterministic=False, label_suffix=" stoch")
            stoch_unique, stoch_avg, stoch_best, stoch_worst, stoch_verdict = \
                self._log_results(stoch_scores, stoch_elapsed, " stoch")

            check_env.close()

            # Cross-check: det=True SINGLE_SCRIPT + det=False MULTIPLE_SCRIPTS =
            # argmax is a script but policy has entropy.
            if self.verbose and det_verdict == "SINGLE_SCRIPT" and stoch_verdict == "MULTIPLE_SCRIPTS":
                print(f"[MemorizationCheck] *** NOTE: det=True is SINGLE_SCRIPT but "
                      f"det=False is MULTIPLE_SCRIPTS ({stoch_unique} unique). "
                      f"Argmax collapses to a script, but the policy retains "
                      f"useful entropy. This is the PPO_35 pattern — see "
                      f"eval_reactivity.py for full verification.")

        # ---- Write CSV row ----
        row = [
            datetime.now().isoformat(), self.num_timesteps,
            len(det_scores) if det_scores else 0,
            det_unique, det_avg, det_best, det_worst, det_verdict
        ]
        if self.check_deterministic_false:
            row.extend([
                stoch_unique, stoch_avg, stoch_best, stoch_worst, stoch_verdict
            ])
        self.track_log_writer.writerow(row)
        self.track_log_file.flush()

    def _on_training_end(self) -> None:
        if hasattr(self, "track_log_file") and not self.track_log_file.closed:
            self.track_log_file.close()
