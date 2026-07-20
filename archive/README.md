# Archive — Historical Training & Evaluation Scripts

These scripts were used for Experiments 1-5 and their evaluations. They were removed from the project root during a cleanup in July 2026 but recovered from git history on 2026-07-19 during the Logical Audit cleanup.

**These are historical reference only.** They are not the active codebase. The only active training script is `train_ppo36.py` in the project root.

## Training Scripts

| Script | Experiment | Description |
|--------|-----------|-------------|
| `train_ppo30a.py` | Experiment 3, Phase 1 | PPO_30a — 100M non-sticky pretraining |
| `train_ppo30b.py` | Experiment 3, Phase 2 | PPO_30b — 300M sticky continuation |
| `train_ppo31a.py` | Experiment 3, Phase 1 | PPO_31a — 300M non-sticky pretraining |
| `train_ppo31b.py` | Experiment 3, Phase 2 | PPO_31b — 100M sticky continuation |
| `train_ppo32.py` | Experiment 4 | PPO_32 — p=0.05 sticky single-phase (killed at 93M) |
| `train_ppo33.py` | Experiment 5A | PPO_33 — frame skip randomization (5 restarts, killed) |
| `train_ppo34.py` | Experiment 5B | PPO_34 — per-episode physics randomization (completed 70M) |
| `train_ppo35.py` | Experiment 5C | PPO_35 — continuous mid-game physics (killed 268M) |

## Evaluation Scripts

| Script | Purpose |
|--------|---------|
| `funnel_recorder_ppo_*.py` | 10k-game single-env evaluations (gold standard) |
| `funnel_recorder_*_nosticky.py` | Sticky-off verification (500 games) |
| `eval_variance_test.py` | Deterministic vs. stochastic comparison |
| `sticky_probability_sweep.py` | p ∈ {0.0, 0.05, 0.10, 0.15, 0.20, 0.25} sweep |
| `calibrate_memorization_check.py` | MemorizationCheckCallback noise baseline calibration |
| `statistical_comparison.py` | Bootstrap CI and significance test utility |

## Why Archived

The project root was cleaned up to reduce clutter as the experimental focus shifted to GymBreakout dynamics randomization (Experiments 5+). However, deleting these scripts made the project unreproducible without git history access. They're kept here for reference.

## Reproducibility Note

To reproduce an archived experiment:
1. Check EXPERIMENTS.md for the exact configuration
2. Verify hyperparameters match the archived script
3. Note that models/checkpoints may no longer exist on disk
4. Some dependencies (ALE, gymnasium versions) may have changed

The archival doesn't guarantee bitwise reproducibility, but preserves the intent and configuration of each experiment.
