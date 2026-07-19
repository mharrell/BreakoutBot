# BreakoutBot — CLAUDE.md

## Project Identity

BreakoutBot is a solo PPO-based Atari Breakout RL project using Stable-Baselines3/PyTorch. The developer is Mr. Mike (address him as "Mr. Mike"). Single RTX 3060 Ti, Windows 11.

**Current direction (July 2026):** Dynamics randomization on authentic ALE Breakout using `setRAM()` for runtime state perturbation. After 43 PPO runs on a custom Breakout engine produced promising reactivity results, we're validating against the real game.

**Repo:** [github.com/mharrell/BreakoutBot](https://github.com/mharrell/BreakoutBot)

## What We Know (Empirically, Not Dogma)

These are findings from experiments, not laws of nature. They hold on our hardware with our config. Challenge them with data.

1. **Sticky actions don't prevent memorization in deterministic Breakout.** Confirmed across 6 PPO models. Sticky noise masks memorized scripts; nosticky verification reveals collapse. This matches Zhang et al. (2018).
2. **Dynamics randomization (perturbing environment physics during training) forces the policy to maintain action-distribution entropy.** PPO_35 on our custom engine, tested with the intervention protocol (teleport the ball mid-game), retains 47% of normal score. Noise-only models collapse to 8-26%.
3. **The argmax (det=True) can be a deterministic script while the policy distribution (det=False) contains useful entropy.** This is not a contradiction — it's expected behavior in a deterministic MDP. A perfect policy in a deterministic environment produces identical outcomes from identical starts. The intervention test is what distinguishes a blind script from a sighted one.
4. **Latent dropout stabilizes policy entropy under dynamics randomization.** PPO_36 (ball noise + dropout): zero entropy collapses across 294M steps. PPO_37 (ball noise only): 4 collapses in 100M steps.

## Truth-Source Hierarchy (Fixed July 2026)

The old hierarchy was circular — the "ground truth" CSV was known to be invalid for half the models. This is the corrected version:

1. **For behavioral questions (is the policy reactive?):** `eval_reactivity.py` output and `eval_intervention.py` output. These directly measure whether the policy responds to game state.
2. **For training progress:** Checkpoint filenames in `models/*/checkpoint/` and eval callback data in `logs/*/evaluations.npz`.
3. **For memorization tracking (ALE-trained models only):** `recordings/PPO_*_memorization_track.csv`. WARNING: this data is from the MemorizationCheckCallback and is only valid when the callback environment matches the training environment. For custom-engine models, it tests the wrong environment and is meaningless.
4. **Documentation:** `EXPERIMENTS.md`, `RL_REFERENCE.md`, `FLAWS.md` (human-maintained, may lag).

## Key Documentation

| File | Purpose |
|------|---------|
| `EXPERIMENTS.md` | Full experiment writeup — starts with current direction and rationale |
| `RL_REFERENCE.md` | PPO parameter guide, metric diagnostics, lessons learned |
| `FLAWS.md` | **READ THIS before interpreting results.** Catalog of known flaws |
| `EVALUATION_PROTOCOL.md` | Standardized protocol for running and interpreting evaluations |
| `REACTIVITY_ANALYSIS.md` | Intervention test results — the most direct behavioral measurement |

## Critical Rules

These are guardrails derived from confirmed mistakes. They are stronger than conventions.

1. **Never judge a model by eval score alone.** Eval score and single-env quality were inverted across PPO_25/26/27 (RL_REFERENCE.md Lesson #23).
2. **Never trust `explained_variance=1.0` or `value_loss≈0`.** These are memorization collapse signatures (RL_REFERENCE.md Lesson #30).
3. **Never report results as "final" without verifying row counts.** Check with `wc -l`.
4. **Never attribute an outcome to one variable without listing every other variable that changed.** See FLAWS.md F-002 for the canonical violation.
5. **Never use "unique score count" alone as a reactivity metric.** In a deterministic environment with a deterministic policy, a perfectly reactive policy can produce 1 unique score. Score diversity measures environmental + policy stochasticity, not reactivity. The intervention test (`eval_intervention.py`) is the correct diagnostic.
6. **Never compare models across experiments without checking n_envs, LR, clip_range, and total step counts.** See FLAWS.md F-006, F-015.
7. **Never propose an experiment without checking whether it was already tried.** See EXPERIMENTS.md and RL_REFERENCE.md Part 6.
8. **Never conclude a policy is dead from deterministic inference alone.** PPO_30b: det=True → 2 unique scores, 99.8% zeros. Same model, det=False → 43 unique scores, avg 23.5. The argmax can collapse while the policy retains entropy.
9. **Never claim "generalization" or "reactivity" without completing the diagnostic checklist** (see below).
10. **Never make design decisions silently.** Surface each decision, the options, the recommendation, and the rationale. Get explicit approval before writing code. This includes: parameter values, distribution shapes, what to include/exclude from standard pipelines. Never launch a training run without explicit confirmation.

## Diagnostic Checklist for Reactivity Claims

Before writing ANY claim that a model is reactive, tracks the ball, or generalizes:

- [ ] **1. Both inference modes tested** — `det=True` AND `det=False`, ≥100 games each, on the evaluation environment (no training noise)
- [ ] **2. Intervention test run** — `eval_intervention.py` on the same model. Does the policy adapt when the ball is teleported? Retention >30% = sighted; retention <15% = blind.
- [ ] **3. Comparison baseline** — at least one other model tested under identical conditions. If a "failed" approach produces similar numbers, the claim is wrong.
- [ ] **4. Distribution shape analyzed** — `eval_reactivity.py` reports clustering vs. continuity. Single-script argmax + diverse sampling is expected; it's not a contradiction.
- [ ] **5. Environment match confirmed** — does the evaluation environment match the training environment's observation space? Are you testing on the same engine you trained on?
- [ ] **6. Falsification test stated** — write: "This claim would be proven wrong if _______." Run that test before writing the claim.

### Important Distinction: Determinism of Outcome ≠ Blindness

In a fully deterministic environment, a perfect state-conditioned policy produces the SAME outcome from the SAME starting state every time. `det=True` producing 1 unique score with std=0.0 is EXPECTED — it does NOT indicate memorization. The intervention test distinguishes:
- **Blind script**: teleport the ball → policy keeps playing the timed sequence → score collapses
- **Sighted script**: teleport the ball → policy adjusts actions to new position → score is retained

Both produce 1 unique score under normal conditions. Only the intervention test tells them apart.

## Session Bootstrap

1. Read `EXPERIMENTS.md` "Project Direction" section — understand current state and direction
2. Read `FLAWS.md` — refresh awareness of active limitations
3. Check `models/*/checkpoint/` — newest checkpoint filenames give actual step counts
4. If live runs exist, check `.opencode/instructions.md` "Live Run Status"
5. If evaluating a model, run the **Diagnostic Checklist** above before writing claims

## Conventions

- Each PPO run gets its own `train_ppoNN.py` file
- Use `remaining = TARGET - model.num_timesteps` for continuation (not `reset_num_timesteps`)
- n_envs=32, batch_size=1024, n_steps=128, n_epochs=4, gamma=0.99, ent_coef=0.006
- All training scripts must have an `if __name__ == "__main__":` guard
- Validate findings at full sample size (10k games for gold-standard, 500 for verification)
- After experiment completion, cross-check EXPERIMENTS.md tables against raw data
- Read `FLAWS.md` before writing any new conclusions
- **Design decisions must be presented before implementation.** List each decision, options, recommendation, and rationale. Get explicit approval before writing code.

## Custom Engine Status

`breakout_env_vendor/` and `gym_breakout.py` remain in the repo for reference and for the intervention test infrastructure. No new training runs should be launched on the custom engine. All new experiments use ALE with `setRAM()`-based randomization.

The custom engine's contributions:
- Proved dynamics randomization > sticky actions for forcing policy entropy
- Proved dropout stabilizes entropy under perturbation
- Developed the intervention test methodology
- Produced the "sighted script" insight (PPO_35)

Its limitations (why we're moving to ALE):
- No ball speed-up mechanic
- Different ball/paddle geometry
- Simplified collision physics
- Custom rendering not validated against real Atari
