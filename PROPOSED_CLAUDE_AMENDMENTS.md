# Proposed CLAUDE.md Amendments — Logical Audit Cleanup (2026-07-19)

These amendments address the logical flaws identified in `LOGICAL_AUDIT.md`. They should be applied after Phase 1 calibration results are complete and reviewed.

---

## Amendment 1: New Critical Rules

Add after existing Critical Rule #10 (the list currently has 11 items; the numbering below is for the additions):

### 12. Never kill an experiment based on wrong-environment data.

PPO_35 was killed because its memorization track (ALE/Breakout-v5) showed 268 consecutive SINGLE_SCRIPT verdicts — but PPO_35 trained on GymBreakout, and the callback tested ALE. The project's own documentation warns this data is "meaningless for GymBreakout-trained models." If a metric comes from the wrong environment, it cannot support a kill decision. Always verify that the data source matches the training environment before acting on it.

### 13. Never claim causation from ≤3 data points without a statistical test.

Two snapshots cannot establish a trend. PPO_36's "dissolution regression" was diagnosed from two checkpoints 13M steps apart. Every model in this project's history shows wide checkpoint-to-checkpoint oscillation. Before claiming a directional change, compute whether the difference exceeds what would be expected from normal between-checkpoint variance.

### 14. Every new metric or classification must be calibrated against a known-dead baseline before being used to support claims.

This applies to: intervention test retention percentage, eval_reactivity.py shape classification (CLUSTERED/CONTINUOUS), top-3 concentration thresholds, "dissolution" trajectory analysis, and any future diagnostic. The calibration logic from F-001 (run a confirmed-dead model through the same test, measure what the "null" signal looks like) applies universally, not just to the MemorizationCheckCallback. If a dead script produces the same signal as the model being tested, the signal is not evidence of reactivity.

### 15. Never interpret "0% zero-score" as evidence of reactivity.

PPO_26 had 0% zero-score across 10,000 games and was a 60-point memorized script. Zero-zero-score means the policy never produces a score of exactly zero — it doesn't mean the policy tracks the ball. A dead script that consistently scores 5 points also has 0% zero-score. This is a floor-quality metric (the policy isn't completely broken), not a reactivity metric.

### 16. "Argmax-script + policy-entropy" and similar interpretive categories are hypotheses, not established entities.

Terms like "script diversification," "dissolution," and "CLUSTERED vs. CONTINUOUS" describe patterns in score distributions. They do not directly measure ball-tracking, state-conditioned action selection, or reactivity. The project's history shows that score diversity has multiple explanations (sticky noise, script-switching under stochastic sampling, cross-checkpoint cycling) — only one of which is genuine reactivity. Frame-level action analysis would be needed to distinguish these, and it has not been done. Treat these categories as descriptive labels, not as diagnostic verdicts.

---

## Amendment 2: Breakthrough Verification Protocol — Additional Gates

Add after existing Gate 6 in the protocol table:

| # | Gate | How to check | What it catches |
|---|------|-------------|-----------------|
| 7 | **Intervention calibration** | Run a confirmed-dead model (≤2 unique scores, det=True, no env noise) through the identical intervention test. If the dead model's retention % overlaps with the tested model's, the intervention result is not evidence of reactivity. | PPO_35 false positive (dead script = 47.7% retention, tested model = 47%) |
| 8 | **Adversarial review** | Before writing the claim, task an agent or reviewer with: "Using only existing data and documented project knowledge, try to refute this claim. List the strongest counterarguments." Document the refutation attempt in the memory file. | Asymmetric skepticism — the tendency to scrutinize abandoned hypotheses more rigorously than current ones |
| 9 | **Shape classifier calibration** | If using eval_reactivity.py shape verdicts (CLUSTERED/CONTINUOUS), bootstrap the thresholds: run the classifier on a known-dead model, compute 95% CI on top-3 concentration and singleton ratio. If the CI spans the CLUSTERED/CONTINUOUS boundary for either model, the verdict is UNCLEAR regardless of point estimate. | Threshold reification (L-014) |

---

## Amendment 3: Updated Session Bootstrap

Replace step 1 in Session Bootstrap:

**Old:**
> 1. Read `recordings/PPO_*_memorization_track.csv` — the only ground-truth live state

**New:**
> 1. Read `recordings/PPO_*_memorization_track.csv` — ground-truth live state. **WARNING: meaningless for GymBreakout-trained models (PPO_33/34/35+) — the callback tests ALE, not GymBreakout. Use eval callback data (`logs/*/evaluations.npz`) and `eval_reactivity.py` output for these models instead.**

Add new step:

> 7. If interpreting intervention test results: verify that a dead-model calibration has been run. If not, the retention % is uninterpretable. See LOGICAL_AUDIT.md L-001.

---

## Amendment 4: Key Documentation Table

Add to the Key Documentation table:

| `LOGICAL_AUDIT.md` | **READ THIS alongside FLAWS.md.** Catalog of 16 logical flaws in reasoning, interpretation, and evidence standards. Complements FLAWS.md (which covers methodological flaws). |
| `archive/` | Historical training and evaluation scripts from Experiments 1-5. Recovered during 2026-07-19 cleanup. Reference only — not active code. |

---

## Amendment 5: Known Methodological Limitations

Add to the Known Methodological Limitations section:

> - **L-001 (confirmed 2026-07-19): Intervention test uncalibrated.** PPO_34 (confirmed dead argmax script, unique=1, std=0.0) retains 47.7% score under intervention — indistinguishable from PPO_35's reported 47%. The intervention test's retention percentage is not a reliable indicator of reactivity without a dead-model calibration baseline. See LOGICAL_AUDIT.md L-001.
> - **L-007: GymBreakout-to-ALE transfer unvalidated.** All Experiments 5+ train on a custom engine. Zero validation that findings transfer to authentic ALE Breakout. All post-Experiment-4 conclusions should carry a "custom engine — ALE validation pending" caveat until the transfer is quantified. See LOGICAL_AUDIT.md L-007.
> - **L-014: eval_reactivity.py shape classifier uses uncalibrated thresholds.** The CLUSTERED/CONTINUOUS/UNCLEAR classification uses arbitrary cutoffs (top-3 >50%, <35%) with no statistical justification. Bootstrap CIs should be reported alongside point estimates. See LOGICAL_AUDIT.md L-014.

---

## Amendment 6: Conventions

Add to Conventions:

> - **Historical training scripts archived in `archive/`** — recovered from git history 2026-07-19. The only active training script is the one at the project root. See `archive/README.md`.
> - **Never kill a run based on wrong-environment memorization track data.** The callback's environment must match the training environment for the verdict to be meaningful. For GymBreakout-trained models, use `make_env_fn` with `MemorizationCheckCallback` (see its docstring).
> - **Intervention test results must include a dead-model calibration baseline.** Report: "Dead baseline retention: X%. Model retention: Y%. Difference: Z percentage points." Without the baseline, the number is uninterpretable.
