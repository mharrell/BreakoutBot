# BreakoutBot — CLAUDE.md

## Project Identity

BreakoutBot is a solo PPO-based Atari Breakout RL project using Stable-Baselines3/PyTorch. The developer is Mr. Mike (address him as "Mr. Mike"). Single RTX 3060 Ti, Windows 11. The project investigates sticky actions (`repeat_action_probability=0.25`) as a mitigation for policy memorization in deterministic environments, and serves as a professional portfolio piece demonstrating hands-on ML engineering discipline.

**Repo:** [github.com/mharrell/BreakoutBot](https://github.com/mharrell/BreakoutBot)

## New Session Quickstart (READ THIS FIRST)

If you're a new Claude session, start here:

1. **Read `CURRENT_STATE.md`** — the definitive status document. Claim status board, model roster, what we've learned, what's next. 5-minute read. Everything else references this.

2. **The one-paragraph brief:**
   > After 117 Breakout and 2 BeamRider experiments, no PPO model has ever genuinely generalized on any Atari game. NatureCNN CAN track the ball perfectly (1.9px MAE, proven) but PPO never learns to use those features. The perception-policy gap has three confirmed chapters: (1) CNN encodes ball position but policy ignores it, (2) aux supervision bakes features in but policy still ignores them, (3) cursor wrapper shapes the distribution to track the ball but the argmax still ignores it. BeamRider was claimed as the first verified reactive argmax — but split-watcher verification on July 30 revealed BOTH BeamRider models are SINGLE_SCRIPT (std=0.0, unique=1). The distribution-vs-argmax confound is universal across environments. Every diagnostic except the split-watcher measures the policy distribution, not the argmax. Before making any claim about any model, run split-watcher verification: if the agent moves identically on a different game state (px_corr > 0.99), the argmax is memorized.

3. **What NOT to trust:**
   - MemorizationCheckCallback "GENERALIZING" verdicts for sticky models — confirmed invalid (F-001)
   - **MULTIPLE_SCRIPTS memcheck verdicts without split-watcher confirmation** — PPO_114/115 had MULTIPLE_SCRIPTS but confirmed memorized (F-005)
   - **Intervention probe reversal rates as evidence of argmax reactivity** — measures distribution shifts, not argmax changes. Models with 33-50% reversal confirmed memorized by split-watcher (F-006)
   - Intervention test retention percentages without dead-model calibration (L-001)
   - det=False score diversity as evidence of reactivity — dead scripts produce it too (L-012)
   - Shape classifier verdicts (CLUSTERED/CONTINUOUS) without bootstrap CIs (L-014)
   - Any finding from Experiments 5+ on the custom engine — ALE transfer gap confirmed (L-007)
   - Memory files with "TENTATIVE:" prefix — not yet validated
   - **Any "reactive" signal not verified by split-watcher** — every diagnostic except split-watcher measures distribution, not argmax. This now includes BeamRider memcheck/eval results — both models SINGLE_SCRIPT under split-watcher.

4. **Then run the full Session Bootstrap** below.

## Truth-Source Hierarchy

1. **Ground truth:** `CURRENT_STATE.md` — claim status board, confirmed/tentative/falsified verdicts
2. **Primary data:** `calibration_phase1_results.json` (dead-model calibration), `cross_eval_PPO_35_results.json` (ALE transfer gap), `recordings/PPO_*_memorization_track.csv` (memorization tracks — meaningless for GymBreakout-trained models)
3. **Secondary:** Checkpoint filenames in `models/*/checkpoint/` (step count embedded in name)
4. **Tertiary:** TensorBoard `tensorboard/*/events.out.*` (binary, need TensorBoard to read)
5. **Documentation:** `CURRENT_STATE.md` > `LOGICAL_AUDIT.md` > `FLAWS.md` > `EXPERIMENTS.md` > `RL_REFERENCE.md` (human-maintained, may lag behind ground truth)

## Key Documentation

| File | Purpose |
|------|---------|
| `CURRENT_STATE.md` | **READ THIS FIRST.** Definitive status — claim board, model roster, lessons learned, next steps. Updated 2026-07-30. |
| `FINDINGS_2026_07_30.md` | **Split-watcher verification report.** All cursor models confirmed memorized. Distribution-vs-argmax confound documented. |
| `DIAGNOSTIC_IDEAS.md` | Reference for building new diagnostics. Decision points, dead baselines, implementation notes. |
| `LOGICAL_AUDIT.md` | 17-entry logical flaw catalog. L-001/002/007 confirmed with data. Complements FLAWS.md. |
| `FLAWS.md` | 23-entry methodological flaw catalog with severity ratings. Read before interpreting any result. |
| `EXPERIMENTS.md` | Full experiment writeup — all experiments, results, conclusions. Updated 2026-07-27. |
| `RL_REFERENCE.md` | PPO parameter guide, metric diagnostics, 31+ lessons, decision framework |
| `COMBINATION_MATRIX.md` | Anti-memorization method results matrix (OF, YP, RS, HE, Dropout combos). |
| `REVENGE_BRUNCH.md` | RBO project — superhuman Breakout via deep pretraining + sticky. |
| `REACTIVITY_ANALYSIS.md` | Intervention test results (historical — conclusions falsified, see correction notice). |

## Critical Rules (Never Do These)

1. **Never judge a model by eval score alone.** Eval score and single-env quality are inverted across PPO_25/26/27 (see RL_REFERENCE.md Lesson #23). Always verify against single-env funnel data.
2. **Never trust `explained_variance=1.0` or `value_loss≈0`.** These are memorization collapse signatures, not signs of a perfect model (RL_REFERENCE.md Lesson #30).
3. **Never report results as "final" until the funnel log has exactly 10,000 data rows** (not counting header). Verify with `wc -l`.
4. **Never attribute an outcome to one variable without listing every other variable that changed.** The pretraining/sticky-step anti-correlation in Experiment 3 (FLAWS.md F-002) is the canonical example of this error.
5. **Never trust the GENERALIZING verdict for sticky-action models — it's CONFIRMED invalid.** A dead policy + p=0.25 noise produces 8-14 unique scores. The only reliable behavioral test is nosticky verification: run the model without sticky actions and check for collapse to ≤2 unique scores. See `calibrate_memorization_check.py` and FLAWS.md F-001.
6. **Never compare models across experiments without checking n_envs, LR restart values, clip_range, and total step counts.** These differ between Experiment 1 and 3 (FLAWS.md F-006, F-015).
7. **Never propose an experiment without checking whether it was already tried and rejected.** See EXPERIMENTS.md Option D and the full run history in RL_REFERENCE.md Part 6.
8. **Always run nosticky verification on EVERY sticky-trained model before claiming it generalizes.** PPO_30b, PPO_31b, and PPO_26 ALL appeared to generalize with sticky on but collapsed to deterministic scripts without sticky. Every sticky-trained model ever tested in this project has been memorized. No exceptions.
9. **Never conclude a policy is dead from deterministic inference alone.** PPO_30b with det=True, sticky=off: 2 unique scores, 99.8% zeros. Same model with det=False, sticky=off: 43 unique scores, avg 23.5. The argmax can collapse while the policy retains useful entropy. Always test both.
10. **Never claim that sticky fine-tuning cures memorization.** Every sticky-trained model ever tested in this project (PPO_26, PPO_28, PPO_29, PPO_30b, PPO_31b) collapsed to a deterministic script without sticky actions. Sticky actions mask memorization with noise; they do not prevent or cure it. The only untested path is preventing memorization from forming during early training (Experiment 4: low-sticky single-phase).
11. **Never make design decisions silently — always present them before implementing.** Any new script, wrapper, or experiment component has design decisions embedded in it (parameter values, distribution shapes, what to include or exclude from standard pipelines). Before writing code, surface each decision explicitly: what it is, what the options are, what the recommendation is, and why. Then get explicit approval. Do not write the code first and explain the decisions after. This applies to everything from a 30-line wrapper to a full experiment design. It also means: never launch a training run without explicit confirmation — that includes not structuring integration tests in a way that could accidentally start training.
12. **Never kill an experiment based on wrong-environment data.** PPO_35 was killed because its memorization track (ALE/Breakout-v5) showed 268 SINGLE_SCRIPT verdicts — but PPO_35 trained on GymBreakout, and the callback tested ALE. The project's own documentation warns this data is "meaningless for GymBreakout-trained models." If a metric comes from the wrong environment, it cannot support a kill decision. Always verify that the data source matches the training environment before acting on it. See LOGICAL_AUDIT.md L-003.
13. **Never claim causation from ≤3 data points without a statistical test.** Two snapshots cannot establish a trend. PPO_36's "dissolution regression" was diagnosed from two checkpoints 13M steps apart. Every model in this project's history shows wide checkpoint-to-checkpoint oscillation. Before claiming a directional change, compute whether the difference exceeds what would be expected from normal between-checkpoint variance. See LOGICAL_AUDIT.md L-004.
14. **Every new metric or classification must be calibrated against a known-dead baseline before being used to support claims.** This applies to: intervention test retention percentage, eval_reactivity.py shape classification (CLUSTERED/CONTINUOUS), top-3 concentration, "dissolution" trajectory analysis, and any future diagnostic. The calibration logic from F-001 (run a confirmed-dead model through the same test) applies universally. If a dead script produces the same signal as the model being tested, the signal is not evidence of reactivity. See LOGICAL_AUDIT.md L-001, L-014.
15. **Never interpret "0% zero-score" as evidence of reactivity.** PPO_26 had 0% zero-score across 10,000 games and was a 60-point memorized script. Zero-zero-score means the policy never produces a score of exactly zero — it doesn't mean the policy tracks the ball. A dead script that consistently scores 5 points also has 0% zero-score. This is a floor-quality metric, not a reactivity metric. See LOGICAL_AUDIT.md L-006.
16. **Treat interpretive categories as descriptive labels, not diagnostic verdicts.** Terms like "argmax-script + policy-entropy," "script diversification," "dissolution," and "CLUSTERED vs. CONTINUOUS" describe patterns in score distributions. They do not directly measure ball-tracking, state-conditioned action selection, or reactivity. Score diversity has multiple explanations (noise masking, script-switching under sampling, cross-checkpoint cycling) — only one of which is genuine reactivity. Frame-level action analysis would be needed to distinguish these and has not been done. See LOGICAL_AUDIT.md L-012.

17. **Never claim argmax reactivity without split-watcher verification.** Every diagnostic in this project except the split-watcher measures the policy distribution, not the argmax. The intervention probe, intervention gradient, SCAD probe, memcheck, and brick layout test all measure properties of the probability distribution. PPO_107-117 all showed positive signals on these diagnostics and were confirmed memorized by split-watcher. Run `verify_split_watcher.py` before any reactivity claim. If any game shows perfect transfer (px_corr > 0.99 with ALT score ≈ FULL score), the argmax is memorized. Period.

18. **The intervention probe and gradient measure distribution shifts, not argmax changes.** A model with 90% confidence on its top action can show real probability shifts (the other 10% redistributes) without ever changing which action it takes. The 33-50% reversal rates from PPO_107+ are real distribution shifts — the distributions noticed the teleported ball — but the argmax didn't change. Never report intervention probe results as "paddle reversed" or "policy tracked ball" — report them as "distribution shifted."

## Known Methodological Limitations

Before interpreting any result, consult `FLAWS.md`. The most consequential active limitations:

- **Central finding (updated 2026-07-30): No PPO model in this project has ever genuinely generalized — on any Atari game.** Every sticky-trained model tested without sticky actions collapsed to a deterministic script. Every cursor model confirmed memorized by split-watcher (F-005, F-006, F-007). Both BeamRider models confirmed SINGLE_SCRIPT (F-008). The distribution-vs-argmax confound is universal. Sticky actions mask memorization with noise; they do not cure it. Deep non-sticky pretraining produces higher-scoring scripts (60 pts > 31 pts > 0 pts) but never reactive policies.
- **F-001 (CONFIRMED):** The MemorizationCheckCallback "GENERALIZING" verdict is INVALID for sticky models. Calibration: dead policy + p=0.25 noise = 8-14 unique scores (mean 11.3). At p=0.05: 55-63 unique scores. Nosticky verification is the only reliable behavioral test.
- **F-002:** Pretraining duration and sticky-step count are perfectly anti-correlated in Experiment 3. Both models are now confirmed memorized — the "trade-off" is between which script each memorized.
- **F-003 (RESOLVED 2026-07-14):** PPO_26 CONFIRMED MEMORIZED. Nosticky: every game = 60.0 points, 264 frames — a single fixed script. Deep non-sticky pretraining produces higher-scoring memorized scripts but does NOT produce generalization.
- **F-004 (RESOLVED):** PPO_31b's 10k-game evaluation complete (10,000 games). Stats: avg 22.2, 2.4% zero-score.
- **L-001 (confirmed 2026-07-19): Intervention test uncalibrated.** PPO_34 (confirmed dead argmax script: unique=1, std=0.0) retains 47.7% score under intervention — indistinguishable from PPO_35's reported 47%. The intervention test's retention percentage is not a reliable indicator of reactivity without a dead-model calibration baseline. See LOGICAL_AUDIT.md L-001.
- **L-007 (CONFIRMED 2026-07-19): GymBreakout-to-ALE transfer is catastrophic.** PPO_35 cross-evaluated on ALE/Breakout-v5: GymBreakout 212 pts → ALE 2 pts (99.1% drop). The custom engine does not approximate authentic Atari Breakout. All post-Experiment-4 conclusions are custom-engine findings pending ALE replication. See LOGICAL_AUDIT.md L-007.
- **L-014: eval_reactivity.py shape classifier uses uncalibrated thresholds.** The CLUSTERED/CONTINUOUS/UNCLEAR classification uses arbitrary cutoffs (top-3 >50%, <35%) with no statistical justification. Bootstrap CIs should be reported alongside point estimates. See LOGICAL_AUDIT.md L-014.
- **F-005 (CONFIRMED 2026-07-30): MULTIPLE_SCRIPTS memcheck verdicts can be false positives from timing variance.** PPO_114 and PPO_115 had MULTIPLE_SCRIPTS on det=True but confirmed memorized by split-watcher. Score variance from life-loss timing and ball-bounce stochasticity produces multiple unique scores from a single memorized sequence. The split-watcher is the definitive behavioral test.
- **F-006 (CONFIRMED 2026-07-30): The intervention probe and gradient measure distribution shifts, not argmax changes.** All cursor models (PPO_107-117) showed positive intervention signals (33-50% reversal, AUC=0.33) while confirmed memorized by split-watcher. The probe measures P(action | ball_before) ≠ P(action | ball_after) — a real distribution shift that doesn't change the argmax. The dead baseline (0%) is correct but insufficient: a second calibration point is needed — a model known to have reactive distributions but memorized argmax. PPO_116 fits this profile.
- **F-007 (CONFIRMED 2026-07-30): Perfect transfer (px_corr > 0.99 on altered layout) is definitive memorization.** A reactive policy cannot produce identical paddle positions on different brick layouts because different bricks cause different ball bounces. Every cursor model tested showed at least one perfect-transfer game.
- **F-008 (CONFIRMED 2026-07-30): BeamRider reactivity claim is FALSIFIED.** BEAMRIDER_baseline: SINGLE_SCRIPT, 4200 pts every game, std=0.0. BEAMRIDER_MULTILIFE: SINGLE_SCRIPT, 2160 pts every game, std=0.0. The "first verified reactive PPO argmax" was the same distribution-vs-argmax confound as Breakout — score variance mistaken for argmax diversity. The distribution-vs-argmax confound is universal across Atari games.

## Session Bootstrap (run these in order)

0. **Read `CURRENT_STATE.md`** — claim status board, model roster, what to trust/distrust. 5-minute orientation.
1. **Read `FINDINGS_2026_07_30.md`** — split-watcher verification results, distribution-vs-argmax confound documentation.
2. Read `recordings/PPO_*_memorization_track.csv` — ground-truth live state. **WARNING: MULTIPLE_SCRIPTS verdicts can be false positives from timing variance (F-005). Verify with split-watcher before trusting.**
3. Check `models/*/checkpoint/` — newest checkpoint filenames give actual step counts
4. Compare memorization track + checkpoint data against `CURRENT_STATE.md` — flag discrepancies
5. Read `FLAWS.md` to refresh awareness of active limitations
6. Read `LOGICAL_AUDIT.md` to refresh awareness of reasoning pitfalls
7. If console logs exist: `Get-Content -Encoding Unicode recordings/PPO_*_console.log -Tail 30`
8. **If interpreting ANY reactivity signal:** run `verify_split_watcher.py` before claiming reactivity. The intervention probe, intervention gradient, SCAD, and memcheck ALL measure distribution, not argmax. Only the split-watcher measures the argmax directly.
9. If interpreting intervention test results: it measures distribution shifts, not argmax changes. A positive signal does not mean the paddle tracks the ball — it means the probability distribution changed (F-006).

## Conventions

- Each PPO run gets its own `train_ppoNN.py` file
- Phase 1 (non-sticky) → Phase 2 (sticky) via separate scripts (e.g., train_ppo30a.py → train_ppo30b.py)
- Use `remaining = TARGET - model.num_timesteps` for continuation (not `reset_num_timesteps`)
- Conservative LR restart at phase switch: 1e-4→1e-5 (not 2.5e-4)
- n_envs=32, batch_size=1024, n_steps=128, n_epochs=4, gamma=0.99, ent_coef=0.006
- Validate findings at full sample size (10k games) before drawing conclusions
- **Every Phase 2 model must have a corresponding `funnel_recorder_{RUN_NAME}_nosticky.py`**
- **PPO_26 nosticky verification COMPLETE** — confirmed memorized (60-point, 264-frame script × 500 games)
- All training scripts and most standalone scripts must have an `if __name__ == "__main__":` guard — importing a script (for testing, introspection, or documentation) must never start a training run or evaluation as a side effect. This was added retroactively to all `train_ppo*.py` files on 2026-07-14
- **Design decisions must be presented before implementation, not discovered after.** For any new script, wrapper, or component: list each decision point, the options, the recommendation, and the rationale. Get explicit approval before writing code. See Critical Rule #11
- After experiment completion, cross-check EXPERIMENTS.md tables against raw CSV data in `recordings/`
- Read `FLAWS.md` before writing any new conclusions
