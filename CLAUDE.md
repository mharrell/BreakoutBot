# BreakoutBot — CLAUDE.md

## Project Identity

BreakoutBot is a solo PPO-based Atari Breakout RL project using Stable-Baselines3/PyTorch. The developer is Mr. Mike (address him as "Mr. Mike"). Single RTX 3060 Ti, Windows 11. The project investigates sticky actions (`repeat_action_probability=0.25`) as a mitigation for policy memorization in deterministic environments, and serves as a professional portfolio piece demonstrating hands-on ML engineering discipline.

**Repo:** [github.com/mharrell/BreakoutBot](https://github.com/mharrell/BreakoutBot)

## Truth-Source Hierarchy

1. **Ground truth:** `recordings/PPO_*_memorization_track.csv` (written by MemorizationCheckCallback, updated every 10M steps)
2. **Secondary:** Checkpoint filenames in `models/*/checkpoint/` (step count embedded in name)
3. **Tertiary:** TensorBoard `tensorboard/*/events.out.*` (binary, need TensorBoard to read)
4. **Documentation:** `EXPERIMENTS.md`, `RL_REFERENCE.md`, `FLAWS.md`, `.opencode/instructions.md` (human-maintained, may lag)

## Key Documentation

| File | Purpose |
|------|---------|
| `EXPERIMENTS.md` | Full experiment writeup — all three experiments, results, conclusions |
| `RL_REFERENCE.md` | PPO parameter guide, metric diagnostics, 31+ lessons, decision framework |
| `FLAWS.md` | **READ THIS before interpreting results.** Catalog of 20 known flaws in experimental process and data interpretation, with severity ratings |
| `.opencode/instructions.md` | Session bootstrap, agent guardrails, known misinterpretation traps |
| `LOGICAL_AUDIT.md` | **READ THIS alongside FLAWS.md.** Catalog of 16 logical flaws in reasoning, interpretation, and evidence standards. Complements FLAWS.md (which covers methodological flaws). |
| `archive/` | Historical training and evaluation scripts from Experiments 1-5. Recovered during 2026-07-19 logical audit cleanup. Reference only — not active code. See `archive/README.md`. |

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

## Known Methodological Limitations

Before interpreting any result, consult `FLAWS.md`. The most consequential active limitations:

- **Central finding (2026-07-14): No model in this project has ever genuinely generalized.** Every sticky-trained model tested without sticky actions (PPO_26, PPO_28, PPO_29, PPO_30b, PPO_31b) collapsed to a deterministic script. Sticky actions mask memorization with noise; they do not cure it. Deep non-sticky pretraining produces higher-scoring scripts (60 pts > 31 pts > 0 pts) but never reactive policies. See Critical Rule #10.
- **F-001 (CONFIRMED):** The MemorizationCheckCallback "GENERALIZING" verdict is INVALID for sticky models. Calibration: dead policy + p=0.25 noise = 8-14 unique scores (mean 11.3). At p=0.05: 55-63 unique scores. Nosticky verification is the only reliable behavioral test.
- **F-002:** Pretraining duration and sticky-step count are perfectly anti-correlated in Experiment 3. Both models are now confirmed memorized — the "trade-off" is between which script each memorized.
- **F-003 (RESOLVED 2026-07-14):** PPO_26 CONFIRMED MEMORIZED. Nosticky: every game = 60.0 points, 264 frames — a single fixed script. Deep non-sticky pretraining produces higher-scoring memorized scripts but does NOT produce generalization.
- **F-004 (RESOLVED):** PPO_31b's 10k-game evaluation complete (10,000 games). Stats: avg 22.2, 2.4% zero-score.
- **L-001 (confirmed 2026-07-19): Intervention test uncalibrated.** PPO_34 (confirmed dead argmax script: unique=1, std=0.0) retains 47.7% score under intervention — indistinguishable from PPO_35's reported 47%. The intervention test's retention percentage is not a reliable indicator of reactivity without a dead-model calibration baseline. See LOGICAL_AUDIT.md L-001.
- **L-007: GymBreakout-to-ALE transfer unvalidated.** All Experiments 5+ train on a custom engine. Zero validation that findings transfer to authentic ALE Breakout. All post-Experiment-4 conclusions should carry a "custom engine — ALE validation pending" caveat. See LOGICAL_AUDIT.md L-007.
- **L-014: eval_reactivity.py shape classifier uses uncalibrated thresholds.** The CLUSTERED/CONTINUOUS/UNCLEAR classification uses arbitrary cutoffs (top-3 >50%, <35%) with no statistical justification. Bootstrap CIs should be reported alongside point estimates. See LOGICAL_AUDIT.md L-014.

## Session Bootstrap (run these in order)

1. Read `recordings/PPO_*_memorization_track.csv` — ground-truth live state. **WARNING: meaningless for GymBreakout-trained models (PPO_33/34/35+) — the callback tests ALE, not GymBreakout. Use eval callback data (`logs/*/evaluations.npz`) and `eval_reactivity.py` output for these.**
2. Check `models/*/checkpoint/` — newest checkpoint filenames give actual step counts
3. Compare memorization track + checkpoint data against `.opencode/instructions.md` "Live Run Status" — flag discrepancies
4. Read `FLAWS.md` to refresh awareness of active limitations
5. Read `LOGICAL_AUDIT.md` to refresh awareness of reasoning pitfalls
6. If console logs exist: `Get-Content -Encoding Unicode recordings/PPO_*_console.log -Tail 30`
7. If interpreting intervention test results: verify that a dead-model calibration has been run (see LOGICAL_AUDIT.md L-001)

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
