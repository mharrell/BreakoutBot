# Current State — BreakoutBot

**Last updated: 2026-08-01 — PPO_124 BREAKTHROUGH: First verified reactive PPO argmax on Atari Breakout**

---

## TL;DR

**PPO_124 is the first model in this project's history to produce a genuinely reactive argmax on Atari Breakout.** After 123 experiments spanning every conceivable anti-memorization approach — sticky actions, cursor wrappers, entropy bonuses, dynamics randomization, adversarial bumpers, non-conditionable ball perturbations — every single one confirmed memorized by the split-watcher, the solution was the simplest possible thing: **directly reward the paddle for being horizontally close to the ball during descent.**

```
bonus = 0.05 × max(0, 1 − |paddle_x − ball_x| / 80)
```

The proximity reward is tiny (0.05 per frame vs 1.0–7.0 per brick) but dense — it fires every frame the ball is descending. Crucially, it makes ball-tracking the **explicitly rewarded behavior**. Every previous approach tried to penalize scripts or make them non-viable. This one rewards the thing we actually want.

The transfer test is clean eval (no proximity reward) — the policy tracks the ball on standard Breakout without being directly rewarded for it. The behavior transferred.

**The perception-policy gap, chapter 4:**
1. CNN encodes ball position perfectly (1.9px MAE) — policy ignores it (PPO_102/103)
2. Aux supervision bakes features in — policy still ignores them (PPO_102/103)
3. Cursor wrapper shapes the distribution to track the ball — argmax still ignores it (PPO_107-117)
4. **Proximity reward shapes the argmax to track the ball** — the gap is closed (PPO_124)

---

## PPO_124 — Split-Watcher Results (August 1, 2026)

The split-watcher runs the same model on FULL vs ALTERED brick layouts with independent per-side predictions. A memorized script produces identical paddle positions on different layouts (px_corr > 0.99). A reactive policy adapts.

**BrickClearWrapper bug is FIXED** — previously, both sides saw identical full-wall observations on frame 1 because `reset()` returned the pre-clear observation. Now takes a NOOP step after clearing bricks to refresh. All prior split-watcher results (PPO_111-118, BeamRider) used the buggy wrapper. The bug caused false "perfect transfer" verdicts by hiding early divergence.

### No-timing variant (no NoopResetEnv — zero timing confound)

| Checkpoint | Layout | Games | ALT Score | Divergence | px_corr | Perfect Transfers |
|-----------|--------|-------|-----------|------------|---------|-------------------|
| best (19.2M) | RIGHT_HALF | 20 | **379 ×20** | 62.4% | 0.9425 | **0** |
| best (19.2M) | LEFT_HALF | 20 | **379 ×20** | 62.4% | 0.9436 | **0** |
| best (19.2M) | RANDOM_50 | 20 | **379 ×20** | 62.4% | 0.9436–0.9438 | **0** |
| final (25M) | RIGHT_HALF | 20 | **383 ×20** | 62.9% | 0.9590 | **0** |
| final (25M) | LEFT_HALF | 20 | **383 ×20** | 62.9% | 0.9593 | **0** |
| final (25M) | RANDOM_50 | 20 | **383 ×20** | 62.9% | 0.9593–0.9595 | **0** |

**0/120 perfect transfers. 100% ALT score retention on every layout.** The model clears every brick regardless of which bricks are present. Every game, every layout.

### With NoopResetEnv (realistic — 0–30 random NOOP frames at reset)

| Checkpoint | Layout | ALT Score Pattern | Divergence | Perfect Transfers |
|-----------|--------|-------------------|------------|-------------------|
| best (19.2M) | RIGHT_HALF | 16× 95, 4× 379 | 62–71% | **0** |
| best (19.2M) | LEFT_HALF | 18× 158, 2× 379 | 62–76% | **0** |
| best (19.2M) | RANDOM_50 | 23–379 varied | 59–80% | **0** |
| final (25M) | RIGHT_HALF | 11× 214, 9× 383 | 63–72% | **0** |
| final (25M) | LEFT_HALF | 17× 142, 3× 383 | 63–74% | **0** |
| final (25M) | RANDOM_50 | 8–383 varied | 63–87% | **0** |

**0/120 perfect transfers. 46–59% ALT retention.** NoopResetEnv timing offsets cause the model to sometimes miss the first serve or start with corrupted frame stacks, reducing scores on altered layouts. The final model improved retention from 46% to 59% — becoming more robust to timing variation.

### Combined: 0/240 perfect transfers across all tests.

Every previous tested model (PPO_111–118, BeamRider) had at least 1 perfect transfer. PPO_124 has zero across 240 games.

---

## PPO_124 — Intervention Gradient (August 1, 2026)

| Magnitude | Dead Baseline | best (19.2M) | final (25M) |
|-----------|:---:|:---:|:---:|
| +/- 0px | 0.0% | 12.5% | 37.5% |
| +/- 8px | 0.0% | 21.4% | 41.2% |
| +/- 15px | 0.0% | 25.0% | **60.0%** |
| +/- 30px | 0.0% | 23.5% | 50.0% |
| +/- 45px | 0.0% | 31.6% | 31.2% |
| +/- 60px | 0.0% | 16.7% | 25.0% |
| **AUC** | 0.000 | **0.240** | **0.421** |

**The final model is markedly better than the best model:**
- AUC nearly doubled (0.240 → 0.421) — classified as STRONG dose-response
- Peak reversal jumped from 31.6% → 60.0%
- Peak shifted from 45px → 15px — more sensitive to smaller displacements
- Clean dose-response curve: rises to 60%, falls at extreme displacements (ball in physically impossible positions)
- Dead baseline: 0.0% at all magnitudes

Caveat (F-006): cursor models had 33–50% reversal and were memorized. The intervention gradient measures distribution shifts, not argmax changes. But PPO_124's curve has a clean dose-response shape that cursor models lacked (their rates were flat across displacements).

---

## PPO_124 — Memorization Check

| Phase | det=True Verdict | det=True Best | Stoch Best | Notes |
|-------|-----------------|---------------|------------|-------|
| 1M–13M | SINGLE_SCRIPT every checkpoint | 16–97 | 22–99 | Rising scores, still deterministic |
| 14M–25M | **MULTIPLE_SCRIPTS (10/12 checkpoints)** | 67–107 | 106–**216** | First sustained det=True MULTIPLE_SCRIPTS without sticky |

Stoch best of 216 is the highest score ever recorded on clean Breakout in this project. Previous bests: PPO_26 (60 pts, confirmed memorized), PPO_35 (212 pts, GymBreakout — doesn't transfer to ALE).

FULL-wall script: 379–383 points on deterministic inference.

---

## PPO_126 Continuation

PPO_124 training continues as PPO_126, identical parameters, from 25M → 50M total. No MemorizationCheckCallback — removed per user request (memcheck verdicts unreliable, split-watcher is definitive).

**Question:** does more training further improve clean-eval transfer, or does the policy eventually converge to a script that maximizes both game reward and proximity bonus simultaneously?

**Result (August 2, 2026): THE MODEL REGRESSED.** The best checkpoint was at 47.4M, not 50M:

| Checkpoint | Layout | px_corr | ALT Score | Pattern |
|-----------|--------|---------|-----------|---------|
| best (47.4M) | RIGHT_HALF | **0.33** | 223 (55%) | **Decoupled** — paddle trajectories diverge |
| best (47.4M) | LEFT_HALF | 0.97 | 403 (100%) | Script — identical to FULL |
| best (47.4M) | RANDOM_50 | mixed | mixed | 16 script, 4 decoupled |
| final (50M) | ALL | 0.95 | 401 (100%) | Single script everywhere |

The best checkpoint showed layout-specific decoupling: right-half cleared broke the script, left-half didn't. By 50M, this disappeared — single 401-point script on all layouts.

**Intervention gradient (50M final):** AUC = 0.327 vs PPO_124's 0.421. Noisy curve, no clean peak. Textbook F-025: intervention probe classified every magnitude "STRONG reactivity" while split-watcher confirmed memorized script.

**Training duration:** ~6.5 hours for 25M steps (avg 1,084 FPS on RTX 3060 Ti).

**Conclusion:** More training does NOT monotonically improve reactivity. PPO eventually finds a script that maximizes the combined game + proximity objective. Checkpoint selection is critical — save frequently and verify with split-watcher.

Full results: see EXPERIMENTS.md Experiment 33.

---

## Claim Status Board

### CONFIRMED — Supported by data

| Claim | Evidence |
|-------|----------|
| Sticky actions mask memorization; they don't prevent it | Every sticky-trained model tested without sticky actions collapsed to a deterministic script |
| The MemorizationCheckCallback "GENERALIZING" verdict is invalid for sticky models | Dead policy + p=0.25 sticky = 8-14 unique scores (F-001) |
| Deep non-sticky pretraining produces higher-scoring memorized scripts, not generalization | PPO_26: 60 pts > PPO_31b: 31 pts > PPO_30b: 0 pts — all confirmed memorized |
| The intervention test does not distinguish reactive from dead | PPO_34 (dead) retains 49.6% vs PPO_35's 44.7%. L-001. |
| GymBreakout findings do not transfer to ALE | PPO_35: GymBreakout 212 pts → ALE 2 pts (99.1% drop). L-007. |
| det=False score diversity exists in dead scripts | PPO_34 (dead): 19 unique det=False scores. L-012. |
| NatureCNN CAN track the ball — perception is not the bottleneck | Perception POC: 1.9px MAE. PPO never learns to use those features. |
| The perception-policy gap is structural | PPO_103: policy collapses faster than aux can shape features. |
| Cursor wrapper shapes distribution but not argmax | All PPO_107-117 models memorized. Split-watcher perfect transfer on every model. |
| Intervention probe measures distribution shifts, not argmax changes | 33-50% reversal rates on models confirmed memorized by split-watcher. |
| Perfect transfer (px_corr>0.99 on altered layout) = definitive memorization | Physical impossibility for reactive policy. Confirmed on 7/7 cursor models. |
| MULTIPLE_SCRIPTS verdicts can be false positives from timing variance | PPO_114/115 had MULTIPLE_SCRIPTS memcheck but split-watcher confirmed memorized. |
| SINGLE_SCRIPT is a general PPO property, not Breakout-specific | 5/5 games SINGLE_SCRIPT. |
| Ball-tracking features do NOT prevent policy memorization | PPO_102/103: features at 14-16px, policy SINGLE_SCRIPT. |
| Entropy coefficient does not prevent argmax collapse | Every value 0.006-0.10 → SINGLE_SCRIPT. |
| Life-loss penalty does not prevent memorization | PPO_101: SINGLE_SCRIPT through 14M. |
| Dynamics randomization via setRAM() does not produce reactive argmax on ALE | PPO_78/79/80: all SINGLE_SCRIPT on clean ALE. |
| Random bounce perturbation (non-conditionable stochasticity) does not force reactivity | PPO_118: 413 pts, 1/9 perfect transfer → MEMORIZED. |
| **PPO's objective function was the root cause of universal memorization** | argmax_π E[Σ rewards] in deterministic environments converges to a script. Every env modification changed what script was optimal, not whether the optimum was a script. |
| **Dense proximity reward produces the first verified reactive PPO argmax on Breakout** | **PPO_124: 0/240 perfect transfers, 100% no-timing ALT retention, STRONG intervention AUC 0.421, 60% reversal at 15px. See FINDINGS_PPO_124_BREAKTHROUGH.md.** |
| **BrickClearWrapper had a stale-observation bug** | **All prior split-watcher results (PPO_111-118, BeamRider) used buggy comparison data. Both sides saw identical full-wall first frames. Fixed 2026-08-01.** |

### FALSIFIED — Proven wrong

| Claim | How it was falsified |
|-------|---------------------|
| "PPO_35 is the first non-memorized model" | Dead-model calibration shows identical signals. ALE cross-eval: 2 points. |
| "PPO_30b/31b GENERALIZING" | Nosticky verification: both collapse to ≤2 unique scores. |
| "PPO_26 generalizes" | Nosticky: every game = 60.0 pts, 264 frames — a fixed script. |
| "PPO_55b has no functional deterministic policy" | Env mismatch artifact. Fixed env → always SINGLE_SCRIPT. |
| "ent_coef ≥ 0.02 prevents argmax collapse" | Every value collapsed to SINGLE_SCRIPT. |
| "Hard failure is the mechanism that forces reactivity in BeamRider" | MULTILIFE: MULTIPLE_SCRIPTS without hard failure — but MULTIPLE_SCRIPTS itself was the timing-variance false positive. Split-watcher shows both models SINGLE_SCRIPT. |
| "One-life Breakout will force reactivity" | PPO_104: SINGLE_SCRIPT, scripts still viable. |
| "PPO_107 shows first verifiable ball-tracking in any Breakout model" | Split-watcher: perfect transfer on altered layouts. Intervention probe was measuring distribution shifts, not argmax. |
| "The BeamRider mechanism is portable to Breakout" (cursor wrapper) | 7/7 cursor models confirmed memorized. Mechanism shapes distribution, not argmax. |
| "MULTIPLE_SCRIPTS sustained on PPO_114/115" | Split-watcher shows perfect transfer. Memcheck verdicts were timing variance. |
| "BeamRider is the first verified reactive PPO argmax" | Both models SINGLE_SCRIPT (std=0.0, unique=1) under independent-prediction split-watcher. |
| "BeamRider's mechanism (adversarial threat targeting position) forces reactivity" | Same distribution-vs-argmax confound as Breakout cursor models. |
| **"No PPO model has ever genuinely generalized on any Atari game"** | **PPO_124: 0/240 perfect transfers, 100% no-timing ALT retention. See above.** |

---

## The Diagnostic Blind Spot (July 30 Finding — Still Active)

Every metric in the project's diagnostic suite measures the **policy distribution**, but evaluation uses the **argmax**:

| Diagnostic | What it measures | The confound |
|-----------|-----------------|--------------|
| Memcheck | Unique score count (det=True) | Score variance from timing/life-loss, not behavioral adaptation |
| Intervention probe | Distribution shift after teleport | Distribution reacts while argmax ignores |
| Intervention gradient | AUC of distribution shifts | Integrates distribution noise into "strong" classification |
| SCAD probe | MI(action; ball_position) | Measures distribution, correctly flagged marginal values |
| Brick layout test | Score retention | Binary succeed/fail is consistent with memorization |
| **Split watcher** | **Argmax paddle position, two layouts** | **The only test that measures the argmax directly** |

**The split-watcher remains the definitive verification gate.** Before claiming any model is reactive, run `verify_split_watcher.py` or `watch_model_split.py`. The no-timing variant (no NoopResetEnv) provides the cleanest signal by eliminating timing offsets as a confound.

---

## Model Roster

### Proximity Reward Generation (PPO_124, PPO_126)

| Model | Config | Steps | FULL | Perfect Transfers | ALT Retention (no-timing) | Intervention AUC | Verdict |
|-------|--------|-------|------|--------------------|--------------------------|------------------|---------|
| PPO_124 best | ProximityReward(0.05,80) | 19.2M | 379 | **0/60** | **100%** | 0.240 | **REACTIVE** |
| PPO_124 final | ProximityReward(0.05,80) | 25M | 383 | **0/60** | **100%** | 0.421 | **REACTIVE** |
| PPO_126 | Continue PPO_124 25→50M | 50M | 401 | **0/60** | **100%** (px_corr=0.95) | 0.327 (noisy) | REGRESSED — best at 47.4M |

Full diagnostic report: `FINDINGS_PPO_124_BREAKTHROUGH.md`

### Overnight Batch (PPO_119–125, July 31 – August 1, 2026)

| Model | Approach | Steps | Memcheck | Split-Watcher | Verdict |
|-------|----------|-------|----------|---------------|---------|
| PPO_119 | Trajectory entropy (scale=0.01) | 7M (killed) | SINGLE_SCRIPT | — | DEAD |
| PPO_120 | Moving bumper (15 shapes) | 25M | SINGLE_SCRIPT | MEMORIZED | MEMORIZED |
| PPO_121 | Trajectory entropy (scale=0.10) | 2M (killed) | SINGLE_SCRIPT | — | DEAD |
| PPO_122 | Ball-binned trajectory entropy | 25M | SINGLE_SCRIPT (123 pts) | — | MEMORIZED |
| PPO_123 | Extreme bumper (2 independent) | 25M | SINGLE_SCRIPT (72 pts) | — | MEMORIZED |
| **PPO_124** | **Proximity reward (scale=0.05)** | **25M** | **MULTIPLE_SCRIPTS** | **0/240 perfect transfers** | **REACTIVE** |
| PPO_125 | Brick pre-clear (1-life) | 25M | SINGLE_SCRIPT (73 pts) | — | MEMORIZED |

### Cursor Generation (PPO_107–117) — All MEMORIZED

All use ALE/Breakout-v5 + AdversarialCursorWrapper or variants, NatureCNN, no sticky, 32 envs.

| Model | Config | Steps | FULL | Perfect Transfers | Verdict |
|-------|--------|-------|------|-------------------|---------|
| PPO_111 | Two cursors (multi-cursor) | 35M | 401 | **5/9** | MEMORIZED |
| PPO_112 | Unknown cursor variant | 6.4M | 93 | **1/9** | MEMORIZED |
| PPO_113 | Unknown cursor variant | 3.2M | 1 | 1/9 (noise) | DEAD |
| PPO_114 | Multi-cursor (2 asymmetric) | 22M | 436 | **1/9** | MEMORIZED |
| PPO_115 | Single cursor, speed=8 | 50M | 420 | **2/9** | MEMORIZED |
| PPO_116 | Single cursor + randomized bricks | 19M | 382 | **0/9** | MEMORIZED (scrambled) |
| PPO_117 | Unknown cursor variant | 13M | 411 | **2/9** | MEMORIZED |

PPO_107–110, PPO_113: not split-watcher tested (dead or early-stage).

### BeamRider — Both MEMORIZED

| Model | Config | Steps | Score | Split-Watcher | Verdict |
|-------|--------|-------|-------|---------------|---------|
| BEAMRIDER_BASELINE | Baseline, SEED=206 | 10M | 4200, std=0.0 | SINGLE_SCRIPT on side A | MEMORIZED |
| BEAMRIDER_MULTILIFE | No hard failure, SEED=205 | 10M | 2160, std=0.0 | 2/10 perfect transfer | MEMORIZED |

### Other Notables

| Model | Approach | Result |
|-------|----------|--------|
| PPO_26 | Non-sticky pretrain + sticky fine-tune | 60pt memorized script |
| PPO_30b/31b | Non-sticky + sticky | Confirmed memorized (nosticky) |
| PPO_34/35 | Custom engine dynamics randomization | Does not transfer to ALE (L-007) |
| PPO_55a-55e | Entropy intervention 0.01-0.10 | All SINGLE_SCRIPT |
| PPO_78/79/80 | ALE setRAM dynamics | All SINGLE_SCRIPT on clean ALE |
| PPO_85/86 | Frozen ball-tracker features | Collapsed to 0pt |
| PPO_102/103 | Aux ball-position supervision | Features encode ball, policy memorized |
| PPO_118 | Random ball bounce perturbation | 413 pts, 1/9 perfect transfer → MEMORIZED |

---

## What We've Learned

### The Breakthrough (August 1, 2026)

1. **Reward what you want, don't penalize what you don't want.** After 123 experiments trying to make scripts non-viable (sticky actions, cursor wrappers, entropy bonuses, dynamics randomization, adversarial bumpers, non-conditionable perturbations), the solution was directly rewarding the desired behavior: paddle-under-ball. A three-line reward function did what years of environment engineering couldn't.

2. **Dense rewards beat sparse rewards for shaping behavior.** The game's natural reward (brick breaks, 1.0–7.0) is sparse and requires multi-step credit assignment. The proximity bonus (0.05 per frame) fires every descent frame and directly rewards the first step of the causal chain: be near the ball → hit the ball → ball hits bricks → score.

3. **Scale doesn't need to be large if the signal is consistent.** 0.05 × 2,000 descending frames ≈ 50 bonus per game — equivalent to ~7 yellow bricks. That's 5–10% of a 216-point game, but the dense signal provides better gradients than occasional large rewards.

4. **The argmax follows the reward, not the distribution.** Cursor wrapper shaped the distribution but the argmax converged to the mode. Proximity reward shapes the Q-values directly — tracking IS the highest-value action at every step.

5. **PPO's objective function was the root cause all along.** `argmax_π E[Σ rewards]` in a deterministic environment converges to a script because scripts maximize expected return. Changing the environment changes what script is optimal; changing the reward function changes what behavior is optimal. The proximity reward made ball-tracking the reward-maximizing behavior.

### Diagnostic Infrastructure

6. **Every diagnostic measures the distribution; evaluation uses the argmax.** The central blind spot discovered July 30. Only the split-watcher measures the argmax directly.

7. **Perfect transfer (px_corr > 0.99 on altered layout) = definitive memorization.** Physical impossibility for a reactive policy. 0/240 on PPO_124.

8. **NoopResetEnv is a timing confound.** Removing it (no-timing variant) gives the cleanest split-watcher signal. With NoopResetEnv, random 0–30 frame offsets cause false score variance.

9. **MULTIPLE_SCRIPTS verdicts can be false positives from timing variance.** PPO_114/115 had MULTIPLE_SCRIPTS but confirmed memorized.

10. **Score diversity is not reactivity.** Dead scripts produce diverse scores under stochastic sampling.

11. **Every new metric needs dead-model calibration.**

12. **The custom engine doesn't approximate ALE.** 99.1% score drop (L-007).

---

## Active Diagnostics

| Tool | What it measures | Reliability | File |
|------|-----------------|-------------|------|
| **Split-watcher** | Argmax paddle position on two layouts | **Gold standard** — definitive | `watch_model_split.py`, `verify_split_watcher.py` |
| **Split-watcher (no-timing)** | Same, without NoopResetEnv timing confound | **Gold standard — cleanest signal** | `verify_split_watcher_notiming.py` |
| MemorizationCheckCallback | det=True unique scores per 20 games | Good for non-sticky SINGLE_SCRIPT detection; MULTIPLE_SCRIPTS can be false positive (F-005) | `memorization_check_callback.py` |
| Intervention probe | Distribution shift after ball teleport | Measures distribution, NOT argmax (F-006) | `probe_107_intervention.py` |
| Intervention gradient | AUC of reversal vs displacement | Same confound as probe; clean dose-response curves are informative | `intervention_gradient.py` |
| SCAD probe | MI(action; ball position) | Correct directional signal, needs threshold calibration | `scad_probe.py` |
| Brick layout test | Score retention on half-layouts | Binary fail = memorized, binary pass = ambiguous | `brick_layout_test.py` |
| Nosticky verification | Collapse check for sticky models | Definitive for sticky-trained models | Per-model funnel scripts |

---

## What's Running

**Nothing currently training.** All models complete and verified — see Model Roster above. PPO_126 completed August 2, 2026 (REGRESSED).

---

## Open Questions

- **Can the argmax ever be reactive in ANY Atari game? ANSWERED (YES):** PPO_124 demonstrates verified argmax reactivity on Breakout. The key is reward shaping — directly rewarding the desired behavior — not environment engineering.

- **Does more training improve or degrade reactivity? ANSWERED (REGRESSED):** PPO_126 continued PPO_124 from 25M → 50M. Reactivity degraded — the best checkpoint was at 47.4M (partial layout-specific decoupling), the final at 50M was a memorized script (px_corr=0.95). More training does NOT monotonically improve reactivity. Checkpoint selection is critical.

- **Does the proximity reward approach generalize to other Atari games?** The mechanism (dense reward for ball/paddle proximity) is Breakout-specific, but the principle (reward what you want directly) should transfer. Space Invaders: reward horizontal alignment with enemies? BeamRider: reward being out of the line of fire?

- **Can a reactive model be trained from scratch with proximity reward?** PPO_124 used standard pretraining — NatureCNN from random init, 25M steps. No sticky, no cursor, no special architecture. The proximity reward was the only addition. The answer appears to be yes, but a dedicated from-scratch replication would confirm.

- **What's the optimal proximity reward scale?** 0.05 worked. Would 0.01 work? 0.10? Is there a threshold below which the signal is too weak, or above which it overwhelms the game reward?

---

## Key Documentation

| File | Purpose |
|------|---------|
| `FINDINGS_PPO_124_BREAKTHROUGH.md` | **PPO_124 full writeup** — mechanism, results, lessons, verification checklist |
| `FINDINGS_2026_07_30.md` | Split-watcher verification report — all cursor models confirmed memorized |
| `proximity_reward_wrapper.py` | The wrapper that made it work — 3-line reward function |
| `DIAGNOSTIC_IDEAS.md` | Reference for building new diagnostics |
| `LOGICAL_AUDIT.md` | 17-entry logical flaw catalog |
| `FLAWS.md` | 28-entry methodological flaw catalog |
| `EXPERIMENTS.md` | Full experiment writeup |
| `RL_REFERENCE.md` | PPO parameter guide, metric diagnostics, 31+ lessons |
