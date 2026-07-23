# Current State — BreakoutBot

**Last updated: 2026-07-23 (Y-perturb and entropy experiments complete)**

---

## TL;DR

After 58+ PPO runs, **no model in this project has ever genuinely generalized** to reactive ball-tracking in Breakout. The return to authentic ALE Breakout using `setRAM()` for dynamics randomization has produced the same pattern seen on the custom engine: det=False maintains score diversity while det=True collapses to a single memorized argmax script. Y-perturb (ball Y-axis perturbation via `setRAM(101)`, ±8px, cooldown=30f, prob=10%) sustains stochastic diversity but does not prevent the argmax from memorizing. Entropy coefficient increases from 0.006 up to 0.10 (16.7×) do not prevent argmax collapse — they only affect the quality of the resulting script. The next direction is higher perturbation probability (prob ≥ 0.25) to test whether more frequent perturbation makes memorized scripts non-viable.

A critical infrastructure bug was discovered and fixed during this experimental cycle: `make_check_env` was missing `EpisodicLifeEnv` and `AutoResetWrapper`, causing the MemorizationCheckCallback to produce 18+ consecutive INCOMPLETE det=True verdicts for PPO_55b — all of which were false negatives. With the fix, det=True always completes and the true argmax behavior is visible from the first check. All training scripts have been updated.

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
| Paddle-bounce teleport at 10-20% does not force reactivity | PPO_44/45/46: three dead scripts, ~0 mean score. Models learned avoidance. |
| **Y-perturb at 10% does not prevent argmax memorization on ALE** | PPO_55/57/58: all det=True SINGLE_SCRIPT by 12-16M, det=False MULTIPLE_SCRIPTS sustained |
| **Entropy coefficient does not prevent argmax collapse** | 55a (0.01), 55b (0.02), 55c (0.04), 55d (0.025), 55e (0.10) — all SINGLE_SCRIPT on det=True |
| **INCOMPLETE det=True verdicts (July 22-23) were false positives from env mismatch** | make_check_env lacked EpisodicLifeEnv; callback never detected game-over. Fixed in all 14 scripts. |
| **Run-to-run variance in memorization trajectory is real** | PPO_55/57/58: identical configs, different seed → different det=False peak timing and magnitude |

### TENTATIVE — Plausible but not confirmed

| Claim | What's needed to confirm |
|-------|------------------------|
| Higher perturbation probability (≥0.25) makes memorized scripts non-viable | Train PPO at prob=0.25 and prob=0.50; if det=True shows MULTIPLE_SCRIPTS, confirmed |
| The argmax-script + policy-entropy pattern (det=True script, det=False diverse) is the universal outcome of moderate dynamics randomization | More perturbation types beyond Y-axis position needed to establish generality |
| The det=False score peak at ~10M is a real phenomenon across runs | Confirmed across PPO_55/57/58 but mechanism still unknown |

### FALSIFIED — Proven wrong

| Claim | How it was falsified |
|-------|---------------------|
| "PPO_35 is the first non-memorized model" | Dead-model calibration shows identical signals. ALE cross-eval: 2 points. |
| "PPO_30b/31b GENERALIZING" | Nosticky verification: both collapse to ≤2 unique scores. Sticky noise, not generalization. |
| "PPO_26 generalizes" | Nosticky: every game = 60.0 pts, 264 frames — a fixed script. |
| **"PPO_55b has no functional deterministic policy" (18+ INCOMPLETE checks)** | Env mismatch artifact. With fixed env, det=True completes on every check and is always SINGLE_SCRIPT. |
| **"ent_coef ≥ 0.02 prevents argmax collapse"** | 55b (0.02), 55d (0.025), 55c (0.04), 55e (0.10) all collapsed to SINGLE_SCRIPT. The argmax concentrates regardless of entropy coefficient. |

---

## Model Roster

### ALE Return — Y-Perturb Experiments (Current Generation)

All use ALE/Breakout-v5 + `ALEBreakoutYPerturb` (setRAM 101, ±8px, cooldown=30f).
Training: 32 envs, NatureCNN, no sticky, LR 2.5e-4→1e-5, clip 0.2→0.05, ent_coef=0.006 (except entropy variants).

#### Y-Only Baseline Replicates

| Model | Seed | Target | Final Step | det=True | det=False (final) | Notes |
|-------|------|--------|------------|----------|-------------------|-------|
| PPO_55 | default | 50M | 48M | SINGLE_SCRIPT ~15 pts | 10 unique, avg 9.5, best 17 | First Y-only. det=False peak at 10M: 9 unique, avg 11.6, best 16 |
| PPO_57 | 57 | 50M | 48M | SINGLE_SCRIPT | 12 unique, avg ~14, best ~24 | Stronger det=False than PPO_55. Confirmed 10M peak. |
| PPO_58 | 58 | 50M | 48M | SINGLE_SCRIPT ~11-13 pts | 12 unique, avg ~11, best ~24 | Third replicate. Classic pattern. |

**Finding:** Identical configs with different seeds produce meaningfully different score trajectories. The 10M det=False diversity peak is independently confirmed. All three converge to argmax scripts by 12-16M.

#### X-Mirror and Combined (Ablation — Killed)

| Model | Perturbation | Final Step | Outcome |
|-------|-------------|------------|---------|
| PPO_51 | X-mirror 10%, cooldown=30 | 48M | det=True INCOMPLETE* (env bug), det=False: 8 unique, avg 9.4, best 13 |
| PPO_52 | X-mirror 20%, cooldown=30 | 12M | Killed early — 20% too aggressive |
| PPO_53 | X-mirror 5%, cooldown=60 | 48M | det=True INCOMPLETE* (env bug), det=False: 12 unique, avg 12.8, best 20 |
| PPO_54 | X+Y combined, 10% each, c=30 | 22.4M | Killed — SINGLE_SCRIPT at 21M, dual 10% too disruptive |
| PPO_56 | X+Y gentle, 5% each, c=60 | 16M | Killed — stuck at 4 pts SINGLE_SCRIPT |

**Finding:** X-mirror with cooldown (PPO_51/53) shows same det=False diversity as Y-perturb. PPO_53 (5%/60f) hit det=False best=21 — strong for an X-only perturbation. *PPO_51 and 53's INCOMPLETE det=True verdicts are suspect (same env bug as PPO_55b); their true argmax behavior is unknown. X+Y combined at 10% is too disruptive. X+Y at 5% is too gentle.

#### Entropy Intervention (from PPO_55 9.6M checkpoint)

| Model | ent_coef | Multiplier | Final Step | det=True | det=False (final) |
|-------|----------|------------|------------|----------|-------------------|
| PPO_55a | 0.010 | 1.67× | 22.4M | SINGLE_SCRIPT by 16.6M | Collapsed. Killed. |
| PPO_55b | 0.020 | 3.33× | 42.4M | SINGLE_SCRIPT ~13-25 pts | 9-12 unique, avg 10-14, best 23. Running when stopped. |
| PPO_55c | 0.040 | 6.67× | 25.6M | SINGLE_SCRIPT by 14.6M | Collapsed. Killed. |
| PPO_55d | 0.025 | 4.17× | 24.2M | SINGLE_SCRIPT ~15-33 pts | 10-13 unique, avg 8-15, caught 33-pt script. Running when stopped. |
| PPO_55e | 0.100 | 16.7× | 12.8M | SINGLE_SCRIPT on first 3 checks | 10 unique, avg 6.5. Extreme probe. Running when stopped. |
| PPO_57b | 0.020 | 3.33× | 22.4M | SINGLE_SCRIPT ~14-23 pts | 12-13 unique, avg 11-15, **best 31** (project single-game record). From PPO_57 source. |

**Finding (CRITICAL):** No entropy coefficient tested (0.01 through 0.10) prevents argmax collapse. The hypothesis that "sufficient entropy prevents memorization" is FALSIFIED. Entropy at 0.02-0.025 delays the collapse slightly and maintains healthier det=False diversity, but the argmax still concentrates. PPO_55e at 0.10 (16.7×) collapsed by the first post-entropy check at 10.6M — 3 checks, all SINGLE_SCRIPT.

**The INCOMPLETE false positive:** PPO_55b's 18+ consecutive INCOMPLETE det=True verdicts (which prompted the "no functional deterministic policy" claim) were caused by `make_check_env` missing `EpisodicLifeEnv` and `AutoResetWrapper`. The callback's custom autoreset logic never detected game-over, so 0 games completed within `max_check_steps`. After the fix, det=True completes on every check. The INCOMPLETE signal was an infrastructure bug, not a policy property.

#### Mid-Flight Teleport (Early ALE Return — Killed/Superseded)

| Model | Approach | Final Step | Outcome |
|-------|----------|------------|---------|
| PPO_44 | Paddle-bounce teleport 10% | 38.4M | Dead. mean=0.0. Killed. |
| PPO_45 | Paddle-bounce teleport 15% | 48M | Dead. mean=0.4. Full run. |
| PPO_46 | Paddle-bounce teleport 20% | 41.6M | Dead. mean=0.0. Killed. |
| PPO_47 | Mid-flight teleport 60% | 9.6M | Killed early — superseded by Y-perturb |
| PPO_48 | Mid-flight teleport 80% | 3.2M | Killed early — superseded by Y-perturb |
| PPO_49 | Mid-flight teleport variant | 3.2M | Killed early |
| PPO_50 | Mid-flight teleport variant | 3.2M | Killed early |

### Confirmed Memorized (Historical — Nosticky Verification)

| Model | Training | Nosticky result | Score |
|-------|----------|----------------|-------|
| PPO_25 | ALE, no sticky, 1B steps | Multiple scripts via eval cycling | ~20-50 |
| PPO_26 | ALE, PPO_25 pretrain + p=0.25 sticky | 60.0 pts × 500 games, 264 frames | 60 |
| PPO_27 | ALE, p=0.25 sticky from scratch | 100% zero scores, noise-dependent | 0 |
| PPO_28 | ALE, sticky removed from trained | Collapsed to fixed sequence | varies |
| PPO_29 | ALE, sticky removed from trained | Collapsed to fixed sequence | varies |
| PPO_30b | ALE, 100M non-sticky → 300M sticky | 99.8% zeros, 2 unique | 0 |
| PPO_31b | ALE, 300M non-sticky → 100M sticky | All 31-point script, 178 frames | 31 |

### Custom Engine (Historical — Does Not Transfer to ALE)

| Model | Approach | GymBreakout | ALE |
|-------|----------|------------|-----|
| PPO_34 | Per-episode physics randomization | 1 unique, 89 pts det=True | — |
| PPO_35 | Continuous mid-game physics | 1 unique, 212 pts det=True | 1 unique, 2 pts |
| PPO_36 | Ball noise σ=0.3 + dropout | 23 unique det=False at peak | — |

---

## What We've Learned

### This experimental cycle (July 20-23, 2026)

1. **Y-perturb via setRAM works technically.** Writing to RAM address 101 (ball Y) is reliable on ALE v0.11. The wrapper with cooldown mechanism is stable across billions of training steps.

2. **10% perturbation is enough for det=False diversity, not enough for det=True reactivity.** The argmax finds a script that works on the 90% of frames where the ball isn't perturbed. The policy entropy produces diverse scores under sampling, but the mode (argmax) concentrates on a fixed sequence.

3. **Entropy is not the lever.** Every entropy coefficient from 0.006 to 0.10 produces the same outcome: argmax collapses to a script. Entropy widens the distribution but doesn't shift the mode. PPO's optimizer always finds the argmax that maximizes expected return, and that argmax is always a script when scripts are viable.

4. **Infrastructure bugs compound quickly.** The env mismatch (missing EpisodicLifeEnv) produced 18+ consecutive INCOMPLETE verdicts that were interpreted as a breakthrough ("no functional deterministic policy"). The resume-logic bug caused entropy variants to silently restart from 9.6M on every relaunch. The score accumulation bug reported per-life scores instead of per-game. All three bugs were active simultaneously, and the INCOMPLETE interpretation drove experimental decisions for 3 days.

5. **Run-to-run variance matters.** PPO_55, 57, and 58 — identical configs, different random seeds — produced meaningfully different score trajectories. Single-replicate conclusions are unreliable.

### The hard way (historical, still valid)

1. **Sticky actions don't work.** They mask memorization with noise; they don't prevent it.

2. **Non-sticky pretraining causes permanent memorization.** Once a model memorizes during deterministic training, sticky fine-tuning adds noise but doesn't cure it.

3. **Score diversity is not reactivity.** Dead scripts produce diverse scores under stochastic sampling. The only reliable test is det=True nosticky verification.

4. **Every new metric needs dead-model calibration.** This happened with the intervention test, the shape classifier, and the MemorizationCheckCallback. The INCOMPLETE false positive is the same pattern in a new form: an anomalous metric was interpreted as evidence of reactivity before the infrastructure producing it was verified.

5. **The custom engine doesn't approximate ALE.** PPO_35: 212 pts → 2 pts. All custom-engine findings need ALE replication.

---

## What's Next

### Immediate: Higher Perturbation Probability

The central question after this cycle: at what perturbation probability do scripts become non-viable?

**Design (presented for approval):**
- Two fresh runs: **prob=0.25** (2.5× current) and **prob=0.50** (5× current)
- Keep cooldown=30, ±8px range, ent_coef=0.006 — vary only probability
- Fresh starts (not from checkpoint) — need full trajectories
- Target 50M steps each
- At 10%, a 60-point script (~264 frames) experiences ~2-3 perturbations. At 25%, ~5-6. At 50%, ~8-9. At some point the script's timed paddle positions are wrong too often.

**Success criterion:** det=True shows MULTIPLE_SCRIPTS (3+ unique) at 50M steps on any replicate.

### If Higher Probability Also Fails

If even prob=0.50 produces argmax scripts, the next step is multi-parameter perturbation — randomize both X and Y simultaneously (currently Y-only). A ±8px X perturbation would require the script to track ball X position, not just Y timing. Combined with Y-perturb, this makes the script problem 2D rather than 1D.

### If That Also Fails

The fallback is to accept that PPO + NatureCNN on Breakout converges to scripts regardless of perturbation, and the real solution requires either:
- A different architecture (e.g., transformer with attention over frames)
- A different objective (e.g., inverse dynamics prediction as auxiliary loss)
- A different game (Breakout may be structurally too simple to force reactive policies)

### For New Sessions

See `CURRENT_STATE.md` (this file) first — then:
1. `EXPERIMENTS.md` — full experiment history including Experiment 4b and 4c
2. `LOGICAL_AUDIT.md` — now 18 entries including L-017 (env mismatch false positive) and L-018 (entropy hypothesis falsification)
3. `FLAWS.md` — 23 entries (F-022: env mismatch, F-023: missing resume logic)
4. `CLAUDE.md` — critical rules, conventions, diagnostic checklist

The Session Bootstrap in `CLAUDE.md` has the step-by-step procedure.
