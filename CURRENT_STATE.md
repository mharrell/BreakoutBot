# Current State — BreakoutBot

**Last updated: 2026-07-27 (Dose-response curve complete + Experiment 8 reward structure)**

---

## TL;DR

**Perception POC (July 26): NatureCNN CAN track the ball to 1.9px MAE (0.6px median).** The architecture has the perceptual capability. PPO_85 proved that even with those perfect ball-position features frozen in, PPO converges to a 0pt blind script. The SINGLE_SCRIPT problem is in the optimization, not the perception.

**Dose-response curve (July 27): Teleport magnitude changes which script gets memorized, not whether.** ±30px produces a 21pt script (best in the project for dynamics). ±35px is marginal (0-5pt). ±40px+ is dead (0pt). But every single setting produces SINGLE_SCRIPT on clean ALE. Teleports change the script quality, not the memorization outcome.

**Experiment 8 (LAUNCHING): Reward the behavior directly.** If the Atari score signal is the attractor pulling PPO toward blind scripts, then we need to make ball-tracking *more rewarding than the Atari score*. Four variants: ball-hit reward at 1.0 and 2.0, descending-only proximity, and combined (hit + proximity + survival penalty). All train on clean ALE — no teleports, no noise. The goal is to shift the optimization landscape so reactive policies occupy a higher local optimum than blind scripts.

After 91 experiments, no method has broken SINGLE_SCRIPT.

---

## Terminology

| Term | Definition |
|------|-----------|
| **SINGLE_SCRIPT** | ≤2 unique scores on det=True across n_games. The argmax produces the same score every game. |
| **MULTIPLE_SCRIPTS** | 3+ unique scores on det=True. Not yet observed on clean ALE in 91 experiments. |
| **Memorized** | SINGLE_SCRIPT on clean ALE det=True. Does NOT imply 0pt — a 21pt script is still memorized. |
| **Dead** | 0pt exactly on clean ALE det=True. A subset of memorized. |
| **Blind script** | A memorized policy with no evidence of ball-tracking. Synonymous with "memorized script." |
| **Stoch** | det=False (stochastic sampling). Produces score diversity even in memorized policies. |
| **Collapse** | Transition from MULTIPLE_SCRIPTS to SINGLE_SCRIPT, or from nonzero to 0pt. |

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
| **NatureCNN CAN track the ball — perception is not the bottleneck** | Perception POC: 4-frame NatureCNN predicts ball position to 1.9px MAE (0.6px median, 1.7px X, 0.5px Y). 71% improvement over single-frame (6.6px). The conv features encode ball position with near-perfect precision. PPO never learns to use them. |
| **Every method added to OpticalFlow reduces performance** | OF solo (44pt) > every OF+1 combo (20-28pt) > every OF+2 combo (9-22pt). OF+YP+RS+HE with ent=0.02 flatlined at 0pt. The pattern is monotonic. |
| **Dynamics randomization via setRAM() does not transfer to ALE** | PPO_78 (mild teleports), PPO_79 (σ=0.5 noise), PPO_80 (σ=1.5 noise) — all SINGLE_SCRIPT on clean ALE. The mechanism that produced intervention-robust policies on the custom engine does not work on authentic Atari. |
| **Dose-response curve: teleport magnitude changes script quality, not memorization outcome** | PPO_78 (±8px)=0pt, PPO_90 (±30px)=13pt peak, PPO_81 (±30px)=21pt peak at 50M, PPO_91 (±35px)=0-5pt marginal, PPO_89 (±40px)=0pt dead, PPO_87/88 (±45px)=0pt dead. Every setting SINGLE_SCRIPT. ±30px is the sweet spot but still memorized. |
| **Frozen pretrained ball-tracker features don't prevent collapse** | PPO_85 (conv+linear frozen from 1.9px MAE BallTrackerCNN4Frame): collapsed to 0pt SINGLE_SCRIPT by 6M (first zero at 6M, confirmed through 25M). PPO_86 (conv frozen, linear trainable): killed at 1M (0pt). The optimizer actively avoids using available ball-position features. |

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
| PPO_55 | default | 50M | 50M | SINGLE_SCRIPT ~15 pts | 10 unique, avg 9.5, best 17 | First Y-only. det=False peak at 10M: 9 unique, avg 11.6, best 16 |
| PPO_57 | 57 | 50M | 50M | SINGLE_SCRIPT | 12 unique, avg ~14, best ~24 | Stronger det=False than PPO_55. Confirmed 10M peak. |
| PPO_58 | 58 | 50M | 50M | SINGLE_SCRIPT ~11-13 pts | 12 unique, avg ~11, best ~24 | Third replicate. Classic pattern. |

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

### Perception POC (July 26, 2026)

1. **NatureCNN can locate the ball from pixels with near-perfect precision.** 4-frame stacked NatureCNN trained via supervised regression achieves 1.9px MAE (0.6px median). The conv features encode ball position — the architecture can "see" the ball. PPO never learns to attend to these features.

2. **Temporal context is critical for horizontal localization.** Single-frame: 6.2px X error. 4-frame: 1.7px X error. Motion information is how the CNN disambiguates the 2-3px ball from visual clutter. The 4-frame stack the RL agent uses is sufficient.

3. **The SINGLE_SCRIPT problem is an RL optimization bottleneck, not a perception bottleneck.** This is the most important finding of the project to date. After 91 experiments, we now know the CNN can represent ball position perfectly but PPO never discovers this signal. The reward gradient from "paddle swept past ball → scored a point" is too sparse and delayed to compete with the local optimum of "paddle sweeps back and forth → occasionally hits ball → gets some reward."

### Combination Matrix (July 25-26, 2026)

4. **OpticalFlow accelerates memorization, doesn't prevent it.** OF solo (PPO_66) reached 44pt stoch best — the fastest convergence in the project. But it's still SINGLE_SCRIPT. OF speeds up the same trajectory, doesn't change the destination.

5. **Adding methods to OF is strictly subtractive.** OF solo > OF+1 > OF+2 > OF+4. Each additional method adds noise that dilutes the gradient without preventing memorization.

6. **Dynamics randomization via setRAM() failed at three settings.** Mild teleports (PPO_78), moderate per-frame noise (PPO_79), strong per-frame noise (PPO_80) — all SINGLE_SCRIPT on clean ALE. Per-frame noise teaches noise-ignoring, not ball-tracking.

7. **Survival mode is the universal failure mode for multi-method combos.** When scripts are prevented but reactivity isn't taught, the policy defaults to "keep paddle moving, don't die" — long episodes (3000-9000f), low scores (15-30pt).

### This experimental cycle (July 20-23, 2026) — Y-Perturb + Entropy

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

### Running Now (1 GPU, July 27)

| Slot | Run | Recipe | Progress |
|------|-----|--------|----------|
| 1 | RBO_02 | OF + Dropout (Revenge Brunch) | ~109M / 1B |
| 2-5 | PPO_92-95 | Experiment 8: Ball-tracking reward | QUEUED |

PPO_78/79/80/81/85/86/87/88/89/90/91 — all stopped. Results documented above.

### Experiment 8: Ball-Tracking Reward (SCRIPTS READY)

The perception POC proved NatureCNN can track the ball perfectly (1.9px MAE). PPO_85 proved that even with those features frozen in, the optimizer converges to a blind script. The Atari score signal is the dominant optimization attractor — it rewards brick-breaking, not ball-tracking. Experiment 8 tries to change what the optimizer optimizes for.

All four train on **clean ALE** (no teleports, no noise). Eval/check also clean.

| Run | Mode | Reward | Hypothesis |
|-----|------|--------|------------|
| **PPO_92** | Ball-hit only | +1.0/hit | Hit reward = brick reward → tracking enters gradient equally |
| **PPO_93** | Ball-hit double | +2.0/hit | Hit reward > brick reward → tracking dominates optimization |
| **PPO_94** | Descending proximity | 0.005/frame (descent-gated) | Only reward approach phase — cleaner gradient than un-gated |
| **PPO_95** | Combined | hit=1.0 + prox=0.005 + survival=-0.0001 | All three signals for strongest possible tracking gradient |

**Key design differences from PPO_15 (failed):**
- Pixel observations, not RAM (policy must extract ball position from pixels)
- 1/20th to 1/5th the scale (PPO_15 used 0.1; we use 0.005 proximity)
- Hit detection via state machine, not frame-by-frame proximity
- Descent-gated proximity — only during approach, not ascent

### Experiment 5 Complete: Discrete Teleport Dose-Response

Full dose-response curve mapped across 7 runs:

| Run | Magnitude | Peak det | Verdict |
|-----|-----------|----------|---------|
| PPO_78 | ±8px | 10pt | SINGLE_SCRIPT |
| PPO_81 | ±30px | 21pt at 50M | SINGLE_SCRIPT — best dynamics result |
| PPO_90 | ±30px (rep) | 13pt at 5M | SINGLE_SCRIPT — confirms ±30px sweet spot |
| PPO_91 | ±35px | 0-5pt | SINGLE_SCRIPT — marginal, cliff zone (11M) |
| PPO_89 | ±40px | 0pt | SINGLE_SCRIPT — dead by 5M (6M) |
| PPO_87 | ±45px | 0pt | SINGLE_SCRIPT — dead, stoch collapsed (40M) |
| PPO_88 | ±45px (rep) | 0pt | SINGLE_SCRIPT — confirms ±45px dead (35M) |

**Conclusion:** Teleport magnitude changes which script gets memorized and how many points it scores. It does not change the outcome: every setting produces SINGLE_SCRIPT on clean ALE. The sweet spot at ±30px produces the best scripts but never breaks memorization.

### For New Sessions

See `CURRENT_STATE.md` (this file) first — then:
1. `EXPERIMENTS.md` — full experiment history including Experiment 4b and 4c
2. `LOGICAL_AUDIT.md` — now 17 entries including L-017 (env mismatch false positive)
3. `FLAWS.md` — 23 entries (F-022: env mismatch, F-023: missing resume logic)
4. `CLAUDE.md` — critical rules, conventions, diagnostic checklist

The Session Bootstrap in `CLAUDE.md` has the step-by-step procedure.
