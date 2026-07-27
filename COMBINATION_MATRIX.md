# Combination Matrix — Anti-Memorization Method Results

**Date:** 2026-07-26
**Environment:** ALE/Breakout-v5 (authentic Atari)
**Standard hyperparams:** n_envs=32, batch_size=1024, n_steps=128, n_epochs=4, gamma=0.99, lr=2.5e-4→1e-5, clip=0.2→0.05, ent_coef=0.006 (unless noted), vf_coef=0.5, NatureCNN

---

## Method Catalog

| Code | Method | Type | Mechanism |
|------|--------|------|-----------|
| **OF** | OpticalFlow | Architectural | 2-channel [current, \|diff\|] replaces 4-frame stacking. No temporal infrastructure for scripts. |
| **YP** | Y-Perturb | Dynamics | Ball Y teleported ±8px mid-flight (prob=0.25, cooldown=30). Changes arrival timing. |
| **RS** | RandomShiftObs | Perceptual | ±4px spatial offset per episode. Kills pixel-level anchoring. |
| **HE** | High Entropy | Optimization | ent_coef=0.02 (3.3× standard). Prevents argmax from collapsing to single action. |
| **Dropout** | Dropout (p=0.1) | Regularization | Dropout in NatureCNN feature layer. Prevents feature co-adaptation. |

---

## Solo Baselines

| Run | Method | Steps | det=True | Stoch unique | Stoch best | Verdict |
|-----|--------|-------|----------|-------------|------------|---------|
| PPO_66 | OF | 50M | 1u, 17pt | 16u | **44pt** | SINGLE_SCRIPT. Best non-sticky result in project. OF accelerates learning. |

*YP, RS, and HE were not run as solos on ALE. YP solo data exists from PPO_55-58 (different paradigm).*

---

## Two-Method Combinations (6 total)

| # | Run | Methods | Steps | det=True | Stoch unique | Stoch best | Status |
|---|-----|---------|-------|----------|-------------|------------|--------|
| 1 | PPO_67 | **OF+YP** | 50M | 1u, 11pt | 11u | 21pt | COMPLETED |
| 2 | PPO_69 | **OF+RS** | 50M | 1u, 8pt | 13u | 20pt | COMPLETED |
| 3 | PPO_70 | **OF+HE** | 50M | 1u, 12pt | 13u | 28pt | COMPLETED |
| 4 | PPO_71 | **YP+RS** | 18M | 1u, 5pt | **2u** | 6pt | KILLED — stoch SINGLE_SCRIPT |
| 5 | PPO_72 | **YP+HE** | 50M | 1u, 8pt | 14u | 22pt | COMPLETED |
| 6 | PPO_73 | **RS+HE** | 14M | 1u, 18pt | 8u | 15pt | KILLED |

**Key findings:**
- **OF is the core ingredient.** Every 2-method combo with OF reached 20-28pt stoch best. Combos without OF (YP+RS, YP+HE, RS+HE) topped out at 6-22pt and showed worse score diversity.
- **YP+RS was the worst combo.** Both det=True AND stoch collapsed to SINGLE_SCRIPT (2 unique scores) by 18M. First time stoch diversity collapsed in any experiment.
- **OF+HE had the best peak** (68pt at 12M) but regressed to 28pt. The spike was a one-off.
- **Adding a second method to OF always REDUCES performance.** OF solo: 44pt stoch. OF+YP: 21pt. OF+RS: 20pt. OF+HE: 28pt. Every combination scored less than OF alone.

---

## Three-Method Combinations (4 total)

| # | Run | Methods | Steps | det=True | Stoch unique | Stoch best | Status |
|---|-----|---------|-------|----------|-------------|------------|--------|
| 7 | PPO_74 | **OF+YP+RS** | 12M | 1u, 5pt | 4u | 9pt | KILLED |
| 8 | PPO_75 | **OF+YP+HE** | 6M | 1u, 0pt | — | — | KILLED |
| 9 | PPO_76 | **OF+RS+HE** | — | — | — | — | NEVER STARTED |
| 10 | PPO_77 | **YP+RS+HE** | — | — | — | — | NEVER STARTED |

---

## Four-Method (Kitchen Sink)

| Run | Methods | Steps | det=True | Stoch unique | Stoch best | Status |
|-----|---------|-------|----------|-------------|------------|--------|
| PPO_68 | OF+YP+RS+HE (ent=0.02) | 12.8M | 1u, 0pt | 3u | 2pt | KILLED — flatlined, ent too high |
| PPO_68b | OF+YP+RS+HE (ent=0.006) | 60.2M | 1u, 20pt | 11u | 22pt | KILLED — plateaued 30M, survival mode |

**Key findings:**
- **High entropy kills learning with 3+ methods.** PPO_68 (ent=0.02) flatlined at 0-2pt. PPO_68b (ent=0.006) learned but slowly.
- **Survival-mode failure mode:** 6,000+ frame games scoring only 20 points. The policy learns to avoid dying rather than optimize scoring. All 4 methods together break script formation but don't teach ball-tracking — the gradient signal can't distinguish reactive play from conservative paddle-wiggling.

---

## Dynamics Randomization — Discrete Teleport Dose-Response (Complete)

**All runs: ALE/Breakout-v5, training with teleports, eval/check on clean ALE.**
**Standard cadence (PPO_81 family): prob=0.04, cooldown=45, ball_noise_std=0.0.**
**Extreme cadence (PPO_87 family): prob=0.06, cooldown=30.**

| Run | Y Range | X Range | Cadence | Final det | Final stoch | Steps | Verdict |
|-----|---------|---------|---------|-----------|-------------|-------|---------|
| PPO_78 | ±8px | ±6px | prob=0.01, cd=60 | 10pt | 23pt | 50M | SINGLE_SCRIPT every check |
| PPO_81 | ±30px | ±20px | standard | **21pt** | 27pt | 50M | SINGLE_SCRIPT, best dynamics result |
| PPO_90 | ±30px | ±20px | standard (rep, seed=90) | 13pt | 18pt | 15M | SINGLE_SCRIPT, confirms ±30px |
| PPO_91 | ±35px | ±25px | standard | 5pt | 7pt | 11M | SINGLE_SCRIPT, marginal/cliff |
| PPO_89 | ±40px | ±25px | standard | 0pt | 0pt | 6M | SINGLE_SCRIPT, dead at 5M |
| PPO_87 | ±45px | ±30px | extreme (seed=default) | 0pt | 0pt | 40M | SINGLE_SCRIPT, stoch collapsed |
| PPO_88 | ±45px | ±30px | extreme (seed=88) | 0pt | 0pt | 35M | SINGLE_SCRIPT, confirms ±45px dead |

**Cross-cutting finding:** Teleport magnitude changes WHICH script gets memorized and HOW MANY points it scores — but it never changes the outcome. Every single run, at every single setting, produces SINGLE_SCRIPT on clean ALE. The dose-response is ±30px peak → cliff at ±35px → dead at ±40px+. This is a script-quality curve, not a reactivity curve.

### Per-Frame Noise (Failed)

| Run | Settings | Steps | det=True | Stoch best | Status |
|-----|----------|-------|----------|------------|--------|
| PPO_79 | Teleports + σ=0.5 noise | 16M | SINGLE_SCRIPT, 5pt | 3pt | KILLED — noise-exploiting |
| PPO_80 | σ=1.5 noise only, no teleports | 6M | SINGLE_SCRIPT, 0pt | 0pt | KILLED — worse than teleports |

---

## Revenge Brunch (Score Maximization)

| Run | Recipe | Steps | det=True | Stoch best | Status |
|-----|--------|-------|----------|------------|--------|
| RBO_01 | Dropout + pretraining (4-frame) | 199M | 1u, 25pt | 27pt | KILLED — survival mode at 3000f/game |
| RBO_02 | OF + Dropout (2-channel) | ~80M/1B | 1u, 16pt | 17pt | RUNNING |

**Key findings:**
- **RBO_01 reached the same scores as PPO_70 at 6× the steps.** 199M steps of dropout+pretraining = the same 25-27pt as 33M of OF+HE. Dropout prevents entropy collapse but doesn't accelerate learning.
- **OF is the accelerant.** RBO_02 (OF+Dropout) runs at 1450 fps vs RBO_01's 1058 fps — 40% faster throughput from the 2-channel input alone. Final verdict pending at 1B steps.

---

## Cross-Cutting Patterns

### 1. Everything is SINGLE_SCRIPT on clean ALE
Every model in every configuration across 91 experiments produces exactly one argmax score on clean ALE. No method or combination has broken this pattern. Not OF, not YP, not RS, not HE, not Dropout, not dynamics randomization at any teleport magnitude, not auxiliary tracking rewards, not frozen pretrained ball-tracker features. The optimizer always finds a deterministic script.

### 1b. The perception-policy gap is the central problem
NatureCNN CAN encode ball position to 1.9px MAE (proven). A policy with those exact features frozen in still collapses to a 0pt blind script (PPO_85, proven). The optimizer actively avoids using ball-position information that the features provably encode. This is not a perception bottleneck — it's an optimization attractor. The Atari score gradient rewards brick-breaking, and blind paddle-sweeping scripts are a viable local optimum for brick-breaking.

### 2. OpticalFlow is a memorization accelerant, not an anti-memorization method
OF consistently produces faster convergence and higher peak scores than standard 4-frame stacking. But it accelerates the *same* memorization trajectory — the destination is identical, just reached faster. Compare PPO_66 (OF solo, 44pt at 50M) vs PPO_26 (4-frame, 60pt at 838M) — different scale but same curve shape.

### 3. Adding methods subtracts from performance
OF solo > any OF+1 combo > any OF+2 combo (kitchen sink). Each additional method imposes a learning tax: more noise → harder credit assignment → lower scores. The gradient signal gets diluted across more sources of variance.

### 4. Non-OF combos are worse than OF alone
YP+RS, YP+HE, RS+HE all underperformed OF solo. Without the architectural change of 2-channel input, the CNN sticks with 4-frame temporal pattern matching regardless of perturbations.

### 5. Dynamics randomization hasn't worked on ALE yet
The mechanism that produced intervention-robust policies on the custom engine (PPO_35) has not transferred to ALE at any setting tested so far. PPO_78 (mild teleports) and PPO_79 (mild noise+teleports) both produced SINGLE_SCRIPT on clean ALE. PPO_80 (σ=1.5 pure noise) is the last attempt at this hypothesis.

### 6. Survival mode is the universal failure mode
When scripts are prevented but reactivity isn't taught, the policy defaults to "keep paddle moving, don't die." Long episodes (3000-9000 frames), low scores (15-30 points). Seen in PPO_68b, RBO_01, and PPO_79 (noisy env). The reward gradient rewards survival more strongly than scoring efficiency.

---

---

## Perception POC (July 26, 2026) — Foundational Finding

**NatureCNN can track the ball to 1.9px MAE (0.6px median) from 4-frame stacked input.** See `perception_poc.py` and `perception_poc_4frame.py`.

| Metric | 1-Frame | 4-Frame | Improvement |
|--------|---------|---------|-------------|
| MAE (overall) | 6.6px | **1.9px** | 71% |
| X (horizontal) | 6.2px | **1.7px** | 73% |
| Y (vertical) | 0.8px | **0.5px** | 38% |
| P50 (median) | -- | **0.6px** | sub-pixel |

**Implication:** The conv features encode ball position with near-perfect precision. PPO never learns to attend to them. The SINGLE_SCRIPT problem is an RL optimization bottleneck, not a perception bottleneck. This rules out "the CNN can't see the ball" as an explanation for 91 failed experiments.

---

## Experiment 6: Auxiliary Ball-Tracking Reward (COMPLETED — All Dead)

PPO_15 (RAM obs, scale=0.1) tried this and failed: agent mirrored the ball. Exp 6 used pixel observations at 1/20th scale. All three variants dead by 6-12M.

| Run | Mode | Scale | Steps | det=True | Verdict |
|-----|------|-------|-------|----------|---------|
| PPO_82 | Proximity (continuous) | 0.005 | 6M | 0pt | SINGLE_SCRIPT, killed |
| PPO_83 | Coarse (binary ±40px) | 0.005 | 6M | 1-2pt | SINGLE_SCRIPT, killed |
| PPO_84 | Annealing (0.02→0.0 over 25M) | 0.02→0.0 | 12M | 3-5pt | SINGLE_SCRIPT, killed |

Auxiliary proximity reward at small scale does not break memorization. The 0.005 scale is too weak to compete with the Atari score gradient.

## Experiment 7: Frozen Pretrained Backbone (COMPLETED — Both Dead)

Loaded BallTrackerCNN4Frame weights (1.9px MAE) as frozen conv+linear layers for PPO. Policy head starts with "where's the ball?" pre-solved.

| Run | Freeze Config | Steps | det=True | Verdict |
|-----|--------------|-------|----------|---------|
| PPO_85 | Conv + Linear frozen | 25M | 0pt | SINGLE_SCRIPT — collapsed at 22-25M |
| PPO_86 | Conv frozen, Linear trainable | 1M | 0pt | SINGLE_SCRIPT — killed very early |

**Critical finding:** A policy with provably perfect ball-position features in its frozen backbone still converges to a 0pt blind script. The optimizer actively avoids using the ball-position information that the features encode. This is the definitive proof that the SINGLE_SCRIPT problem is an optimization attractor, not a perception deficit.

---

## Experiment 8: Ball-Tracking Reward (July 27, 2026 — QUEUED)

The perception POC proved NatureCNN can track the ball (1.9px MAE). PPO_85 proved that even frozen pretrained ball-tracker features collapse to 0pt. The Atari score signal is the dominant attractor. Experiment 8 changes the reward structure to make ball-tracking more rewarding than brick-breaking.

All four train on **clean ALE** (no teleports, no noise). Eval/check also clean.

| Run | Mode | Signal | Hypothesis |
|-----|------|--------|------------|
| **PPO_92** | hit_only | +1.0 per paddle-ball contact | Hit reward = brick reward → tracking enters gradient |
| **PPO_93** | hit_double | +2.0 per paddle-ball contact | 2× hit makes tracking MORE rewarding than scoring |
| **PPO_94** | descending_proximity | 0.005 × (1 − distance/40), descent-gated | Gating on descent creates cleaner gradient than un-gated |
| **PPO_95** | combined | hit=1.0 + prox=0.005 + survival=−0.0001 | Three signals for strongest tracking gradient |

Hit detection: state machine (IDLE→PENDING→HIT→COOLDOWN) via RAM. Ball enters paddle zone descending, bounces up near paddle X → hit. 15-frame cooldown.

## Running State (2026-07-27)

| Slot | Run | Methods | Progress | Status |
|------|-----|---------|----------|--------|
| 1 | RBO_02 | OF + Dropout | ~80M / 1B | Running |
| 2 | PPO_92 | Ball-hit reward 1.0 | 0 / 50M | QUEUED |
| 3 | PPO_93 | Ball-hit reward 2.0 | 0 / 50M | QUEUED |
| 4 | PPO_94 | Descending proximity | 0 / 50M | QUEUED |
| 5 | PPO_95 | Combined | 0 / 50M | QUEUED |

**Killed/Completed:** PPO_78-91 (all SINGLE_SCRIPT), PPO_82-86 (aux reward + frozen backbone, all SINGLE_SCRIPT).

---

*This document supersedes individual experiment notes. All scores from `recordings/*_memorization_track.csv` final entries.*
