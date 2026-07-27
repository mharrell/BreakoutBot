# Revenge Brunch — Superhuman Breakout via Unapologetic Memorization

**Project started:** 2026-07-23
**Purpose:** Achieve superhuman Breakout scores by embracing what works — deep non-sticky pretraining + sticky actions — without the burden of proving reactivity.

---

## Philosophy

BreakoutBot's main line has chased reactive ball-tracking for 91 PPO runs across two engines. Every model ever tested in this project collapses to a memorized script on det=True. Sticky actions mask the memorization with noise but don't cure it. Entropy doesn't prevent it. Dynamics randomization doesn't prevent it.

Revenge Brunch takes the opposite approach: **if memorization is inevitable, let's do it brilliantly.** The goal is not to prove the policy tracks the ball. The goal is a script so good, and so robust to sticky noise, that it outscores every reactive policy ever trained on Breakout.

The project's own data shows this is achievable:
- PPO_26: 838M non-sticky pretraining → 60-pt script. With sticky noise: **avg 54.3, best 415, 0% zero-score.**
- The secret isn't a better algorithm. It's a deeper pretraining phase.

## What We Know (The Recipe)

| Ingredient | Evidence | Source |
|------------|----------|--------|
| Script quality ∝ non-sticky pretraining depth | 838M→60 pts, 300M→31 pts, 100M→0 pts | PPO_26/31b/30b |
| Sticky noise at inference unlocks scripts | 60-pt script + p=0.25 → avg 54.3, best 415 | PPO_26 funnel |
| Dropout prevents entropy collapse | PPO_36 (dropout) vs PPO_37 (no dropout) | Custom engine |
| Conservative LR at phase switch prevents metric collapse | 1e-4→1e-5 vs 2.5e-4→1e-5 | PPO_30b/31b vs 28/29 |
| EpisodicLifeEnv trains life-loss recovery | Standard in all recent runs | — |
| Clean deterministic env = easier script discovery | No perturbation to confuse the script | Obvious |

## Design

### Phase 1: Deep Non-Sticky Pretraining (RBO_01)

- **Environment:** Clean ALE/Breakout-v5, no perturbation, no sticky actions
- **Architecture:** NatureCNN with dropout (p=0.1) in the feature layer
- **Steps:** 500M (extend if scores still climbing)
- **LR:** 2.5e-4 → 1e-5 (linear)
- **Clip:** 0.2 → 0.05 (linear)
- **ent_coef:** 0.006
- **n_envs:** 32, batch_size=1024, n_steps=128, n_epochs=4
- **frameskip=1** training for maximum temporal precision

### Phase 2: Sticky Fine-Tuning (RBO_02)

- **Environment:** Same, but `repeat_action_probability=0.25`
- **Start from:** RBO_01 best_model
- **LR restart:** 1e-4 → 1e-5 (conservative)
- **Steps:** 500M
- **Everything else:** unchanged

### Evaluation

- **det=True, sticky=off:** The script quality metric. Higher = better memorized sequence.
- **det=False, sticky=on:** The deployed performance metric. This is what the world sees.
- **10k-game funnel:** Gold standard at Phase 2 completion. Compare against PPO_26.

## Agent Roster

| Agent | Phase | Steps | Status | Script Score | Sticky Avg | Best | Zero% |
|-------|-------|-------|--------|-------------|------------|------|-------|
| RBO_01 | Phase 1 (no sticky) | 0/500M | **TRAINING** | — | — | — | — |
| RBO_02 | Phase 2 (sticky) | — | PENDING | — | — | — | — |

## Targets

| Metric | PPO_26 (current record) | Revenge Brunch target |
|--------|------------------------|----------------------|
| Script score (det=True, no sticky) | 60 pts | **100+ pts** |
| Sticky avg (10k games) | 54.3 | **70+** |
| Sticky best | 415 | **500+** |
| Zero-score rate (sticky) | 0.0% | **0.0%** |
| Funnel rate (400+) | 0.07% | **0.10%+** |

## Why "Revenge Brunch"

Because sometimes you spend months trying to do the right thing, and the universe keeps telling you "no." So you sit down, order brunch, and do what actually works instead.

---

*This document is independent of the main BreakoutBot experimental line. Revenge Brunch does not claim to produce reactive, generalizing policies. It claims to produce the highest-scoring Breakout agent possible, by any means the architecture allows.*
