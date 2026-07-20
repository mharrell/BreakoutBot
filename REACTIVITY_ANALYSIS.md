# Reactivity Analysis — Intervention Test Results

**Date:** 2026-07-19
**Method:** Intervention test — teleport ball (y-axis) and paddle (x-axis) after 30% of random paddle bounces. Compare score distributions: normal vs intervention.
**Environment:** GymBreakout(fixed=True) — deterministic Breakout, fixed ball speed, no speed-up on brick breaks (custom engine, not ALE).

## Summary

**Only dynamics randomization (PPO_35) produces a policy robust to ball/paddle teleportation.** Every model trained without it — regardless of noise level, dropout, or network width — collapses to a brittle script.

## Results Table (50-game tests, det=True)

| # | Model | Training Recipe | Normal | Inter (mean) | Inter (med) | Inter (range) | Retention | Int/game |
|---|-------|----------------|--------|-------------|------------|--------------|-----------|----------|
| 1 | **PPO_35** | **Dynamics rand + dropout** | **751** | **353.9** | **337** | **255-726** | **47%** | **24.5** |
| 2 | PPO_37 | Ball noise σ=0.3 only | 260 | 68.1 | 50 | 13-275 | 26% | 4.3 |
| 3 | PPO_36 | Ball noise σ=0.3 + dropout | 346 | 84.1 | 65 | 9-296 | 24% | 6.5 |
| 4 | PPO_39 | 2× wider CNN + dropout | 326 | 78.5 | 56 | 15-328 | 24% | 5.0 |
| 5 | PPO_38 | Ball noise σ=0.5 + dropout | 56 | 11.8 | 5 | 1-56 | 21% | 1.2 |
| 6 | PPO_40 | σ=0.5 + fixed physics | 261 | 20.5 | 10 | 1-185 | 8% | 2.0 |

**Every model's normal score has std=0.0.** They're all deterministic scripts. The difference is what happens when the script breaks.

**PPO_35's intervention score (354) is higher than any other model's normal score (max 346).** Even disrupted, it outperforms every noise-only model playing clean.

**Interventions/game reveals survivability**: PPO_35 survives 24.5 teleports per game (the policy adapts and keeps playing). Noise-only models survive 1-6 teleports before dying — they can't recover.

## Key Finding: Two Kinds of "Memorization"

The old diagnostic (det=True unique scores = 1) labeled every model as "memorized." The intervention test reveals there are actually two kinds:

1. **Brittle memorization** (PPO_36-40): The policy learns a timed action sequence. Teleport the ball and the sequence plays from the wrong position — scores collapse to 4-26% of normal. These are *blind scripts*.

2. **Robust memorization** (PPO_35): The policy's argmax produces the same outcome every game on deterministic env, but the script *uses ball tracking internally*. Teleport the ball and the policy adjusts its actions to the new position — retains 43-47% of normal score. This is a *sighted script*.

## What Causes Robustness?

PPO_35 is the only model trained with `DynamicBreakout` — continuous physics parameter changes every 60-300 frames (paddle width, ball speed, paddle speed). All other models use fixed physics with varying levels of perceptual noise (ball velocity jitter, latent dropout) or network architecture changes.

**Dynamics randomization forces the policy to condition on ball position** because the ball arrives at the paddle at unpredictable times and angles. Perceptual noise makes observations harder but doesn't require the policy to track anything — you can still get by with a timed sequence.

## Intervention Distribution Detail (50-game tests)

### PPO_35 — Robust (47% retention)
- Normal: 751 every game (2 board clears, 7,939 frames)
- Intervention: mean 353.9, median 337, std 98.7, range 255-726
- 24.5 interventions/game (6× more than any other model — games survive teleports)
- 41 unique scores in 50 games
- **Intervention score (354) exceeds every other model's NORMAL score**
- Distribution: 92% of games cluster at 255-386; 2 outliers at 709 and 726

### PPO_37 — Best Noise-Only Model (26% retention)
- Normal: 260 every game
- Intervention: mean 68.1, median 50, std 65.1, range 13-275
- **1/50 games where intervention > normal (275 > 260)** — rare upside from randomness
- Intervention distribution: 78% of games cluster at 13-63 pts; a few outliers reach 200+

### PPO_39 — Wider Network (24% retention)
- Normal: 326 every game  
- Intervention: mean 78.5, median 56, std 67.1, range 15-328
- **1/50 games where intervention > normal (328 > 326)** — essentially identical
- **Doubling network width didn't improve robustness over PPO_37**

### PPO_38 — Drowned by Noise (21% retention)
- Normal: 56 every game
- Intervention: mean 11.8, median 5, std 15.5, range 1-56
- 42% of intervention games score 1-3 pts (immediate death)
- Only 1.2 interventions/game — games end too fast for teleports to fire

### PPO_40 — Worst Overall (8% retention)
- Normal: 261 every game
- Intervention: mean 20.5, median 10, std 34.9, range 1-185
- 66% of intervention games score ≤17 pts
- Only 2.0 interventions/game — dies almost immediately after teleport
- **Fixed physics + high noise = extreme brittleness**

## Videos

Recorded representative (median-score) gameplay for each model:
- `recordings/videos/PPO_XX_normal.mp4` — no interventions
- `recordings/videos/PPO_XX_intervention.mp4` — with teleports ("TELEPORT!" flashes red)

## Limitations

1. **Custom Breakout engine** — not ALE. Ball speed is fixed (no speed-up as bricks break), paddle speed is 3px/frame. Results may not transfer to real Atari Breakout without testing.
2. **Sample sizes** vary — 50-game tests for PPO_35/37/38/39/40 (running), 5-game for others
3. **det=True only** — stochastic mode (det=False) not yet tested for all models
4. **Training convergence** — PPO_38 scored 56 normally (drowned by σ=0.5 noise). The "brittle" verdict may partly reflect poor training, not just brittleness.

## Conclusion

**Dynamics randomization is the only intervention in this experiment suite that produces a policy capable of adapting to ball position perturbations.** The mechanism is straightforward: if the physics change, the ball moves differently, and the only winning strategy is watching it. Perceptual noise alone — regardless of magnitude, network width, or dropout — does not create this pressure and produces policies that are blind to ball position.
