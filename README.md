# BreakoutBot

**Not chasing high scores. Chasing reactivity.**

A reinforcement learning project investigating what forces a PPO agent to track the ball in Atari Breakout rather than memorize a fixed action sequence. Built with Stable-Baselines3/PyTorch on a single RTX 3060 Ti.

---

## Honest Status (July 2026)

**No model in this project has ever genuinely generalized.** After 43 PPO runs, every promising model was found on closer inspection to be a memorized script or a noise-masked dead policy. The project has documented 4 false positives where a promising number was interpreted as proof of reactivity before the falsification test was run.

The good news: we caught every false positive ourselves, built a rigorous diagnostic toolkit, and identified the correct next step. The portfolio value is in the process, not the outcome.

See **[CURRENT_STATE.md](CURRENT_STATE.md)** for the definitive claim status board, model roster, and what's next.

---

## What We Learned

### Sticky actions don't work
`repeat_action_probability=0.25` was the literature-standard fix for memorization in deterministic environments (Machado et al. 2018). We tested it across 7 PPO models. Every sticky-trained model collapsed to a deterministic script when tested without sticky actions. Sticky actions mask memorization with noise; they don't prevent or cure it. This independently confirms Zhang et al. (2018).

### The custom engine doesn't transfer to ALE
Experiments 5-11 ran on a custom GymBreakout engine. PPO_35 scored 212 points on the custom engine and **2 points** on authentic ALE/Breakout-v5 — a 99.1% drop. The rendering, physics, collision geometry, and frame timing differ enough that learned policies don't transfer. All new experiments train and evaluate on ALE.

### Every new metric needs dead-model calibration
A dead policy (confirmed argmax script, 1 unique score) produces score diversity, intervention retention, and shape classifier signals indistinguishable from models claimed to be reactive. Without running the same test on a known-dead model, the number is uninterpretable.

### The right direction: dynamics randomization on ALE
Perturbing environment physics (ball teleportation, variable speed) breaks timed scripts in a way that perceptual noise and action noise cannot. The logic is sound — a script that assumes "ball will be at (x,y) at frame N" fails when the ball is teleported. The missing piece is empirical proof on authentic Atari Breakout.

---

## Current Experiment

**ALE Experiment 1 — Ball teleportation via `setRAM()` (PPO_44)**

Training on authentic `ALE/Breakout-v5` with `ALEBreakoutRandomized`, a wrapper that teleports the ball to a random position on 30% of paddle bounces. This forces the policy to observe where the ball actually is rather than memorizing its expected position.

| Component | Detail |
|-----------|--------|
| Environment | ALE/Breakout-v5 (frameskip=1, nosticky) |
| Wrapper | `ALEBreakoutRandomized(teleport_prob=0.30)` — ball teleport on paddle bounce |
| Architecture | NatureCNN (standard, no dropout) |
| Target | 50M steps |
| Envs | 32 parallel |
| Pipeline | NoopReset → [Teleport] → FireReset → EpisodicLife → GrayscaleResize → ClipReward |

```bash
python train_ppo44.py
```

---

## Quick Start

```bash
git clone https://github.com/mharrell/BreakoutBot
cd BreakoutBot
pip install stable-baselines3[extra] gymnasium[atari] ale-py autorom torch opencv-python
AutoROM --accept-license
```

### Train
```bash
python train_ppo44.py    # ALE Experiment 1: ball teleportation
```

### Evaluate
```bash
python eval_reactivity.py --run PPO_44 --games 100    # Reactivity test (det=True + det=False)
```

### Calibrate
```bash
python calibrate_ale_intervention.py    # Dead-model baseline for intervention test
```

---

## Documentation

| File | Purpose |
|------|---------|
| **[CURRENT_STATE.md](CURRENT_STATE.md)** | **Read first.** Claim status board, model roster, lessons learned, next steps |
| [LOGICAL_AUDIT.md](LOGICAL_AUDIT.md) | 16-entry logical flaw catalog — reasoning patterns to avoid |
| [FLAWS.md](FLAWS.md) | 21-entry methodological flaw catalog |
| [EXPERIMENTS.md](EXPERIMENTS.md) | Full experiment history (claims corrected 2026-07-19) |
| [RL_REFERENCE.md](RL_REFERENCE.md) | PPO parameter guide, 31+ lessons, metric diagnostics |
| [CLAUDE.md](CLAUDE.md) | Project identity, critical rules, session bootstrap |

---

## Hardware

- **CPU:** Intel Core i5-13600K
- **GPU:** NVIDIA GeForce RTX 3060 Ti (8GB)
- **RAM:** 32GB
- Training speed: ~1,500-1,700 fps with 32 environments

---

## Reference

- Machado et al. (2018): Proposed sticky actions as memorization mitigation for deterministic ALE
- Zhang et al. (2018): Showed sticky actions don't prevent memorization in deep ConvNet agents — independently confirmed here
- This project: Dynamics randomization + dead-model calibration + Breakthrough Verification Protocol
