# BreakoutBot 🎮

**Not chasing high scores. Chasing reactivity.**

A reinforcement learning project investigating what actually forces a PPO agent to track the ball in Atari Breakout, rather than memorize a fixed sequence of actions. Built with Stable-Baselines3/PyTorch on a single RTX 3060 Ti.

---

## The Problem

Atari Breakout is deterministic. A CNN policy doesn't need to watch the ball — it can memorize "at frame 47, press LEFT" for every frame it encounters. This looks great on the scoreboard but isn't playing Breakout.

**Sticky actions** (`repeat_action_probability=0.25`) were proposed in 2018 as the fix. We tested them across 7 PPO models totaling billions of training steps. They don't work. ConvNets are naturally noise-robust — the memorized script survives 25% action-repeat with minimal degradation.

Every sticky-trained model in this project collapsed to a deterministic script when tested without sticky actions:
- PPO_26 (1.8B steps): 60-point memorized script, identical every game
- PPO_30b (400M steps): 99.8% zero-score memorized script  
- PPO_31b (400M steps): 31-point memorized script
- PPO_27 (1B steps): 100% zero-score, noise-dependent degenerate

**Zero of these models genuinely tracked the ball.**

---

## What Worked

**Continuous physics domain randomization.** Instead of perturbing the agent's output (sticky actions), we perturb the game's physics continuously during play. Every 60-300 frames, 0-3 of (paddle width, paddle speed, ball speed) smoothly interpolate to new random values over 30 frames.

The agent can't memorize a parameter-conditioned script because the parameters never settle.

### PPO_35 — First Non-Memorized Model (in progress, 64M/400M steps)

| Metric | PPO_35 | PPO_32 (sticky p=0.05) | PPO_34 (per-episode rand) |
|--------|--------|------------------------|---------------------------|
| Unique eval scores | **21+** | 2 | 2 |
| explained_variance | **0.85** | 0.93 | 0.96 |
| Verdict | **NOT MEMORIZED** | Memorized | Memorized |

The explained variance gap tells the story: PPO_35's value function *can't* converge because the physics keep changing. PPO_32 and PPO_34 converged to deterministic attractors and stayed there.

---

## Current State (July 2026)

| Run | Experiment | Approach | Status |
|-----|-----------|----------|--------|
| PPO_32 | Goldilocks (p=0.05) | Low-sticky single-phase, 400M | Memorized 70-pt script, boom-bust cycling |
| PPO_35 | Domain Randomization C | Continuous physics changes, 400M | **First non-memorized model** — 21+ unique eval scores |

See [EXPERIMENTS.md](EXPERIMENTS.md) for the full experimental history, failure analysis, and design rationale. See [FLAWS.md](FLAWS.md) for the catalog of methodological issues discovered along the way.

---

## The Nosticky Verification Protocol

The only reliable behavioral test for memorization: run the model without sticky actions (or other noise sources) and check whether it produces ≤2 unique scores across multiple games. A dead policy + sticky p=0.25 produces 8-14 "unique" scores from noise alone. Unique-score count in noisy environments measures noise level, not policy quality. Every model must pass the nosticky test before claiming generalization.

---

## Setup

```bash
git clone https://github.com/mharrell/BreakoutBot
cd BreakoutBot
pip install stable-baselines3[extra] gymnasium[atari] ale-py autorom torch
AutoROM --accept-license
```

### Train
```bash
python train_ppo35.py    # Experiment 5C: continuous physics randomization
```

### Watch
```bash
python watch_ppo33.py    # Record a high-scoring game from a checkpoint
```

---

## Hardware

- **CPU:** Intel Core i5-13600K
- **GPU:** NVIDIA GeForce RTX 3060 Ti (8GB)
- **RAM:** 32GB
- Training speed: ~1,100-1,700 fps with 32 environments

---

## Key Documentation

| File | Purpose |
|------|---------|
| [EXPERIMENTS.md](EXPERIMENTS.md) | Full experiment history, designs, predictions, outcomes |
| [RL_REFERENCE.md](RL_REFERENCE.md) | PPO parameter guide, 40+ lessons, metric diagnostics |
| [FLAWS.md](FLAWS.md) | 21 known flaws in experimental process (read before interpreting results) |
| [CLAUDE.md](CLAUDE.md) | Project identity, critical rules, session bootstrap |

---

## Reference

- Machado et al. (2018): Proposed sticky actions as memorization mitigation for deterministic ALE
- Zhang et al. (2018): Showed sticky actions don't prevent memorization in deep ConvNet agents — independently confirmed here across 7 PPO models
- This project: Continuous dynamics randomization as the fix, nosticky verification as the test
