# BreakoutBot

**Not chasing high scores. Chasing reactivity.**

A reinforcement learning project investigating what forces a PPO agent to actually track the ball in Atari Breakout rather than memorize a fixed action sequence. Built with Stable-Baselines3/PyTorch on a single RTX 3060 Ti.

---

## Honest Status (July 31, 2026)

**After 120 experiments (118 Breakout + 2 BeamRider), no PPO model has ever genuinely generalized on any Atari game.** Every approach — sticky actions, dynamics randomization, entropy tuning, aux supervision, adversarial threats, life penalties, visible cursor adversaries, and non-conditionable ball bounce perturbation — has produced a memorized argmax.

The root cause is PPO's objective function: `argmax_π E[Σ rewards]`. In deterministic environments where scripts are viable, the expected-return-maximizing policy IS a memorized script. Every environment modification changed what script is optimal, not whether the optimum is a script.

**Current direction: alter the objective function itself.** PPO_119 (trajectory entropy bonus) is now training — it rewards the policy for taking different actions across parallel environments at the same timestep, directly attacking a script's defining property of identical actions every episode.

See **[CURRENT_STATE.md](CURRENT_STATE.md)** for the definitive claim status board, model roster, and what's next.

---

## What We Learned

### The distribution-vs-argmax confound (universal)
Every diagnostic except the split-watcher measures the policy *distribution*, but evaluation uses the *argmax*. PPO learned to maintain reactive-looking probability distributions while converging the argmax to a fixed script. This confound fooled us on cursor models (PPO_107-117, 33-50% intervention reversal rates, all memorized) and on BeamRider (MULTIPLE_SCRIPTS memcheck verdicts, both SINGLE_SCRIPT under split-watcher). **Only the split-watcher measures the argmax directly.**

### Sticky actions don't work
`repeat_action_probability=0.25` was the literature-standard fix for memorization. Every sticky-trained model collapsed to a deterministic script when tested without sticky actions. Sticky actions mask memorization with noise; they don't prevent or cure it.

### The custom engine doesn't transfer to ALE
PPO_35 scored 212 points on the custom GymBreakout engine and **2 points** on authentic ALE/Breakout-v5 — a 99.1% drop. All experiments now train and evaluate on ALE.

### Every new metric needs dead-model calibration
A dead policy produces score diversity, intervention retention, and shape classifier signals indistinguishable from models claimed to be reactive. If a known-dead model produces the same signal, the signal is not evidence of reactivity.

### CNN perception is not the bottleneck
NatureCNN can locate the ball to 1.9px MAE. The features exist. PPO's policy just never learns to use them.

### The right approach: change what PPO maximizes
After 120 experiments, the pattern is clear: environment changes produce different scripts, not reactive policies. The objective function itself must change to make scripts non-optimal. Approaches: trajectory entropy penalty, mutual information objective, adversarial predictability penalty, tracking reward shaping.

---

## Current Experiment

**Experiment 27 — Trajectory Entropy (PPO_119)**

Adds a cross-env action-diversity bonus: `bonus = 0.01 × (1 − p(action))` where `p(action)` is the fraction of parallel envs taking the same action at the same step. A script gets zero bonus (all envs identical). A reactive policy earns bonuses because different ball positions demand different actions.

| Component | Detail |
|-----------|--------|
| Environment | ALE/Breakout-v5 (frameskip=4, nosticky) |
| Wrapper | `TrajectoryEntropyWrapper(scale=0.01)` at VecEnv level |
| Architecture | NatureCNN (standard, no dropout) |
| Target | 25M steps |
| Envs | 32 parallel |
| Eval | Clean Breakout (no wrapper) — transfer test |

```bash
python train_ppo_119.py
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
python train_ppo_119.py    # Experiment 27: trajectory entropy
```

### Verify
```bash
python verify_split_watcher.py --model ./models/PPO_119/best_model.zip    # Gold-standard argmax test
```

---

## Documentation

| File | Purpose |
|------|---------|
| **[CURRENT_STATE.md](CURRENT_STATE.md)** | **Read first.** Claim status board, model roster, lessons learned, next steps |
| [FINDINGS_2026_07_30.md](FINDINGS_2026_07_30.md) | Split-watcher verification report — all cursor + BeamRider models memorized |
| [FLAWS.md](FLAWS.md) | 27-entry methodological flaw catalog |
| [LOGICAL_AUDIT.md](LOGICAL_AUDIT.md) | 17-entry logical flaw catalog — reasoning patterns to avoid |
| [EXPERIMENTS.md](EXPERIMENTS.md) | Full experiment history (27 experiments, 120 PPO runs) |
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
- This project: Split-watcher verification, dead-model calibration, distribution-vs-argmax confound documentation
