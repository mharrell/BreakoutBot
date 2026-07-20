# BreakoutBot 🎮

**Not chasing high scores. Chasing reactivity.**

A reinforcement learning project investigating what actually forces a PPO agent to track the ball in Atari Breakout, rather than memorize a fixed sequence of actions. Built with Stable-Baselines3/PyTorch on a single RTX 3060 Ti.

---

## The Problem

Atari Breakout is deterministic. A CNN policy doesn't need to watch the ball — it can memorize "at frame 47, press LEFT" for every frame it encounters. This looks great on the scoreboard but isn't playing Breakout.

**Sticky actions** (`repeat_action_probability=0.25`) were proposed in 2018 as the fix. We tested them across 7 PPO models totaling billions of training steps. They don't work. ConvNets are naturally noise-robust — the memorized script survives 25% action-repeat with minimal degradation. This independently confirms Zhang et al. (2018).

Every sticky-trained model collapsed to a deterministic script when tested without sticky actions:

| Model | Training | Nosticky result |
|-------|----------|----------------|
| PPO_26 | 838M non-sticky → 163M sticky | 60 pts × 500 games, 264 frames — identical every game |
| PPO_30b | 100M non-sticky → 300M sticky | 0 pts × 99.8%, rare 69 |
| PPO_31b | 300M non-sticky → 100M sticky | 31 pts × 500 games, 178 frames — identical every game |
| PPO_27 | p=0.25 from scratch, ~1B | 0 pts × 100%, 19-frame deaths |

**Zero of these models genuinely tracked the ball.**

---

## What We're Testing Now

**Dynamics randomization.** Instead of perturbing the agent's output (sticky actions), we perturb the game's physics. Three approaches, from weakest to strongest perturbation:

| Experiment | Approach | Hypothesis | Result |
|-----------|----------|------------|--------|
| PPO_33 (5A) | Random frame skip 2-8 | Varying timing prevents memorization | **Failed.** 5 restarts. CNN detects speed from first frames → speed-conditioned scripts. SUPERSEDED. |
| PPO_34 (5B) | Per-episode physics randomization | Varying physics per game prevents memorization | **Partial.** det=True = 89-pt memorized script. det=False = 20 unique scores but bimodal (36-41 cluster + 82-89 tail) — parameter-conditioned scripts. |
| PPO_35 (5C) | Continuous mid-game physics changes | Unpredictable physics shifts force moment-to-moment reactivity | **Most promising.** See below. |

### PPO_35 — The Most Nuanced Result in the Project

PPO_35 trains on `DynamicBreakout` — paddle width, paddle speed, and ball speed change continuously every 60-300 frames. It's evaluated on `GymBreakout(fixed=True)` — standard Breakout defaults with no physics changes.

**Reactivity evaluation (eval_reactivity.py, 2026-07-15, 64M steps, 100 games each):**

| Inference mode | Unique scores | Mean | Std | Range |
|---------------|---------------|------|-----|-------|
| `deterministic=True` (argmax) | **1** (107) | 107.0 | 0.0 | 107-107 |
| `deterministic=False` (sampled) | **26** | 83.8 | 10.9 | 42-104 |

**The argmax is a memorized 107-point script.** Every game with `det=True` is identical — 107 points, same action sequence, zero variance.

**But sampling reveals real policy entropy.** 26 unique scores in 100 games with no environment noise. The policy's action distribution has genuine breadth — it learned to handle diverse physics during training, and that uncertainty persists at inference. Score distribution clusters tightly at 80-94 (not random), suggesting structured behavior rather than script + noise.

**This is NOT the same as previous "generalizing" claims:**
- PPO_30b/31b diversity came from sticky environment noise on a near-zero-entropy policy
- PPO_35 diversity comes from the policy's own action distribution on a noise-free environment
- Whether the policy genuinely tracks the ball or plays a base script with distribution noise is an open question

**Comparison with PPO_34 (per-episode randomization):**

| | PPO_35 (continuous) | PPO_34 (per-episode) |
|---|---|---|
| det=True | 1 unique, 107 pts | 1 unique, 89 pts |
| det=False unique | 26 | 20 |
| det=False mean | 83.8 | 50.2 |
| Distribution | Tight cluster 80-94 | Bimodal: 36-41 (48%) + 82-89 |

Per-episode randomization (PPO_34) produced parameter-conditioned scripts — the CNN detects physics parameters from early frames and selects a script tuned for those parameters. On fixed-default physics, it sometimes picks the wrong script (36-41) and sometimes the right one (82-89). Continuous changes (PPO_35) prevent parameter detection, producing a tighter, higher-scoring distribution.

---

## Current State (July 2026)

| Run | Step | Experiment | Approach | Key Signal |
|-----|------|-----------|----------|-------------|
| **PPO_32** | 93M/400M | Exp 4 | p=0.05 sticky from scratch | **KILLED.** Low-sticky doesn't prevent memorization. |
| **PPO_33** | 3.2M | Exp 5A | Random frame skip 2-8 | **KILLED.** 5 restarts. SUPERSEDED. |
| **PPO_34** | 70M/400M | Exp 5B | Per-episode physics rand | **COMPLETED.** det=False: 20 unique, CLUSTERED/bimodal. |
| **PPO_35** | 268M/400M | Exp 5C | Continuous physics changes | **KILLED.** 268/268 SINGLE_SCRIPT. Argmax never broke. |
| **PPO_36** | **294M**/400M | Exp 6 | Ball noise σ=0.3 + dropout p=0.1 | **KILLED.** Dissolution regressed: 41%→58% top-3. LR decay confirmed as bottleneck. |
| **PPO_37** | **100M**/400M | Exp 6 Ablation | Ball noise only (no dropout) | **KILLED.** 4 entropy collapses. Ball noise=diversity, dropout=stability. |
| **PPO_38** | **7M**/400M | Exp 7 | Ball noise σ=**0.5** + dropout | **KILLED.** σ=0.5 + dynamics drowned agent. |
| **PPO_39** | **111M**/400M | Exp 8 | Wider network (2× CNN) + dropout | **KILLED.** 14→7 unique. Wider doesn't help. |
| **PPO_40** | **130M**/400M | Exp 9 | σ=0.5 + **fixed physics** + dropout | **KILLED.** Scores compressed 2-20. σ=0.5 too aggressive. |
| **PPO_41** | **0M**/400M | Exp 10A | **Constant LR 2.5e-4** | **JUST LAUNCHED.** Does high LR sustain dissolution? |
| **PPO_42** | **0M**/400M | Exp 10B | **Constant LR 1e-4** | **JUST LAUNCHED.** Does mid-range constant LR work? |
| **PPO_43** | **0M**/400M | Exp 10C | **Linear 2.5e-4→5e-5** | **JUST LAUNCHED.** Does a higher floor fix regression? |

### The Breakthrough: Script Clusters Are Dissolving (with LR-Dependent Regression)

PPO_36's det=False distribution over time:

| At | Top-3 conc | Max gap | Shape |
|----|-----------|---------|-------|
| 35M | 70% | 36 pts | CLUSTERED (script-switching) |
| 90M | 65% | 36 pts | CLUSTERED |
| **169M** | **41%** | **5 pts** | **Near-continuous** |
| **182M** | **58%** ⚠ | **21 pts** ⚠ | **CLUSTERED (regressed)** |
| **294M** | **worse** ⚠ | — | **Fully regressed** |

Ball noise + dropout together dissolved script clusters from 70%→41% top-3 concentration over 134M steps. **But dissolution regressed as LR decayed** — the LR passed below ~1.4e-4 and policy updates became too small to overcome script attractors. Dissolution is LR-dependent: the linear decay schedule works against the dissolution trajectory.

PPO_37 (killed at 100M) proved neither mechanism works alone: ball noise creates diversity, dropout stabilizes entropy against collapse. The compound effect requires both.

PPO_39 (killed at 111M) proved wider network doesn't help — 3.5M params produces same CLUSTERED script-switching as 1.7M.

PPO_40 (killed at 130M) proved σ=0.5 is too aggressive — scores permanently compressed to 2-20.

**Experiment 10 (PPO_41/42/43) now testing the LR bottleneck directly** — three identical recipes, only the LR schedule differs.

See [EXPERIMENTS.md](EXPERIMENTS.md) for full history. See [FLAWS.md](FLAWS.md) for methodological issues.

---

## The Verification Problem

The project has produced three false positives where a promising number was interpreted as proof of reactivity before the falsification test was run:

| When | Model | Claim | Why it felt real | What killed it |
|------|-------|-------|-----------------|----------------|
| June | PPO_26 | "Both ingredients work" | 54.3 avg, 0% zero-score | Nosticky: 60-pt script |
| July 12 | PPO_30b/31b | "GENERALIZING streak" | 33/33 checks passing | Calibration: dead policy = same signal |
| July 14 | PPO_35 | "First reactive model" | 21+ eval means, EV=0.85 | std=0.0 on every row; cross-checkpoint cycling |

**The pattern:** a single number was read as proof. The test that would falsify it (nosticky, calibration sweep, std column) existed but wasn't run before the claim was written to memory.

### The Fix: Breakthrough Verification Protocol

Before writing ANY claim that a model "generalizes," "is reactive," or represents a "first ever":

1. **Test both inference modes** — `eval_reactivity.py`: det=True AND det=False, ≥100 games each
2. **Check the std column** — `logs/*/evaluations.npz` — if every row has std=0.0, it's script-switching
3. **Run a calibration baseline** — what does a known-dead policy produce?
4. **Compare against another model** — if "failed" approaches produce similar numbers, "first ever" is wrong
5. **Pre-register the falsification test** — "this claim is wrong if _______" — then run it
6. **Confirm environment match** — MemorizationCheckCallback hardcodes ALE; GymBreakout models need `make_env_fn`

Protocol is in [CLAUDE.md](CLAUDE.md) and [.opencode/instructions.md](.opencode/instructions.md).

### Key Methodological Lessons

- **`deterministic=True` on a deterministic environment produces identical episodes regardless of reactivity.** std=0.0 is expected. You MUST test with `deterministic=False` or environment randomness to detect memorization.
- **Cross-checkpoint eval mean cycling ≠ score diversity.** If eval means vary across checkpoints but std=0.0 at each checkpoint, the model is script-switching (training instability), not reactive.
- **The MemorizationCheckCallback tests ALE Breakout, not GymBreakout.** It's hardcoded on line 84. For PPO_33/34/35 (GymBreakout-trained), this data is OOD testing and meaningless.
- **`explained_variance` from a non-stationary training environment is uninformative.** Low EV means the value function can't converge — expected when physics change mid-game. It doesn't measure policy health.

---

## Quick Start

```bash
git clone https://github.com/mharrell/BreakoutBot
cd BreakoutBot
pip install stable-baselines3[extra] gymnasium[atari] ale-py autorom torch
AutoROM --accept-license
```

### Train
```bash
python train_ppo36.py    # Experiment 6: ball noise + latent dropout
```

### Evaluate
```bash
python eval_reactivity.py --run PPO_36 --games 100    # Reactivity test (det=True + det=False)
python eval_reactivity.py --run PPO_34 --games 100    # Compare against per-episode baseline
```

### Watch
```bash
python watch_ppo33.py    # Record gameplay from a checkpoint
```

---

## Hardware

- **CPU:** Intel Core i5-13600K
- **GPU:** NVIDIA GeForce RTX 3060 Ti (8GB)
- **RAM:** 32GB
- Training speed: ~1,100-1,700 fps with 32 environments

---

## Documentation

| File | Purpose |
|------|---------|
| [EXPERIMENTS.md](EXPERIMENTS.md) | Full experiment history, designs, predictions, outcomes |
| [RL_REFERENCE.md](RL_REFERENCE.md) | PPO parameter guide, 40+ lessons, metric diagnostics |
| [FLAWS.md](FLAWS.md) | 21 known flaws in experimental process |
| [CLAUDE.md](CLAUDE.md) | Project identity, critical rules, Breakthrough Verification Protocol |
| [.opencode/instructions.md](.opencode/instructions.md) | Session bootstrap, agent guardrails, misinterpretation traps |
| `eval_reactivity.py` | Standard reactivity test — det=True + det=False on GymBreakout |

---

## Reference

- Machado et al. (2018): Proposed sticky actions as memorization mitigation for deterministic ALE
- Zhang et al. (2018): Showed sticky actions don't prevent memorization in deep ConvNet agents — independently confirmed here across 7 PPO models
- This project: Dynamics randomization + `deterministic=False` evaluation + Breakthrough Verification Protocol
