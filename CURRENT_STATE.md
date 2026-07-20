# Current State — BreakoutBot

**Last updated: 2026-07-19 (logical audit conclusion)**

---

## TL;DR

After 43 PPO runs, **no model in this project has ever genuinely generalized** to reactive ball-tracking in Breakout. Every model that appeared promising was found, on closer inspection, to be a memorized script or a noise-masked dead policy. The project's single largest risk — that the custom GymBreakout engine doesn't transfer to authentic Atari Breakout — has been **confirmed catastrophic**: PPO_35's 212-point GymBreakout argmax script scores **2 points** on ALE. The path forward is to return to authentic ALE Breakout using `setRAM()` for dynamics randomization, building on the project's hard-won diagnostic toolkit and the design insight that environment dynamics randomization (not perceptual noise, not sticky actions) is the right lever for forcing reactivity.

---

## Claim Status Board

Every major claim made in this project, with its current standing as of 2026-07-19.

### CONFIRMED — Supported by data

| Claim | Evidence |
|-------|----------|
| Sticky actions mask memorization; they don't prevent it | Every sticky-trained model tested without sticky actions collapsed to a deterministic script (PPO_26, PPO_28, PPO_29, PPO_30b, PPO_31b) |
| The MemorizationCheckCallback "GENERALIZING" verdict is invalid for sticky models | Dead policy + p=0.25 sticky = 8-14 unique scores. The verdict measures sticky probability, not policy quality. (F-001) |
| Deep non-sticky pretraining produces higher-scoring memorized scripts, not generalization | PPO_26: 60 pts > PPO_31b: 31 pts > PPO_30b: 0 pts — all confirmed memorized (F-003, F-004) |
| The intervention test does not distinguish reactive from dead | PPO_34 (dead argmax script) retains 49.6% vs PPO_35's 44.7%. Correlation = 0.000 for both. (L-001) |
| GymBreakout findings do not transfer to ALE | PPO_35: GymBreakout 212 pts → ALE 2 pts (99.1% drop). (L-007) |
| det=False score diversity exists in dead scripts | PPO_34 (dead): 19 unique det=False scores. PPO_35: 21. Identical signals. (L-012) |

### TENTATIVE — Plausible but not confirmed

| Claim | What's needed to confirm |
|-------|------------------------|
| Dynamics randomization forces reactive policies (the core design insight) | ALE replication. The insight was developed on the custom engine. Logic is sound but empirical proof on authentic Atari is absent. |
| Dropout in feature space prevents frame-timing memorization | Ablation at matched step counts. PPO_37 (no dropout) was killed before reaching PPO_36's step count. Single-pair comparison, not replicated. |
| Frame-level ball noise (σ=0.3) makes scripting worse than tracking | ALE replication. All ball-noise experiments were on the custom engine. |
| Per-episode physics randomization prevents memorization | ALE replication. PPO_34 tested on custom engine only. |

### FALSIFIED — Proven wrong

| Claim | How it was falsified |
|-------|---------------------|
| "PPO_35 is the first non-memorized model" (original 2026-07-15 claim) | Dead-model calibration shows identical signals. Intervention test doesn't distinguish. PPO_35 is an argmax script. ALE cross-eval: 2 points. (L-001, L-002, L-007) |
| "PPO_35's 47% intervention retention proves sighted behavior" | PPO_34 (dead) retains 49.6%. A dead script produces the same retention. Not evidence of reactivity. (L-001) |
| "PPO_30b GENERALIZING" (MemorizationCheckCallback verdict) | Nosticky verification: 99.8% zero scores, 2 unique scores. Sticky noise, not generalization. (F-001) |
| "PPO_31b GENERALIZING" | Nosticky verification: all 31-point script. (F-001) |
| "PPO_26 generalizes — both ingredients (pretraining + sticky) work" | Nosticky verification: 60 pts × 500 games, 264 frames. (F-003) |

---

## Model Roster

### Confirmed Memorized (nosticky verification)

| Model | Training | Nosticky result | Score |
|-------|----------|----------------|-------|
| PPO_25 | ALE, no sticky, 1B steps | Multiple scripts via eval cycling, single scripts per checkpoint | ~20-50 |
| PPO_26 | ALE, PPO_25 pretrain + p=0.25 sticky | 60.0 pts × 500 games, 264 frames — single script | 60 |
| PPO_27 | ALE, p=0.25 sticky from scratch, ~1B steps | 100% zero scores, 19-frame deaths — noise-dependent | 0 |
| PPO_28 | ALE, sticky removed from trained model | Collapsed to fixed open-loop sequence within ~30M steps | varies |
| PPO_29 | ALE, sticky removed from trained model | Collapsed to fixed open-loop sequence within ~30M steps | varies |
| PPO_30b | ALE, 100M non-sticky → 300M p=0.25 sticky | 99.8% zeros, 2 unique scores | 0 |
| PPO_31b | ALE, 300M non-sticky → 100M p=0.25 sticky | All 31-point script, 178 frames | 31 |

### Custom Engine — Confirmed Argmax Scripts (nosticky, det=True)

| Model | Approach | GymBreakout det=True | det=False | ALE (if tested) |
|-------|----------|---------------------|-----------|-----------------|
| PPO_34 | Per-episode physics randomization, 70M | 1 unique, 89.0 pts | 19 unique, 52.4 pts, UNCLEAR | — |
| PPO_35 | Continuous mid-game physics, 268M | 1 unique, 212.0 pts | 21 unique, 113.0 pts, UNCLEAR | **1 unique, 2.0 pts** |

### Custom Engine — Not Yet Evaluated with Full Protocol

| Model | Approach | Steps | Notes |
|-------|----------|-------|-------|
| PPO_36 | Ball noise σ=0.3 + dropout p=0.1 | 99M | Checkpoints exist at 3.2M increments through 99.2M. Not cross-evaluated on ALE. |
| PPO_33 | Frame skip randomization | 5 restarts | Original "phase transition at 15M" observation from lost run. |

### Never Existed

PPO_37 through PPO_43 were planned/designed but never trained beyond early steps (or were killed before producing checkpoints). PPO_41/42/43 were LR schedule experiments killed on 2026-07-19 to free GPU for ALE return.

---

## What We've Learned

### The hard way (things we believed, then disproved)

1. **Sticky actions don't work.** They were the standard literature recommendation (Machado et al. 2018). They mask memorization with noise; they don't prevent or cure it. Every sticky-trained model ever tested in this project collapsed without sticky actions.

2. **Non-sticky pretraining causes permanent memorization.** Once a model memorizes during non-sticky pretraining, no amount of sticky fine-tuning has ever cured it. The model learns a fixed script during the deterministic phase and the sticky phase just adds noise around it.

3. **Score diversity is not reactivity.** Dead argmax scripts produce 8-21 unique scores under stochastic sampling (det=False). PPO_34 (confirmed dead) and PPO_35 produce indistinguishable score diversity. The only reliable behavioral test is nosticky verification: run the model without sticky actions and check for collapse to ≤2 unique scores.

4. **Intervention retention is not a reactivity metric without calibration.** A dead script that always moves the paddle to position X after a bounce will still score SOME points when the ball is teleported — the paddle doesn't disappear. PPO_34 (dead) retains 49.6% score under teleportation. Without a dead-model calibration baseline, retention percentages are uninterpretable.

5. **The custom engine doesn't approximate ALE.** PPO_35 learned a 212-point argmax script on GymBreakout. The same policy scores 2 points on authentic Atari Breakout. The engines differ in rendering, physics, collision geometry, and frame timing in ways that make learned policies non-transferable.

6. **Every new metric needs dead-model calibration.** The project has a recurring pattern: a new diagnostic is developed, run on the latest promising model, produces a number, and that number is interpreted as evidence of reactivity — without checking what a known-dead model produces on the same diagnostic. This happened with the intervention test, the shape classifier, and the MemorizationCheckCallback verdicts. Critical Rule #14 now requires calibration before any new metric is used to support claims.

### The right direction (things we believe and haven't disproven)

1. **Dynamics randomization is the right lever.** Perturbing environment physics (ball speed, paddle width, frame timing) breaks timed scripts in a way that perceptual noise and action noise cannot. The logic is sound. What's missing is empirical proof on authentic ALE.

2. **Diagnostic toolkit is robust.** Nosticky verification, det=True vs det=False comparison, bootstrap CIs on distribution metrics, and cross-environment evaluation — when applied *before* claims are written — would have caught every false positive in this project's history. The tools exist; the failure was in the sequence (claim first, verify later).

---

## What's Next

### Immediate: ALE Return — First Experiment Spec

The project is returning to authentic ALE/Breakout-v5. Here is the concrete first experiment a new session can start building immediately.

#### Step 0: Verify RAM Addresses (30 min)

The addresses below are from OCAtari — they may be wrong for ALE v0.11. Write a probe script that sets each address and observes the effect:

```
Expected (from OCAtari, UNVERIFIED):
  Paddle X:    72    (write 0-191, observe paddle position)
  Ball X:      99    (write 0-191, observe ball horizontal)
  Ball Y:      101   (write 0-255, observe ball vertical)
  Score:       76-77 (read-only verification)
  Lives:       57    (read-only verification)
```

Pattern: `env.unwrapped.ale.setRAM(addr, value)` — set before `env.step()`. Verify each address by writing known values and visually confirming. Update the address list in `ale-return-direction.md` memory with verified values.

#### Step 1: Build `ALEBreakoutRandomized` Wrapper (1-2 hours)

A `gym.Wrapper` around `ALE/Breakout-v5` (frameskip=1, repeat_action_probability=0) that supports the same intervention test as the GymBreakout version:

- **Ball teleportation**: on paddle bounce, 30% probability, teleport ball to random X/Y via `setRAM()`
- **No other randomization** — single variable for the first experiment
- **Same preprocessing** as existing pipeline: GrayscaleResize(84,84), ClipRewardEnv, Monitor → DummyVecEnv → VecFrameStack(4)

Reference pattern: `calibration_phase1.py` `InterventionBreakout` class (wraps GymBreakout, teleports ball). Same logic, different engine.

#### Step 2: Calibrate on Known-Dead ALE Models (1 hour)

Before training anything new, establish the ALE dead-model baseline. Run the intervention test on:
- **PPO_26** (ALE-trained, confirmed memorized: 60 pts × 500 games nosticky)
- **PPO_30b** (ALE-trained, confirmed memorized: 99.8% zeros)

This gives us: "A known-dead ALE model retains X% under intervention." That's the calibration baseline that was missing for GymBreakout. Without it, intervention retention numbers are uninterpretable (see L-001).

#### Step 3: Train Small Proof-of-Concept (1-2 days GPU)

Train a NatureCNN PPO model on `ALE/Breakout-v5` with the teleportation wrapper from Step 1:

- **Architecture**: NatureCNN (standard — no dropout yet, single variable)
- **Training**: 32 envs, standard PPO hyperparams (matching train_ppo36.py defaults)
- **Target**: ~50M steps — enough to see if the policy develops reactivity
- **During training**: MemorizationCheckCallback with `make_env_fn` pointing to ALE (not GymBreakout — this time the callback env matches the training env)
- **If the model memorizes (≤2 unique nosticky at 50M)**: teleportation alone isn't enough. Consider adding ball velocity perturbation or per-episode physics randomization (setRAM-based).
- **If the model shows diversity (≥10 unique nosticky)**: run full Breakthrough Verification Protocol (all 6 gates, in order).

#### Step 4: Full Verification (if Step 3 succeeds)

- Gate 1: Both inference modes tested (det=True + det=False, 100 games)
- Gate 2: Calibration baseline (Step 2 data)
- Gate 3: Comparison baseline (compare against PPO_26)
- Gate 4: Std column verified
- Gate 5: Falsification pre-registered BEFORE claiming success
- Gate 6: Environment match confirmed (training = ALE, eval = ALE)

**Do not write the "first sighted ALE policy" claim until all 6 gates pass.** The project has 4 documented false positives from skipping gates (see Claim Status Board above).

### For New Sessions

See `CURRENT_STATE.md` (this file) first — then:
1. `LOGICAL_AUDIT.md` — understand what reasoning patterns to avoid
2. `FLAWS.md` — understand what methodological pitfalls exist
3. `CLAUDE.md` — critical rules, conventions, diagnostic checklist
4. `EXPERIMENTS.md` — full experiment history (with corrected claims)

The Session Bootstrap in `CLAUDE.md` has the step-by-step procedure.

### For Portfolio Reviewers

This project demonstrates hands-on ML engineering with honest self-assessment. The key documents:
- **This file** — current state and claim status
- `LOGICAL_AUDIT.md` — 16-entry logical flaw catalog (3 confirmed with data)
- `FLAWS.md` — 21-entry methodological flaw catalog
- `EXPERIMENTS.md` — full experiment history with documented false positives
- `RL_REFERENCE.md` — 31+ lessons learned, PPO parameter guide, metric diagnostics

The story is not "we built a working Breakout AI" — it's "we ran 43 experiments, caught every false positive ourselves, built a rigorous diagnostic toolkit, and identified the correct next step." The portfolio value is in the process, not the outcome.
