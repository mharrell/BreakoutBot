# Current State — BreakoutBot

**Last updated: 2026-07-28 23:50 (PPO_107 launched, BeamRider reactivity CONFIRMED, cursor adversary design)**

---

## TL;DR

**Perception POC (July 26): NatureCNN CAN track the ball to 1.9px MAE (0.6px median).** The architecture has the perceptual capability. PPO_85 proved that even with those perfect ball-position features frozen in, PPO converges to a 0pt blind script. The SINGLE_SCRIPT problem is in the optimization, not the perception.

**Dose-response curve (July 27): Teleport magnitude changes which script gets memorized, not whether.** ±30px produces a 21pt script (best in the project for dynamics). ±35px is marginal (0-5pt). ±40px+ is dead (0pt). But every single setting produces SINGLE_SCRIPT on clean ALE. Teleports change the script quality, not the memorization outcome.

**BeamRider CONFIRMED GENUINELY REACTIVE (July 28 evening).** The noop=0, det=True verification test was run: 100 games, zero random offset, argmax action selection. Result: 6 unique scores, MULTIPLE_SCRIPTS. A memorized script would produce ≤2. The std was 33.7 — higher than with noop noise — because the policy genuinely reacts to what happens each game. This is the **first verified reactive PPO argmax in the project's 107-run history.**

**Revised thesis (July 28):** Hard failure was falsified by BEAMRIDER_MULTILIFE (soft failure, 3 lives, still MULTIPLE_SCRIPTS). The real mechanism is **adversarial entities that target the agent's position.** BeamRider enemies aim at the player's ship — a fixed action sequence puts the ship at a predictable position, enemies shoot there, scripts are non-viable regardless of how many lives you have. The mechanism is: **adversarial entity → threatens position → fixed patterns are exploitable → reactive policy required.**

**Experiment 16 (PPO_107, ACTIVE): Port the mechanism to Breakout.** A visible cursor with its own state machine approaches the ball when the paddle isn't tracking, warns before attacking (visible pulsing), and pushes the ball away on attack. Tracking paddle → cursor retreats and hides. Cursor is only visible during THREATENING/ATTACK — in eval (standard Breakout), no cursor appears because tracking keeps it hidden. Calibration gap: perfect=14.0 vs best script=1.9 (gap=12.1). At 4M: SINGLE_SCRIPT but 20pt script — already the best adversarial-trained script (PPO_105: 13pt, PPO_106: 12pt).

**Direction flip test (July 28):** RAM[105] direction control tested. Dodge mode over-penalizes tracking (14→5), flip mode accidentally helps tracking (14→26). Direction-based adversarial has same fundamental limitation as position-based — doesn't solve the deterministic-function problem.

**July 28 earlier:** Multi-env probes complete. PPO_102/103 prove perception-policy gap is structural. Life-loss penalty (PPO_101) teaches survival, not reactivity. Adversarial push wrappers (PPO_105, PPO_106 v1/v2/v3) all SINGLE_SCRIPT — deterministic wrappers preserve memorizability. See `FINDINGS_2026_07_28.md` for full writeup.

---

## Terminology

| Term | Definition |
|------|-----------|
| **SINGLE_SCRIPT** | ≤2 unique scores on det=True across n_games. The argmax produces the same score every game. |
| **MULTIPLE_SCRIPTS** | 3+ unique scores on det=True. Only observed on BeamRider (hard failure constraints). |
| **Memorized** | SINGLE_SCRIPT on clean ALE det=True. Does NOT imply 0pt — a 21pt script is still memorized. |
| **Dead** | 0pt exactly on clean ALE det=True. A subset of memorized. |
| **Blind script** | A memorized policy with no evidence of ball-tracking. Synonymous with "memorized script." |
| **Stoch** | det=False (stochastic sampling). Produces score diversity even in memorized policies. |
| **Collapse** | Transition from MULTIPLE_SCRIPTS to SINGLE_SCRIPT, or from nonzero to 0pt. |
| **Hard failure** | Environment kills the agent for a mistake (BeamRider: one bullet = death). Scripts non-viable. |
| **Soft failure** | Mistake degrades state but doesn't end game (Breakout: lose ball, re-serve). Scripts remain viable. |

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
| **BeamRider produces MULTIPLE_SCRIPTS — first reactive PPO argmax** | 10/10 MULTIPLE_SCRIPTS at 10M. Hard failure (one bullet = death) makes memorized scripts non-viable. First and only game to break SINGLE_SCRIPT. |
| **SINGLE_SCRIPT is a general PPO property, not Breakout-specific** | Pong (SINGLE_SCRIPT, perfect-win script), Space Invaders (SINGLE_SCRIPT despite UFO randomness), Freeway (SINGLE_SCRIPT 0pt), Breakout (100+ experiments). 4/5 games SINGLE_SCRIPT. |
| **Aux supervision CAN bake ball-tracking features into the CNN during PPO training** | PPO_102 (after callback bug fix): MSE 70.55→0.008 (1344px→14px) in 1.7M aux-training steps. Gradient flows correctly. Features encode ball at ~14px. |
| **Ball-tracking features do NOT prevent policy memorization** | PPO_102 at 14.5M: 14px aux precision, SINGLE_SCRIPT (stoch=1 unique). PPO_103 at 946K: 16px, SINGLE_SCRIPT. Features encode ball position but policy ignores them. |
| **The perception-policy gap is structural** | PPO_103: policy collapses to script in ~200 PPO updates — faster than aux can shape features to pixel precision, even at 10× gradient strength. Not a gradient-strength problem. |
| **Life-loss penalty (-10/life) does not prevent memorization** | PPO_101: SINGLE_SCRIPT through 14M. Scores climbed 0→17. Teaches survival, not ball-tracking. Penalty too small relative to script score. |
| **BeamRider is MULTIPLE_SCRIPTS from the very first checkpoint (1M)** | BEAMRIDER_BASELINE tracked at 1M intervals through 10M. Never SINGLE_SCRIPT. Reactive from training start. |
| **BeamRider stays MULTIPLE_SCRIPTS WITHOUT hard failure** | BEAMRIDER_MULTILIFE (no EpisodicLifeEnv, 3 lives/sector): MULTIPLE_SCRIPTS at 1M, 2M, 3M. Getting MORE diverse (6→6→8 unique). Hard failure not required. |
| **Adversarial threat, not hard failure, is the mechanism that forces reactivity** | BeamRider enemies aim at the player's position. Fixed movement patterns are predictable → enemies shoot where you'll be. This is true with 1 life or 3. The environment actively punishes fixed patterns. Hard failure is neither necessary nor sufficient. |
| **BeamRider reactivity is GENUINE — verified by noop=0, det=True test** | 100 games, noop_max=0, deterministic=True: 6 unique scores, MULTIPLE_SCRIPTS, std=33.7. First verified reactive PPO argmax in project history. A memorized script would produce ≤2 unique. Score diversity survives with zero noop offset — policy genuinely reacts to game state. |
| **Direction-based adversarial (RAM[105]) does not solve the deterministic-function problem** | Dodge mode over-penalizes tracking (14→5), flip mode accidentally helps tracking (14→26). Same fundamental limitation as position-based push: any deterministic f(ball_x, paddle_x) can be optimized by a fixed sequence. |
| **Visible cursor adversary creates 12.1pt calibration gap** | Perfect tracking=14.0, best script=1.9. Cursor has agency (state machine), visibility (pulsing before attack), and retreat (tracking keeps it hidden). Strongest adversarial calibration in the project. |

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
| **"Life-loss penalties force reactive ball-tracking"** | PPO_101: -10/life, SINGLE_SCRIPT through 14M. Teaches survival while scripting. |
| **"Ball-tracking features → reactive policy by construction"** | PPO_102/103: features encode ball position (14-16px), policy is SINGLE_SCRIPT. Features necessary but not sufficient. |
| **"Stronger aux gradient → pixel-precision features → reactivity"** | PPO_103: 10× gradient, features improving (16px at 946K), but policy collapses to script in ~200 updates. PPO memorizes faster than aux can shape features. |
| **"Hard failure (one hit = death) is the mechanism that forces reactivity in BeamRider"** | BEAMRIDER_MULTILIFE (no EpisodicLifeEnv, soft failure): MULTIPLE_SCRIPTS at 1M/2M/3M. Removing hard failure did NOT produce SINGLE_SCRIPT. The real mechanism is adversarial threat — enemies that aim at the player's position punish fixed movement patterns. Hard failure is neither necessary nor sufficient. |
| **"One-life Breakout will force reactivity"** (TENTATIVE) | PPO_104 at 1M: SINGLE_SCRIPT (1 unique, 3.0 avg). Scripts score 3 points on one ball — worse than 5-life scripts but still a viable local optimum. One life doesn't change the fundamental dynamic: the ball follows physics and can't exploit fixed paddle patterns. |

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

### Multi-Environment Replication Probes (COMPLETED July 28)

| Game | Seed | Steps | det=True | det=False | Notes |
|------|------|-------|----------|-----------|-------|
| Pong | 200 | 10M | SINGLE_SCRIPT (2 unique) | MULTIPLE_SCRIPTS (9 unique) | Perfect -21/-20 win script by 4M |
| Space Invaders | 201 | 10M | SINGLE_SCRIPT (2 unique) | MULTIPLE_SCRIPTS (7 unique) | 180-220 pts. UFO randomness insufficient |
| **BeamRider** | **202** | **10M** | **MULTIPLE_SCRIPTS (3 unique)** | MULTIPLE_SCRIPTS (8 unique) | **First reactive argmax in project history** |
| Freeway | 203 | 10M | SINGLE_SCRIPT (1 unique, 0.0) | SINGLE_SCRIPT (1 unique, 0.0) | Never learned. Chicken never crossed. |

**BeamRider analysis:** Hard failure (one bullet = death) forces reactivity. There is no safe sweep. Scripts are non-viable. This is the unifying principle across 104 experiments.

### BeamRider Paired Experiment — Hard vs Soft Failure (COMPLETED/RUNNING July 28)

To isolate whether hard failure or adversarial threat is the mechanism behind BeamRider's reactivity.

| Model | Seed | EpisodicLifeEnv | Failure | Progress | det=True Result |
|-------|------|-----------------|---------|----------|-----------------|
| BEAMRIDER_BASELINE | 206 | YES | Hard (1 life/sector) | 10M COMPLETE | MULTIPLE_SCRIPTS all checkpoints 1M→10M |
| BEAMRIDER_MULTILIFE | 205 | NO | Soft (3 lives/sector) | 3M RUNNING | MULTIPLE_SCRIPTS 1M→3M (getting more diverse) |

**Finding: Hard failure is NOT the mechanism.** Both variants are MULTIPLE_SCRIPTS. Removing EpisodicLifeEnv did not produce SINGLE_SCRIPT. The real mechanism is adversarial threat: BeamRider enemies aim at the player's position. A fixed pattern is predictable → enemies shoot where you'll be → scripts are non-viable regardless of how many lives you have.

### Experiment 10: Life-Loss Penalty — PPO_101 (COMPLETED — Negative)

| Model | Config | Step | det=True | Notes |
|-------|--------|------|----------|-------|
| PPO_101 | -10/life, annealed 5M, SEED=101 | 14M (stopped) | SINGLE_SCRIPT all checks | Scores climbed 0→17. Teaches survival, not reactivity. |

### Experiment 11: Ball-Tracking Representation Supervision — PPO_102 (COMPLETED — Science Done)

| Model | Config | Step | det=True | Notes |
|-------|--------|------|----------|-------|
| PPO_102 | aux_lr=1e-4, epochs=2, SEED=102 | 14.5M | SINGLE_SCRIPT (stoch=1) | Callback bug fixed at 12.8M. Aux MSE 70.55→0.008 (1344px→14px). Features encode ball, policy still memorized. |

**Critical infrastructure bug:** `_train_aux()` silently returned every call for 12.8M steps due to buffer-size mismatch (`rollout_buffer.size()` returns 128 not 4096) and unflattened observation shape. Both fixed. Bug analysis in `FINDINGS_2026_07_28.md`.

### Experiment 12: Stronger Aux from Scratch — PPO_103 (COMPLETED — Science Done)

| Model | Config | Step | det=True | Notes |
|-------|--------|------|----------|-------|
| PPO_103 | aux_lr=5e-4, epochs=4, SEED=103 | 946K | SINGLE_SCRIPT (0.45pt, 231 updates) | 16px aux. Policy collapsed FASTER than aux could shape features. 10× gradient vs PPO_102. |

### Experiment 13: One-Life Breakout (Hard Failure) — PPO_104 (ACTIVE)

| Model | Config | Step | det=True | Notes |
|-------|--------|------|----------|-------|
| PPO_104 | OneLifeWrapper, frameskip=1, SEED=104 | 1M | SINGLE_SCRIPT (3.0 avg, 1 unique) | Scripts score 3pts on one ball. Worse than 5-life scripts but still a viable local optimum. Hard failure alone insufficient for Breakout. |

### Experiment 14: Adversarial Breakout — PPO_105 (COMPLETED — SINGLE_SCRIPT)

| Model | Config | Step | det=True | Notes |
|-------|--------|------|----------|-------|
| PPO_105 | AdversarialBallWrapper (strength=2.5, zone=140), SEED=105 | 10M | SINGLE_SCRIPT 3-13pt | Constant push insufficient. SINGLE_SCRIPT at every checkpoint. |

**Design:** Ball heading downward + below paddle zone → horizontal push away from paddle. Paddle tracks ball → push≈0, normal physics. Paddle doesn't track → ball dodges, no hits. Scripts are directly punished. Eval/check uses standard Breakout (no adversarial wrapper) to test transfer.

**Why it failed:** Deterministic wrappers preserve memorizability. Push = f(ball_x, paddle_x), both determined by action sequence. Fixed actions → fixed push trajectory → fixed score. PPO finds scripts that minimize expected push.

### Experiment 15: Adversarial Breakout (Proportional) — PPO_106 v1/v2/v3 (COMPLETED — SINGLE_SCRIPT)

| Version | Config | Step | det=True | Notes |
|---------|--------|------|----------|-------|
| v1 | fs=1, constant push ±2.5 | 6M | 0pt dead | fs=1 amplifies push 4× |
| v2 | fs=1, proportional, max_push=15→4 | 3M | 0pt dead | Unplayable at fs=1 |
| v3 | fs=4, proportional, max_push=3 | 9M+ | SINGLE_SCRIPT 0-12pt | Same pattern as PPO_105 |

**Design:** Proportional push with dead zone: push = sign(error) × min((|error| − dead_zone) × gain, max_push). Creates learnable gradient — track better → less push → more reward. Calibration: perfect=12, scripts=0-1 at max_push=3.

**Why it failed:** Same fundamental issue — deterministic function of instantaneous state. PPO finds action sequences that are more sophisticated than calibration strategies. Learned scripts keep paddle close enough to ball to minimize push.

### Experiment 16: Adversarial Cursor (Visible Agent) — PPO_107 (ACTIVE)

| Model | Config | Step | det=True | Notes |
|-------|--------|------|----------|-------|
| PPO_107 | AdversarialCursorWrapper (approach=2, threat=8, warn=5f, push=4, cooldown=60f), SEED=107 | 4M / 50M | SINGLE_SCRIPT 20pt | det=False: 8 unique, avg 17.9. Best adversarial script yet (PPO_105: 13pt, PPO_106: 12pt). |

**Design:** Visible cursor with state machine — the first "secondary agent" in BreakoutBot. Cursor approaches ball when paddle isn't tracking, pulses as warning (5 frames), attacks by pushing ball away. Paddle tracks → cursor retreats and hides. Cursor only visible during THREATENING/ATTACK states — in eval (standard Breakout), no cursor appears. Calibration gap: 12.1 (strongest in project).

**Key innovation:** VISIBLE ENTITY WITH AGENCY. Not a deterministic function of instantaneous state — the cursor has memory (state machine), movement (approaches/retreats at finite speed), and visibility (appears before acting). This mirrors BeamRider's structure: visible enemy → threat → react → survive.

**At 4M:** SINGLE_SCRIPT but the script scores 20pts on clean eval — 54% higher than PPO_105 (13pt) and 67% higher than PPO_106 (12pt). det=False maintains diversity (8 unique). Pattern is recognizable: argmax-script + policy-entropy, same as PPO_105/106. Early days.

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

### Multi-Environment Replication (July 28, 2026)

6. **SINGLE_SCRIPT is a general PPO property, not Breakout-specific.** 4/5 Atari games tested produce SINGLE_SCRIPT. Only BeamRider (hard failure) produces MULTIPLE_SCRIPTS.

7. **Stochastic elements that don't kill the agent don't prevent memorization.** Space Invaders' random UFO timing adds score variance (some games hit UFO, some don't) but doesn't require a reactive firing pattern.

8. **Sparse-reward games may never escape the zero-score attractor.** Freeway: 0pt, never learned to move the chicken. The no-op local optimum dominates.

### Experiment 15: Adversarial Breakout (July 28, 2026)

16. **Constant push at fs=4 (PPO_105) degrades scripts but doesn't prevent them.** SINGLE_SCRIPT 3-13pt at every checkpoint 1M→10M. Effective push of 0.625 px/frame is too subtle.

17. **Proportional push at fs=4, max_push=3 (PPO_106 v3) also SINGLE_SCRIPT.** SINGLE_SCRIPT 0-12pt at every checkpoint 1M→9M. PPO finds action sequences that work around the push — learned scripts are more sophisticated than calibration strategies.

18. **frameskip=1 is structurally incompatible with per-frame push.** Any push applied every ALE frame is amplified 4× vs fs=4. Even 1px/frame push kills perfect tracking. Both v1 (constant) and v2 (proportional) died at 0pt.

19. **Calibration framework works for param selection but underestimates PPO.** Simple strategies (sweep, center-hold, edge-camp) score 0-1 with adversarial push. But PPO finds optimized scripts that calibration doesn't cover.

20. **Deterministic wrappers preserve memorizability.** Push is f(ball_x, paddle_x), both determined by action sequence. Fixed actions → fixed push trajectory → fixed score. Any deterministic environment modification can be memorized.

21. **setRAM teleportation looks unnatural.** Modifying ball position changes position but not velocity. Ball snaps to new location, natural velocity carries it back → zig-zag. Ball moves 1px/frame naturally; 3px push is 3× natural speed.

22. **Direction control (RAM[105]) is the next avenue.** Modifying ball direction instead of position produces natural curves. Initial "dodge" test (force direction away from paddle) over-penalizes tracking — needs tuning. Unclear if stochastic/visible threat is required.

### Open Design Questions (Pending)

- **Visible vs invisible threat:** Would making the adversarial push visible in the observation help PPO learn to track?
- **Stochastic vs deterministic:** Would adding noise (probability of push, magnitude jitter) break the memorization attractor?
- **Ball speed modification:** Can we find and modify the ball speed RAM address? Only probed level 1 — speed-up in later levels untested.

### Running Now (July 28)

| Run | Config | Progress | Status |
|------|--------|----------|--------|
| PPO_107 | Exp 16 — Adversarial Cursor (visible agent) | ~4M / 50M | SINGLE_SCRIPT det=True, 20pt script, det=False diverse (8 unique) |

### Recently Completed

| Run | Result |
|-----|--------|
| PPO_104 | Exp 13 — COMPLETED: SINGLE_SCRIPT 1-3pt. One-life doesn't force reactivity. |
| PPO_105 | Exp 14 — COMPLETED: SINGLE_SCRIPT 3-13pt. Constant push insufficient. |
| PPO_106 v1 | Exp 15a — KILLED: 0pt dead. fs=1 constant push → error amplification. |
| PPO_106 v2 | Exp 15b — KILLED: 0pt dead. fs=1 proportional push → unplayable. |
| PPO_106 v3 | Exp 15c — COMPLETED: SINGLE_SCRIPT 0-12pt. Scripts adapt to proportional push. |
| BEAMRIDER_BASELINE | Paired experiment — COMPLETED: MULTIPLE_SCRIPTS 1M→10M, all tracked |
| BEAMRIDER_MULTILIFE | Soft failure — COMPLETED: MULTIPLE_SCRIPTS 1M→10M. Hard failure FALSIFIED. |
