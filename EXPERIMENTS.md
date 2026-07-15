# Experiments

**SUMMARY — Three experiments testing sticky actions in Breakout RL:**
- **Experiment 1 (COMPLETE — key finding overturned 2026-07-14):** PPO_25 (no sticky) vs PPO_26 (deep pretraining + sticky) vs PPO_27 (fresh + sticky). **PPO_26 CONFIRMED MEMORIZED by nosticky verification (2026-07-14).** Without sticky actions: Game 1 = 67 pts, Games 2+ = every game exactly 60.0 points, 264 frames — a fixed script. PPO_26's sticky-on performance (avg 54.3, 0% zero-score) was noise-masked memorization, identical to PPO_30b/PPO_31b. **The "both ingredients" recipe does NOT produce generalization — it produces better memorized scripts.** PPO_27 (sticky from scratch, p=0.25) remains the only model trained with stochasticity throughout — and it was the worst single-env performer. **Revised conclusion:** non-sticky pretraining causes permanent memorization that no amount of sticky fine-tuning has ever cured. Deep pretraining produces higher-scoring scripts (PPO_26: 60 pts > PPO_31b: 31 pts > PPO_30b: 0 pts) but never reactive policies.
- **Experiment 2 (COMPLETE):** PPO_28/29 — removed sticky actions from trained models. **Finding:** both collapsed to fixed open-loop action sequences within ~30M steps. Training metrics (EV, value_loss, entropy) lied during collapse. Sticky actions are required at inference time, not just during training.
- **Experiment 3 (COMPLETE — conclusions overturned by post-hoc analysis):** PPO_30/31 — non-sticky pretraining duration sweep at 400M total budget. **Both models CONFIRMED MEMORIZED.** The Phase 2 "GENERALIZING" verdicts were sticky-action noise — calibration shows a dead policy + p=0.25 sticky produces 8-14 unique scores, matching the observed 10-19 range. Nosticky verification confirms both collapse to ≤2 unique scores. PPO_30b: 2 unique (99.8% zeros). PPO_31b: 2 unique (all 31-point script). The "trade-off" is between which memorized script each learned, not between generalization quality. See Post-Hoc Analysis section below.

**BREAKTHROUGH (2026-07-15): PPO_35 — the first non-memorized model in the project.** Continuous mid-game physics changes (Experiment 5C) succeeded where sticky actions, frame skip randomization, and per-episode physics randomization all failed. At 64M steps: 21+ unique eval scores, explained_variance=0.85 (vs 0.93-0.96 for memorized models), eval scores cycling through genuinely different strategies. First evidence that a PPO agent can learn reactive ball-tracking in Breakout.

**Currently training:** PPO_32 (Experiment 4, p=0.05 sticky, 400M target, memorized with boom-bust cycles) and PPO_35 (Experiment 5C, continuous physics randomization, 400M target). See Experiment 5 section below for the full dependency chain that led here.

## ⚠️ Known Limitations

This document reports results from experiments with known methodological flaws. A comprehensive audit is maintained in `FLAWS.md`. **Before citing any conclusion from this document, read the relevant flaw entries.** The most consequential active limitations:

- **F-001 (CONFIRMED):** The MemorizationCheckCallback "GENERALIZING" verdict is INVALID for sticky-action models. Calibration: dead policy + p=0.25 sticky = 8-14 unique scores (mean 11.3). At p=0.05: 55-63 unique scores. The verdict measures sticky probability, not policy quality. Nosticky verification is the only reliable behavioral test.
- **F-002:** In Experiment 3, pretraining duration and sticky-step count are perfectly anti-correlated at the 400M total budget. Cannot attribute outcomes to either variable independently.
- **F-003 (RESOLVED 2026-07-14):** PPO_26 CONFIRMED MEMORIZED by nosticky verification. Without sticky actions: every game exactly 60.0 points, 264 frames — a fixed script. PPO_26's sticky-on dominance was a better memorized script + noise, not generalization. No model in this project has ever genuinely generalized.
- **F-004 (RESOLVED):** PPO_31b's 10k-game evaluation was completed (10,000 games as of 2026-07-14). Stats: avg 22.2, median 20, 2.4% zero-score.
- **F-010:** Funnel rate comparisons (0-7 events in 10,000 trials) are not statistically significant. Treat funnel rate rankings as directional indicators, not settled differences.

These and other limitations (confounded LR restart values, unmatched n_envs, mid-training evaluations, missing sticky-off verification, etc.) are documented with severity ratings and recommended remedies in `FLAWS.md`.

## Experiment 1: Sticky Actions and Training Regime — PPO_26 vs PPO_27

> **Status update — Experiment 1 COMPLETE.** PPO_26 finished at 1,001,828,352 total steps (eval peak 134.16 @ 905.6M). PPO_27 is still training and holds the all-time eval record (147.02 @ 867.2M). But the full 10,000-game single-env funnel comparison is now in for **all three models**, and it tells the opposite story from eval score: **PPO_26 wins every single-env metric; PPO_27 is the weakest single-env performer of the three**, with a zero-score rate (21.27%) statistically matching PPO_25's (20.0%) despite having sticky actions. See the updated Results and Conclusions below — this is the central finding of the experiment.

### Background

After PPO_25 reached 1 billion steps and an eval record of 140.94, two behavioral problems remained unresolved:

- **Dead balls:** After losing a life, the agent would frequently let subsequent balls pass through untouched, suggesting it had memorized paddle positions rather than learned to track the ball in real time.
- **Inconsistent tunnel exploitation:** The agent discovered the tunnel strategy (drilling through the brick wall to trap the ball behind it) emergently, but only completed it in roughly 3-5% of single-env games. It consistently carved multiple partial tunnels simultaneously rather than committing to one.

Both problems pointed to the same root cause: the agent was memorizing sequences of positions rather than developing a reactive ball-tracking policy. This is a known failure mode in Atari RL documented by Machado et al. (2018), who proposed **sticky actions** (`repeat_action_probability=0.25`) as the standard mitigation. With sticky actions enabled, 25% of steps repeat the previous action regardless of the agent's choice, making pure memorization unreliable and forcing more adaptive behavior.

The question was whether to apply sticky actions to the existing trained weights or start fresh. To answer this, two runs were launched simultaneously:

- **PPO_26:** Started from PPO_25's best_model (the 838M step checkpoint that achieved 140.94 eval), with sticky actions added. 64 parallel environments, same hyperparameters as PPO_25.
- **PPO_27:** Started from random initialization with sticky actions enabled from step one. 32 parallel environments (reduced to share hardware with PPO_26), otherwise same hyperparameters.

Both used `repeat_action_probability=0.25` in both training and eval environments.

---

### Configurations

**PPO_26 (train.py)**
```python
env = make_atari_env("ALE/Breakout-v5", n_envs=64, seed=None,
                     env_kwargs={"repeat_action_probability": 0.25})
# Started from PPO_25 best_model with reset_num_timesteps=True
model = PPO.load(PPO25_BEST_MODEL_PATH, env=env, device="cuda",
                 custom_objects={"n_envs": 64})
model.learning_rate = linear_schedule(2.5e-4, 1e-5)
model.clip_range = linear_schedule(0.2, 0.05)
model.ent_coef = 0.006
```

**PPO_27 (train_ppo27.py)**
```python
env = make_atari_env("ALE/Breakout-v5", n_envs=32, seed=None,
                     env_kwargs={"repeat_action_probability": 0.25})
# Fresh agent — no inherited weights
model = PPO(
    "CnnPolicy", env,
    n_steps=128, batch_size=1024, n_epochs=4, gamma=0.99,
    learning_rate=linear_schedule(2.5e-4, 1e-5),
    clip_range=linear_schedule(0.2, 0.05),
    ent_coef=0.006, vf_coef=0.5,
)
```

---

### Results

#### Eval Score Trajectory

| Milestone | PPO_26 (inherited weights) | PPO_27 (fresh agent) |
|-----------|---------------------------|----------------------|
| First eval above 80 | ~22M steps | ~129M steps |
| First eval above 90 | ~131M steps | ~169M steps |
| First eval above 100 | 304M steps | 299M steps |
| All-time eval peak | 134.16 @ 905.6M steps (run complete) | **147.02 @ 867.2M steps** 🏆 — new all-time record, surpasses PPO_25's 140.94 (run ongoing) |

PPO_26's inherited weights gave it a significant early advantage — it hit 80+ within 22M steps while PPO_27 took 129M. PPO_27 caught up and both crossed 100 at nearly identical step counts (~300M). **Since then, PPO_27 has pulled ahead decisively on eval score** — its 147.02 peak beats both PPO_26's final 134.16 and PPO_25's original 140.94. Note, however, that eval score and single-env score tell different stories here — see the updated Single-Env Watch Performance section below, where PPO_26 currently leads.

#### Convergence Speed

At equivalent wall-clock training time (~319,000 seconds elapsed for both), the runs stood at:
- PPO_26: 317M steps, rollout mean ~81
- PPO_27: 236M steps, rollout mean ~85

PPO_27 was producing higher rollout means despite fewer steps, because 32 envs run faster per step than 64. Normalizing by steps rather than time, PPO_26 held a small advantage in peak scores at that point — but this has since reversed (see above).

#### Training Stability

PPO_26 showed significantly wider oscillation throughout, with eval scores swinging between 47 and 113 in the 250-400M step range. In the most recent logged batch (889.6M–947.2M), PPO_26 continued this pattern, swinging from 71.50 to 134.16 — a wide range that includes both its new personal record and its batch low arriving just a few checkpoints apart. Notably, this batch **ends on a downward note** (71.50 at the final logged step), which is worth monitoring for a real regression vs. ordinary exploration noise.

PPO_27's oscillation has remained tighter and more consistent by comparison. In its most recent logged batch (798.4M–856.0M), PPO_27 averaged 112.3 against PPO_26's 104.2 over a comparable batch — and PPO_27's lowest score in that batch (88.34) is still well above PPO_26's batch low (71.50). This is visible in the approx_kl values from earlier in the runs — PPO_27 consistently showed lower KL divergence (0.007-0.020) vs PPO_26 (0.025-0.090).

#### Single-Env Watch Performance — Matched 10,000-Game Funnel Runs (Complete — All Three Models)

The original single-env comparison table (snapshot, small samples, inconsistent thresholds between the writeup's prose and its own table) has been fully superseded. All three models have now been run for the full 10,000 games using matched scripts (`funnel_recorder_ppo_25.py` / `_ppo_26.py` / `_ppo_27.py`), identical funnel threshold (400+), identical sticky-action config per run, and identical scoring methodology.

| Metric | PPO_25 (no sticky, deep) | PPO_26 (sticky + deep) | PPO_27 (sticky, fresh) |
|--------|--------------------------|--------------------------|--------------------------|
| Average score | 34.6 | **54.3** 🏆 | 27.95 (worst) |
| Median score | 30.0 | **46.0** 🏆 | 23.0 (worst) |
| Std dev | 39.5 | 46.4 | 32.2 |
| Min score | 0 | **5** 🏆 | 0 |
| Zero-score games | 1,998 (20.0%) | **0 (0.0%)** 🏆 | 2,127 (21.27%, worst) |
| Best score | 406 | **415** 🏆 | 406 |
| Funnel rate (400+) | 2/10,000 (0.02%) | **7/10,000 (0.07%)** 🏆 | 1/10,000 (0.01%, worst) |

**PPO_26 wins outright on every single-env metric. PPO_27 — the all-time eval-score record holder — is the worst performer of the three on every single-env metric.** This is the most important result of the whole experiment, and it overturns the eval-score-based narrative built up over the course of this writeup.

**The "needs both ingredients" hypothesis is now confirmed.** Earlier observation (informal, pre-data): PPO_26 "seldom" produced zero-score games while PPO_27 "tends to have a lot of zeros." This is now precisely quantified — PPO_27's zero-score rate (21.27%) is statistically indistinguishable from PPO_25's (20.0%), despite PPO_27 having sticky actions and PPO_25 not. **Sticky actions alone do not fix the zero-score blind spot.** Only PPO_26 — which has both sticky actions *and* PPO_25's billion steps of inherited prior training — eliminates it completely (0.0%). The deep prior training appears to be doing the real work here, with sticky actions only able to leverage it once it's present. A fresh agent trained with sticky actions from step one (PPO_27) gets neither benefit.

**Eval score and single-env score aren't just different rankings here — they're inverted.** PPO_27 holds the all-time eval record (147.02) and has the worst single-env quality of the three models by every measure. PPO_26 has a middling eval peak (134.16, lower than both PPO_25's 140.94 and PPO_27's 147.02) and the best single-env quality by a clear margin. Whatever PPO_27's policy is doing well under parallel 50-episode eval sampling is not the same thing as being good at single-env sequential play — see RL_REFERENCE.md Lesson #23 (now significantly strengthened by this result) and the "floundering" instinct that prompted running this funnel test early: PPO_27's eval score had also plateaued (mean dropped from 111.7 in one quarter to 107.5 in the next, no new record in 60 evals) right around the same time this single-env weakness was confirmed. The two observations are consistent with the same underlying story — PPO_27 may be overfitting to the eval procedure rather than developing genuinely robust play.

---



### Investigation: The "Quick Death" Cluster and Direction-Correctness (Side Investigation)

Following up on PPO_26's much milder version of the zero-score pattern (no literal zeros, but a real cluster of very-low-score games with short, similar frame counts), a hunting script (`watch_quickdeath_ppo26.py`) was built to specifically capture games matching that signature, with per-frame RAM tracing (paddle x, ball x, ball y) and the raw action chosen each frame.

**Methodology note — RAM coordinate convention correction.** Initial analysis assumed `paddle_x`/`ball_x` of 0 and 191 meant "left wall" and "right wall" respectively. Empirical testing across the captured traces showed this was backwards relative to the action labels: `LEFT` consistently increases `paddle_x` (139 increases vs. 11 decreases across the sample) and `RIGHT` consistently decreases it. This doesn't indicate a bug — it just means the RAM convention runs opposite to naive intuition, and all directional analysis was corrected to account for it.

**Hypothesis tested:** that quick-death games show the paddle choosing the geometrically wrong direction (relative to the ball) more often than ordinary play — i.e., a "panic" effect.

**Result: not supported.** A direction-correctness metric (does the chosen LEFT/RIGHT action move the paddle toward the ball, given its pre-action position) was computed for the 6 captured quick-death games and for 40 unfiltered control games (`watch_controltrace_ppo26.py`):

| Sample | Direction-correct rate |
|--------|------------------------|
| Quick-death cluster (6 games) | 61.3% |
| Control — unfiltered (40 games) | 55.3% |
| Control — low score (<20 pts) | 57.1% |
| Control — mid score (20-49 pts) | 55.0% |
| Control — high score (50+ pts) | 55.2% |

Direction-correctness sits flat around 55% regardless of how well the game went, and the quick-death cluster actually scored *higher* on this metric than the control baseline — the opposite of the panic hypothesis. A velocity-aware version of the metric (predicting where the ball is heading rather than just its current position) was also tested and produced essentially the same result (54.6%), ruling out "the metric ignores velocity" as the explanation.

**Conclusion:** this specific diagnostic is inconclusive and not worth refining further. Whatever causes the milder quick-death cluster in PPO_26, it isn't captured by frame-level instantaneous direction-correctness. Given that PPO_26's actual headline result (zero zero-score games, better-than-PPO_25 on every other axis) is already a strong, settled finding independent of this investigation, this thread is closed for now rather than pursued further.

**Open question raised during this investigation, not yet tested:** would disabling sticky actions (`repeat_action_probability=0.0`) after training with them on for a substantial period improve precision without reintroducing the original positional-memorization problem? This is genuinely untestable from existing data — it requires running the experiment. It maps directly onto **Option B** in the Planned Next Steps below, and this investigation is a (weak, inconclusive) point in favor of trying it, since execution noise from sticky actions remains a plausible contributor to imprecision even though we couldn't isolate it cleanly here.

---

### Conclusions

**1. Inherited weights with sticky actions show diminishing returns on eval score, but win decisively on single-env quality.**
PPO_26 recovered quickly early on (reaching 80+ in 22M steps) but this eval-score advantage eroded over training, and PPO_27 went on to set the all-time eval record. However, on the complete single-env matched-script comparison, PPO_26 is unambiguously the strongest model of the three — highest average, highest median, highest floor, most funnels, and the only one to fully eliminate the zero-score blind spot. Eval score and single-env score are not telling the same story here — see Lesson #23.

**2. Fresh training with sticky actions is sample-efficient for eval score — but this is now confirmed NOT to produce a genuinely strong model.**
PPO_27 broke 100 at 299M steps and went on to set the all-time eval record (147.02). But the completed single-env comparison shows PPO_27 is the *weakest* of the three models on every real-gameplay metric, including a zero-score rate (21.27%) statistically indistinguishable from PPO_25's. **The original framing of this as "fresh + sticky resolves the open question in favor of starting fresh" no longer holds** — fresh-start sticky-action training optimizes for something eval score measures well and single-env play doesn't, or vice versa. This is the central reversal of this experiment.

**3. Sticky actions produce more stable eval training dynamics, independent of final single-env quality.**
PPO_27 showed consistently lower KL divergence, tighter eval oscillation, and more predictable improvement curves than PPO_26 during training — that observation stands. But stable eval-training dynamics did not translate into a stable, competent single-env policy. Training-time stability and final-policy quality are separate things.

**4. SUPERSEDED — Sticky actions do not trade funnel rate for general consistency; the real driver is inherited depth, not stickiness itself.**
*(Original conclusion, kept for the record: "Sticky-action training improves general play consistency but reduces tunnel exploitation rate." This was based on a small, inconsistently-thresholded snapshot.)* The complete data shows sticky actions alone (PPO_27) don't reliably improve anything on single-env metrics relative to no-sticky (PPO_25) — average score is actually lower, zero-score rate is statistically the same, funnel rate is lower. The real driver of PPO_26's strong single-env performance appears to be the combination of sticky actions *with* deep inherited training, not sticky actions on their own. See **Lesson #22** in RL_REFERENCE.md ("needs both ingredients," now confirmed) for the full reasoning.

**5. Eval score and single-env quality are inverted across these three models, not just "different."**
PPO_27 holds the eval-score record (147.02) and has the worst single-env quality. PPO_26 has a middling eval peak (134.16) and the best single-env quality. PPO_25 sits in between on eval score and in between on most single-env metrics too. Whatever the parallel 50-episode eval procedure is measuring, it is not a reliable proxy for single-env, sequential, real-gameplay competence — at least not across this specific set of training configurations. This is likely the single most important methodological finding from this entire experiment for future runs (including PPO_28): **eval score alone should not be used to pick a "best" checkpoint without single-env verification.**

---

### Status

**Done — Experiment 1 complete.** All three models (PPO_25, PPO_26, PPO_27) have full 10,000-game matched-script single-env funnel baselines. PPO_26 wins on every single-env metric; PPO_27 (the eval-score record holder) is the weakest single-env performer of the three. The "needs both ingredients" hypothesis (sticky actions + deep inherited training, not either alone) is confirmed. Quick-death/direction-correctness side investigation closed (inconclusive).

---

### Caveats

This is not a perfectly controlled experiment. Confounding variables include:

- **Environment count:** PPO_26 used 64 envs, PPO_27 used 32. This affects both training dynamics and the effective batch size despite batch_size being scaled proportionally.
- **Starting point:** PPO_26 inherited not just PPO_25's weights but also its optimizer state and learning rate schedule, which were reset. The schedule restart may have affected early training dynamics.
- **Hardware sharing:** Both runs trained simultaneously on the same GPU, with each receiving roughly half the available compute. Neither was running at full speed.
- **System restarts:** Multiple unplanned interruptions throughout both runs may have affected continuity, including a recurring bug (now fixed) where `total_timesteps` was being treated as "train N more steps" on every restart rather than an absolute target.
- **Small quick-death sample:** The direction-correctness investigation above used only 6 quick-death games against 40 control games — both are small samples, and the inconclusive result could simply reflect insufficient data rather than a genuinely flat relationship.

Despite these limitations, the directional conclusions on training stability and single-env quality are consistent across multiple metrics and observation methods.

---



## Option D (Considered and Rejected): Process-Based Reward via Paddle Bounces + Episode Length

Raised as an alternative to score-based reward, on the reasoning that human skill training often benefits from rewarding correct *process* over a graded *outcome* metric (e.g. math pedagogy rewarding correct steps over just the final answer). The proposed implementation: reward successful paddle bounces and episode length instead of score, on the theory that these are "fundamental skills" that can't be metagamed.

**Rejected for two independent reasons:**

1. **Already tried, twice, and failed both times.** PPO_15 rewarded ball-tracking (a process proxy similar in spirit) and the agent learned to mirror the ball without scoring. PPO_18 rewarded paddle hits directly and collapsed to an eval score of 19.0. This is the same failure mode recurring, not new territory.
2. **Structurally unsound, not just under-tuned.** Both proposed metrics (bounces, episode length) can be maximized by a degenerate strategy that never engages with the actual task: bounce the ball straight up at 0°, park the paddle beneath it, and let it loop against the ceiling indefinitely. This clears a column or two of bricks incidentally on the way to settling into the loop, then accumulates large reward from survival time alone while doing nothing further toward the actual goal. There's no equivalent shortcut available for "bricks removed" — see commentary below.

**Resolved clarification — the project has actually been running a process-style reward all along.** `ClipRewardEnv` flattens every brick hit to exactly 1.0 regardless of point value, so every run (PPO_5–PPO_27) has trained on "bricks removed," not raw score — raw score has only ever been an eval/comparison readout, never the training signal. This is a useful contrast case: "bricks removed" is a process metric that has no cheap degenerate maximum (you cannot rack up brick-clear reward without clearing bricks), while "bounces + survival time" does. The lesson isn't "process rewards don't work" — it's that a process proxy is only safe if there's no way to satisfy it without doing the real task. Full writeup added to RL_REFERENCE.md Part 3 and Key Lessons Learned #22.

---

## Planned Next Steps

**Experiment 1 complete. Experiment 2 complete (see below). The non-sticky-pretraining-duration sweep (item 6 below) is the natural next experiment.**

| # | Investigation | Cost | Status |
|---|---------------|------|--------|
| 1 | Diagnose PPO_26's training dips | Free | Partially done. Lower priority. |
| 2 | Verify whether sticky actions fixed the dead-ball problem | Free | **Done** — requires both ingredients, not sticky alone. |
| 3 | Sticky-action intensity sweep | Low | Deprioritized. |
| 4 | Revisit `net=[512,512]` | Medium | Not started. Lower priority. |
| 5 | Two-phase sticky-then-off | Medium | **Done as Experiment 2 — confirmed memorization collapse. Closed.** |
| 6 | **(Top priority)** Non-sticky-pretraining-duration sweep: train fresh agents with varying non-sticky phase lengths (e.g. 100M / 300M / 500M) before switching to sticky actions, and measure single-env zero-score rate at each. Tests how much non-sticky pretraining is actually needed to get PPO_26-like results. | Medium-High | Not started. |

**Deferred:** funnel-commitment reward shaping — still carries the most Goodhart risk.

---

## Experiment 2: Sticky-Then-Off — PPO_28 and PPO_29

### Background

Experiment 1 established that PPO_26 (deep non-sticky pretraining + sticky actions) was the strongest model on single-env metrics, and that the "needs both ingredients" hypothesis was confirmed — sticky actions alone (PPO_27) didn't fix the zero-score blind spot. The natural follow-up was Option B from the original plan: what happens if you take a well-trained sticky-action model and *remove* stickiness for a final phase?

Two runs were launched using the OpenRouter/DeepSeek external toolchain (same SB3/PyTorch stack, different training orchestration):

- **PPO_28:** continued from PPO_26's best_model with `repeat_action_probability=0.0`
- **PPO_29:** continued from PPO_27's best_model with `repeat_action_probability=0.0`

Both used 32 parallel envs, linear LR schedule **restarted from 2.5e-4→1e-5**, and the absolute-step-count fix (`remaining = TARGET - model.num_timesteps`).

---

### Configurations

**PPO_28**
```python
env = make_atari_env("ALE/Breakout-v5", n_envs=32, seed=None,
                     env_kwargs={"repeat_action_probability": 0.0})
# Loaded from PPO_26 best_model; LR schedule restarted
model.learning_rate = linear_schedule(2.5e-4, 1e-5)
```

**PPO_29**
```python
env = make_atari_env("ALE/Breakout-v5", n_envs=32, seed=None,
                     env_kwargs={"repeat_action_probability": 0.0})
# Loaded from PPO_27 best_model; LR schedule restarted
model.learning_rate = linear_schedule(2.5e-4, 1e-5)
```

---

### Results

#### Training Metrics — PPO_29 (the spectacular-looking collapse)

Within ~30M steps of removing stickiness, PPO_29's training metrics transformed dramatically:

| Metric | PPO_27 (sticky on) | PPO_29 (sticky off, ~1,007M steps) |
|--------|-------------------|-------------------------------------|
| `ep_rew_mean` | 107-110 | **409-410** |
| `ep_len_mean` | ~1,740 | ~2,420 |
| `explained_variance` | 0.976-0.999 | **1.000** |
| `value_loss` | 0.852-3.42 | **0.001** |
| `entropy_loss` | -0.239 to -0.510 | -0.142 to -0.301 |
| `loss` | 0.383-1.630 | **negative** |

Eval scores jumped from ~110 to 418.88 in fewer than 2M steps — roughly 400 gradient updates. This is pathologically fast. The negative total loss (dominated by entropy regularization when policy and value losses are both near zero) and `explained_variance=1.0` are the definitive signatures of complete convergence to a fixed deterministic solution.

#### Training Metrics — PPO_28 (the slower collapse)

PPO_28's metrics looked healthier throughout: entropy -0.42 to -0.55, explained_variance 0.76-0.92, value_loss 1.3-3.9, ep_rew_mean declining from ~256 to ~224. This appeared to be evidence of genuine ongoing learning rather than memorization — but see the behavioral confirmation below.

PPO_28 eval scores: started at 340 at 968M steps, peaked at 366.66 at 969.6M, then oscillated wildly (127-366 range, stdev 64.5) with no stable upward trend. Mean across all 18 recorded evals: 297.8.

#### Behavioral Confirmation — The Diagnostic Test

A side-by-side diagnostic (`diagnostic_ppo28.py`) ran 20 games each in two modes:

**Approach A:** `seed=None`, single persistent env, plain `env.reset()` between games (the same approach used for PPO_25/26/27's 10,000-game funnel runs).

**Approach B:** fresh env created per game with an explicit random seed (thought to guarantee varied starting conditions — see seeding investigation below).

| Approach | PPO_28 scores | PPO_29 scores |
|----------|---------------|---------------|
| A (persistent env) | 104, then 60×19 — 2 unique scores | Not tested; already confirmed below |
| B (fresh env, explicit seed) | 104×20 — 1 unique score | 355×40 — 1 unique score |

Both models produce a fixed score from every starting condition across both testing approaches. PPO_29 plays a universal 395-frame, 355-point script from any seed. PPO_28 plays a 383-frame, 104-point script (Approach B) or a 383-frame, 104-point script followed by a 60-point script when the ALE carries internal state (Approach A). These are fixed open-loop action sequences, not policies that respond to game state.

**The `deterministic=False` test:** switching to stochastic sampling in `model.predict()` for PPO_29 produced identical results to `deterministic=True`. This is the definitive behavioral confirmation — if the action distribution had any meaningful spread, sampling would produce different actions. The policy entropy at inference is functionally zero even though training rollouts showed entropy_loss of -0.14 to -0.30 (see below for why).

---

### Why the Metrics Lied

**Why PPO_29's training metrics looked spectacular:** `ep_rew_mean=409` means ~409 bricks cleared per episode — nearly 4 complete board clears. This is achievable from a fixed action sequence if that sequence happens to keep the ball in play for a long time from the default ALE starting state. With 32 parallel envs all experiencing `seed=None`, if all 32 envs settled to the same deterministic trajectory (which a memorized policy drives them toward), the rollout statistics reflect 32 identical replays rather than diverse experience. Negative `loss` occurs when policy gradient and value losses are both near zero and the entropy regularization term dominates — a mathematical artifact of complete convergence, not a sign of good learning.

**Why PPO_28's training metrics looked healthy:** 32 envs each starting from a slightly different initial ALE state (due to env creation timing) may settle into *different* memorized loops, making the value function's prediction problem appear diverse when measured in aggregate. EV < 1 and real value_loss reflect genuine disagreement between the 32 different fixed trajectories, not genuine policy diversity within any single environment. The declining `ep_rew_mean` (256→224) is the model converging toward its own settled attractor rather than improving.

**Why `entropy_loss` didn't go to zero:** the -0.14 to -0.30 entropy reading in PPO_29's rollouts comes from the full 32×N_STEPS rollout batch, which includes transitions from mid-sequence states the policy encounters across all 32 envs. Some of these states may have broader action distributions even if the policy is functionally deterministic for the specific game-start states we tested. Training entropy and inference entropy are not the same measurement.

**Why eval scores also lied:** the eval callback uses `seed=None` with a single eval env running 50 episodes. Without sticky actions, a memorized policy drives the eval env into the same ALE state loop every episode after the first reset. All 50 eval episodes replay the same fixed sequence. The 418.88 mean is 50×355-point scripts (with score multipliers from high-value bricks not captured by ClipRewardEnv mapping everything to 1.0 per brick).

---

### The Seeding Investigation

A significant portion of this experiment's time was spent investigating what appeared to be a seeding bug but turned out to reveal something more fundamental.

**Initial observation:** the PPO_29 funnel recorder showed game 1 = 355 points, games 2+ = 86 points identically. This was first attributed to `env.seed()` not reaching the ALE through the VecEnv wrapper chain (VecFrameStack → VecTransposeImage → DummyVecEnv).

**Attempted fixes, in order:**
1. `env.seed(int)` — no error, but didn't change scores
2. `ale.setInt('random_seed', seed)` directly — changed game 1 to varied seeds but game 2+ still identical
3. Fresh env per game with `make_atari_env(seed=int)` — every game scored 355 from different seeds

The fresh-env approach with confirmed varied seeds still produced identical results, which established the actual cause: **PPO_29 plays a fixed action sequence regardless of the game state it encounters.** The seed doesn't matter because the policy isn't reading the pixels.

**Secondary finding about ALE seeding in Breakout:** the game's ball launch direction appears to be determined by internal ALE frame-timing state rather than the ALE random seed. The original PPO_25/26/27 funnel recorders produced varied game outcomes not because seeds varied but because: (a) sticky actions randomly overrode moves, creating different ALE end-states after each game; and (b) `seed=None` with `env.reset()` allowed natural ALE internal state accumulation between episodes. Passing explicit integer seeds to `make_atari_env` appears to reset this internal state to a fixed deterministic starting point regardless of the integer value, which is why the fresh-env approach eliminated the apparent variation it was intended to introduce.

This seeding behavior is specific to Breakout and may not generalize to other Atari games. It is now documented as a known project-specific constraint. For future evaluation scripts on models trained without sticky actions, the correct approach is `seed=None` with a single persistent env — not explicit seeding.

---

### Conclusions

**1. Removing sticky actions from any trained model in this project causes rapid memorization collapse.**
Both PPO_28 (PPO_26 lineage) and PPO_29 (PPO_27 lineage) converged to fixed open-loop action sequences within tens of millions of steps of stickiness removal. PPO_26's non-sticky pretraining foundation did not protect against this. The two-phase sticky-then-off recipe does not work for these models.

**2. Sticky actions are required at inference time, not just during training.**
The memorization collapse happens because removing stickiness creates a fully deterministic environment. A deterministic policy in a deterministic environment will always find a fixed-sequence solution faster than a general reactive policy, because the fixed sequence is a much simpler optimization target. Stickiness actively suppresses this attractor at every step during training — its role is ongoing, not just formative.

**3. Training metrics cannot be trusted to detect memorization collapse.**
`ep_rew_mean`, `explained_variance`, eval scores, and even `entropy_loss` all produced misleading readings during and after the collapse. The only reliable diagnostic was behavioral: playing the model against varied starting conditions and checking whether scores vary. `deterministic=False` producing the same results as `deterministic=True` is the earliest detectable signal.

**4. The spectacular eval numbers (418.88 for PPO_29, 340-366 for PPO_28) are artifacts, not records.**
These should not be logged as performance milestones. They reflect 50 repetitions of a memorized script against a fixed ALE default state, not generalized gameplay ability.

**5. The "needs both ingredients" conclusion from Experiment 1 is now understood more deeply.**
Sticky actions must remain on throughout the model's lifetime. The "both ingredients" are not just "non-sticky pretraining + sticky fine-tuning" — they are "non-sticky pretraining + sticky actions permanently." The sticky component is not a training phase; it is a permanent environmental constraint that prevents the policy from collapsing to memorization whenever it finds a fixed sequence that scores consistently.

---

### Status

**Experiment 2: COMPLETE. Both PPO_28 and PPO_29 confirmed as memorization artifacts. Runs terminated.**

---

## Experiment 3: Non-Sticky Pretraining Duration Sweep — PPO_30 and PPO_31

### Background

Experiments 1 and 2 together established two things: PPO_26's recipe (deep non-sticky pretraining followed by sticky-action training) produces the best single-env performance, and removing sticky actions from any trained model causes rapid memorization collapse. The mechanism behind PPO_26's success appears to be the non-sticky pretraining phase specifically — not just total step count, since PPO_27 accumulated similar steps entirely under sticky actions and still showed PPO_25-level failure.

The open question: **how much non-sticky pretraining is actually necessary?** PPO_25/26 used roughly 1 billion non-sticky steps before the switch. That's expensive. If 100M or 300M steps is sufficient to build the foundation, the two-phase recipe becomes far more practical as a general approach.

Two hypotheses being tested:

- **Hypothesis A (basic competency):** The agent just needs to learn ball tracking before sticky actions are introduced. Even 100M steps might be enough — PPO_22 achieved eval ~87 by 57M steps, suggesting the core skill develops quickly.
- **Hypothesis B (policy depth):** The foundation needs to be deep enough that sticky-action training can't easily overwrite it. 100M might fail while 400M or more succeeds.

### Configurations

Both experiments use 32 parallel envs, two-phase structure, and a conservative LR restart (1e-4 → 1e-5, not 2.5e-4) at the phase switch — to avoid the aggressive early updates that contributed to PPO_28/29's collapse.

| Run | Phase 1 | Phase 2 | Tests |
|-----|---------|---------|-------|
| PPO_30a | 100M steps, no sticky, fresh agent | — | Non-sticky baseline at 100M |
| PPO_30b | 300M steps, sticky on, loaded from PPO_30a | PPO_30a → PPO_30b | Hypothesis A — is 100M enough? |
| PPO_31a | 300M steps, no sticky, fresh agent | — | Non-sticky baseline at 300M |
| PPO_31b | 100M steps, sticky on, loaded from PPO_31a | PPO_31a → PPO_31b | Hypothesis B — does 300M beat 100M? |

Both runs cap at 400M total steps.

PPO_30a and PPO_31a run simultaneously (Phase 1 in parallel). PPO_30b starts as soon as PPO_30a completes (~4-8 hrs), without waiting for PPO_31a (~16-32 hrs).

**Key design difference from PPO_28/29:** the LR schedule at the Phase 1→2 switch restarts at 1e-4 (not 2.5e-4). This is deliberately conservative — high LR at phase transition was identified as a likely contributor to PPO_28/29's fast collapse.

**Preserved checkpoints:** `PPO_30a/final_model` and `PPO_31a/final_model` are saved at the phase transition and can be independently evaluated via funnel recorder to answer a bonus question: what does non-sticky performance look like at 100M vs 400M steps?

---

### Results

**Tooling note:** a `MemorizationCheckCallback` (see `memorization_check_callback.py`) was added to all four training scripts after this experiment began. It runs a 20-game in-memory check every 10M steps and appends to `{RUN_NAME}_memorization_track.csv`, so the trajectory is now visible incrementally rather than only at manual check-in points. PPO_30a and PPO_31a were restarted (from checkpoint, no progress lost) to pick up the callback.

#### Phase 1 — Early Memorization, Both Runs (confirmed, then tracked)

A manual check at ~11M steps (before the automated callback existed) found both PPO_30a and PPO_31a already collapsed to 1-2 repeated scores — concerning, since this happens in runs with sticky actions never having been present, unlike PPO_28/29 which collapsed *after* stickiness was removed from an already-trained policy. This raised the open question of whether non-sticky Breakout training reliably finds a fixed-script local optimum almost immediately and needs a long runway to escape it (possibly explaining why PPO_25 needed close to a billion steps to show real variance).

Automated tracking since then:

| Run | Step | Unique Scores | Avg | Best | Worst | Verdict |
|-----|------|---------------|-----|------|-------|---------|
| PPO_30a | 10,000,000 | 1 | 40.0 | 40.0 | 40.0 | MEMORIZED |
| PPO_30a | 20,000,000 | 2 | 40.2 | 44.0 | 40.0 | MEMORIZED |
| PPO_30a | 30,000,000 | 2 | 9.9 | 27.0 | 9.0 | MEMORIZED |
| PPO_30a | 40,000,000 | 1 | 27.0 | 27.0 | 27.0 | MEMORIZED |
| PPO_30a | 50,000,000 | 2 | 5.2 | 27.0 | 4.0 | MEMORIZED |
| PPO_30a | 60,000,000 | 2 | 98.2 | 102.0 | 98.0 | MEMORIZED |
| PPO_30a | 70,000,000 | 2 | 39.4 | 40.0 | 27.0 | MEMORIZED |
| PPO_30a | 80,000,000 | 2 | 9.6 | 40.0 | 8.0 | MEMORIZED |
| PPO_30a | 90,000,000 | 2 | 7.0 | 27.0 | 6.0 | MEMORIZED |
| PPO_30a | 100,000,000 ✅ | 2 | 98.9 | 102.0 | 40.0 | MEMORIZED |
| PPO_31a | 10,000,000 | 2 | 8.9 | 64.0 | 6.0 | MEMORIZED |
| PPO_31a | 20,000,000 | 2 | 10.4 | 76.0 | 7.0 | MEMORIZED |
| PPO_31a | 30,000,000 | 2 | 61.7 | 64.0 | 18.0 | MEMORIZED |
| PPO_31a | 40,000,000 | 2 | 3.8 | 18.0 | 3.0 | MEMORIZED |
| PPO_31a | 50,000,000 | 1 | 78.0 | 78.0 | 78.0 | MEMORIZED |
| PPO_31a | 60,000,000 | 1 | 64.0 | 64.0 | 64.0 | MEMORIZED |
| PPO_31a | 70,000,000 | 2 | 8.9 | 64.0 | 6.0 | MEMORIZED |
| PPO_31a | 80,000,000 | 1 | 64.0 | 64.0 | 64.0 | MEMORIZED |
| PPO_31a | 90,000,000 | 1 | 106.0 | 106.0 | 106.0 | MEMORIZED |
| PPO_31a | 100,000,000 | 2 | 20.6 | 31.0 | 20.0 | MEMORIZED |
| PPO_31a | 110,000,000 | 2 | 18.5 | 28.0 | 18.0 | MEMORIZED |
| PPO_31a | 120,000,000 | 2 | 8.8 | 25.0 | 8.0 | MEMORIZED |
| PPO_31a | 130,000,000 | 2 | 23.7 | 132.0 | 18.0 | MEMORIZED |
| PPO_31a | 140,000,000 | 2 | 77.8 | 81.0 | 18.0 | MEMORIZED |
| PPO_31a | 150,000,000 | 1 | 63.0 | 63.0 | 63.0 | MEMORIZED |
| PPO_31a | 160,000,000 | 1 | 23.0 | 23.0 | 23.0 | MEMORIZED |
| PPO_31a | 170,000,000 | 1 | 64.0 | 64.0 | 64.0 | MEMORIZED |
| PPO_31a | 180,000,000 | 2 | 39.9 | 41.0 | 18.0 | MEMORIZED |
| PPO_31a | 190,000,000 | 2 | 26.6 | 27.0 | 18.0 | MEMORIZED |
| PPO_31a | 200,000,000 | 2 | 21.1 | 81.0 | 18.0 | MEMORIZED |
| PPO_31a | 210,000,000 | 2 | 20.3 | 64.0 | 18.0 | MEMORIZED |
| PPO_31a | 220,000,000 | 1 | 63.0 | 63.0 | 63.0 | MEMORIZED |
| PPO_31a | 224,000,000 | 1 | 18.0 | 18.0 | 18.0 | MEMORIZED |
| PPO_31a | 234,000,000 | 1 | 64.0 | 64.0 | 64.0 | MEMORIZED |
| PPO_31a | 244,000,000 | 2 | 20.3 | 64.0 | 18.0 | MEMORIZED |
| PPO_31a | 254,000,000 | 2 | 20.3 | 64.0 | 18.0 | MEMORIZED |
| PPO_31a | 264,000,000 | 1 | 18.0 | 18.0 | 18.0 | MEMORIZED |
| PPO_31a | 274,000,000 | 2 | 20.3 | 64.0 | 18.0 | MEMORIZED |
| PPO_31a | 284,000,000 | 2 | 20.3 | 64.0 | 18.0 | MEMORIZED |
| PPO_31a | 294,000,000 | 2 | 20.3 | 64.0 | 18.0 | MEMORIZED |

**PPO_31a Phase 1 COMPLETE at ~298,000,000 steps.** The cycling pattern never broke — every single check across the full 300M steps showed MEMORIZED (1-2 unique scores). The policy settled into a stable attractor cycling between exactly two scripts: a 64-point script and an 18-point script (with rare 20.3-point variants). final_model saved at ~298M.

**PPO_30a Phase 1 COMPLETE at 100,002,816 steps.** The cycling pattern came full circle — the policy returned to its 102-point peak script right at the finish line (avg=98.9, best=102), after spending steps 65M-95M in crash/trough phases. Rollout at completion: ep_rew_mean=67-70, LR=1e-05, approx_kl=8e-6, clip_fraction=0.000183 — completely frozen. `best_model.zip` captures the 98-102 point quality. **PPO_30b launched from `PPO_30a/best_model`.**

#### Phase 2 — Sticky Actions Added

**PPO_30b memorization track (sticky on, target 400M total, LR restart at 1e-4→1e-5):**

| Run | Step | Unique Scores | Avg | Best | Worst | Verdict |
|-----|------|---------------|-----|------|-------|---------|
| PPO_30b | 100,002,848 | 10 | 6.8 | 16.0 | 0.0 | **GENERALIZING** |
| PPO_30b | 110,002,848 | 13 | 18.9 | 51.0 | 4.0 | **GENERALIZING** |
| PPO_30b | 120,002,848 | 15 | 20.9 | 44.0 | 1.0 | **GENERALIZING** |
| PPO_30b | 130,002,848 | 14 | 18.6 | 41.0 | 4.0 | **GENERALIZING** |
| PPO_30b | 140,002,848 | 17 | 22.1 | 48.0 | 3.0 | **GENERALIZING** |
| PPO_30b | 150,002,848 | 16 | 22.2 | 53.0 | 3.0 | **GENERALIZING** |
| PPO_30b | 157,602,848 | 14 | 25.4 | 111.0 | 4.0 | **GENERALIZING** |
| PPO_30b | 167,602,848 | 18 | 30.6 | 67.0 | 5.0 | **GENERALIZING** |
| PPO_30b | 177,602,848 | 17 | 28.8 | 54.0 | 6.0 | **GENERALIZING** |
| PPO_30b | 187,602,848 | 15 | 28.9 | 84.0 | 8.0 | **GENERALIZING** |
| PPO_30b | 197,602,848 | 19 | 33.2 | 69.0 | 4.0 | **GENERALIZING** |

**PPO_30b at 200M (100M sticky steps):** 🏆 **First non-sticky→sticky phase transition in this project to break memorization.** Within 10M steps of stickiness being added, the policy went from 2 unique scores to 10, and has maintained 14-19 unique scores across every check since — 100M sticky steps with zero relapses to MEMORIZED. The conservative LR restart (1e-4→1e-5) and the 100M non-sticky foundation together appear to be the winning combination. However, the single-env average score is flat at ~28-33 — the model is generalizing but not yet improving. Rollout `ep_rew_mean` (60-71) is substantially higher than the single-env memorization check average, suggesting the multi-env rollout captures better play than single-env sequential testing. 200M more sticky steps remain to convert generalization into score improvement.

**PPO_31b memorization track (sticky on, target 400M total, LR restart at 1e-4→1e-5):**

| Run | Step | Unique Scores | Avg | Best | Worst | Verdict |
|-----|------|---------------|-----|------|-------|---------|
| PPO_31b | 300,001,312 | 10 | 5.5 | 15.0 | 0.0 | **GENERALIZING** |
| PPO_31b | 310,001,312 | 13 | 14.7 | 31.0 | 4.0 | **GENERALIZING** |
| PPO_31b | 320,001,312 | 15 | 16.1 | 33.0 | 3.0 | **GENERALIZING** |
| PPO_31b | 330,001,312 | 15 | 19.4 | 53.0 | 7.0 | **GENERALIZING** |
| PPO_31b | 340,001,312 | 14 | 17.7 | 68.0 | 0.0 | **GENERALIZING** |
| PPO_31b | 350,001,312 | 15 | 21.9 | 44.0 | 1.0 | **GENERALIZING** |
| PPO_31b | 360,001,312 | 13 | 17.6 | 35.0 | 6.0 | **GENERALIZING** |
| PPO_31b | 370,001,312 | 16 | 19.9 | 59.0 | 0.0 | **GENERALIZING** |
| PPO_31b | 380,001,312 | 19 | 28.1 | 77.0 | 9.0 | **GENERALIZING** |
| PPO_31b | 390,001,312 | 16 | 25.2 | 62.0 | 4.0 | **GENERALIZING** |

**PPO_31b at 400M (100M sticky steps):** GENERALIZING on all 10 checks (10/10). Unique scores range 10-19. Average score rose from 5.5 (immediately post-transition) to 25-28 in the final checks, with best scores reaching 77. The model shows the same breakout-from-memorization pattern as PPO_30b — immediate jump to 10+ unique scores on the first check after stickiness is added, sustained across the full 100M sticky phase. However, PPO_31b's best scores are substantially lower than PPO_30b's (77 vs. 378), consistent with having only 100M sticky steps vs. PPO_30b's 300M.

**What to watch for in Phase 2:** when each run transitions to sticky actions (PPO_30b, PPO_31b), the key early warning sign from Experiment 2 was `ep_rew_mean` doubling or tripling within the first 10-30M steps while `explained_variance` → 1.0 and `value_loss` → ~0. The automated `MemorizationCheckCallback` is wired into both Phase 2 scripts with `sticky_actions=True`, so this should now surface automatically in the tracking CSVs rather than requiring a manual check.

**Predicted outcome table** (to be confirmed by data):

| Outcome | Interpretation |
|---------|----------------|
| PPO_30b works (low zero-score, varied funnel data), PPO_31b works | Basic competency is sufficient. 100M non-sticky steps is enough. The recipe is cheap. |
| PPO_30b fails (memorization), PPO_31b works | Depth matters. 400M is the right ballpark. PPO_26 needed ~1B because it started from PPO_24's 300M — not because 1B is inherently necessary. |
| Both fail | Something more fundamental is going on. The "non-sticky pretraining" hypothesis may be wrong, or the phase switch LR/config needs more work. |
| PPO_30b works, PPO_31b fails | Unexpected. Would suggest a Goldilocks effect — too much non-sticky pretraining creates habits that sticky-action training can't adapt. Worth investigating but low prior probability. |

**Interim finding (at 200M):** PPO_30b has achieved the "works" threshold for generalization — 14-19 unique scores sustained for 100M sticky steps with zero relapses. But score improvement hasn't followed yet (avg flat at ~28-33). Whether this counts as "works" in the final analysis depends on whether generalization converts into higher scores in the remaining 200M steps.

---

### Final Results — 10,000-Game Single-Env Evaluations (Sticky On)

> ⚠️ **PPO_31b evaluation is INCOMPLETE.** The funnel log (`recordings/PPO_31b_funnel_log.csv`) contains only 9,247 games (753 short of the 10,000 target). All PPO_31b statistics below were computed from this incomplete sample. While 9,247 games is a large sample and the statistics are unlikely to change dramatically, they should not be cited as final until the remaining 753 games are run. See FLAWS.md F-004.

Both models evaluated at 400M total steps using `final_model` with identical methodology to PPO_25/26/27 (persistent env, seed=None, sticky on).

#### Head-to-Head Baseline Comparison

| | PPO_26 | PPO_25 | PPO_27 | **PPO_30b** | **PPO_31b** |
|---|---|---|---|---|---|
| Pretrain→Sticky | 838M→1B | 1B→0 | 0→~880M | 100M→300M | 300M→100M |
| **Average** | 54.3 | 34.6 | 27.95 | **27.7** | **22.1** |
| **Median** | 46 | 30 | 23 | 21 | 20 |
| **Best** | 415 | 406 | 406 | 393 | 364 |
| **Zero-score** | 0.0% | 20.0% | 21.3% | **23.2%** | **2.3%** |
| **Funnel 400+** | 0.07% | 0.02% | 0.01% | 0.00% | 0.00% |

Neither model approaches PPO_26. The "100M beats 300M" claim from interim rollout data was wrong — there's a real trade-off.

#### Score Distribution Analysis

| Threshold | PPO_30b | PPO_31b |
|-----------|---------|---------|
| ≤0 | 23.2% | 2.3% |
| ≤5 | 29.0% | 11.3% |
| ≤10 | 35.3% | 23.8% |
| ≤20 | 49.4% | **51.9%** |
| ≤30 | 63.9% | 75.4% |
| ≤40 | 75.6% | 89.3% |
| ≤50 | 85.4% | 95.6% |
| ≤60 | 91.0% | 98.2% |
| ≤100 | 98.3% | 99.9% |

| Percentile | PPO_30b | PPO_31b |
|------------|---------|---------|
| P5 | 0 | 4 |
| P10 | 0 | 5 |
| P25 | 4 | 11 |
| P50 (median) | 21 | 20 |
| P75 | 40 | 30 |
| P90 | 58 | 41 |
| P95 | 72 | 49 |
| P99 | 231 | 67 |

**Conditional stats (non-zero games only):**

| | PPO_30b | PPO_31b |
|---|---|---|
| Non-zero count | 7,677 (76.8%) | 9,030 (97.7%) |
| Non-zero average | 36.1 | 22.7 |
| Non-zero median | 29 | 20 |

### Outcome — A Trade-Off, Not a Winner

The central finding of Experiment 3 is that sticky steps and non-sticky pretraining contribute to different things:

- **More sticky training (PPO_30b):** Produces the capability for high scores — P99 of 231, non-zero average of 36.1, individual games up to 393. But 23.2% of games score zero, worse than PPO_25's baseline. The model has genuine skill but a catastrophic failure mode.
- **More non-sticky pretraining (PPO_31b):** Suppresses catastrophic failure — only 2.3% zero-score, the best of any model except PPO_26. But scores are tightly capped: P99 of 67, no game above 364, average only 22.1. The model is consistent but never brilliant.

The medians are nearly identical (21 vs 20). The 5.6-point gap in averages comes entirely from the right tail — PPO_30b occasionally plays well, while PPO_31b almost never does.

**PPO_26 had both** (0% zero-score AND 54.3 average) because it had massive amounts of both ingredients (~838M non-sticky + ~1B sticky). At a 400M budget, you have to choose which to favor, and neither choice gets close to the full recipe.

**Interim signals were misleading.** The rollout mean gap (101 vs 43), the memo check GENERALIZING streak (33/33), and the massive outlier scores (368, 378) painted PPO_30b as dominant. The 10k-game data shows PPO_30b is just PPO_27 with a slightly fatter tail — same zero-score problem, same median. PPO_31b is actually the more interesting model: it achieves a 2.3% zero-score rate with only 400M total steps, while PPO_25 needed 1B non-sticky steps to reach 20%. The non-sticky pretraining is clearly building something real.

### Status

**EXPERIMENT 3 — COMPLETE.** Both models evaluated at 10k games. PPO_30a, PPO_31a, PPO_30b, PPO_31b all finished. Results documented above. See Experiment 4 for next directions.

---

### Post-Hoc Analysis: Memorization Confirmation (2026-07-14)

After Experiment 3 was declared complete, three additional analyses were run to verify whether the Phase 2 models genuinely generalized or were memorized policies masked by sticky-action noise. The results overturn the central "both models generalize" conclusion.

#### Memorization Check Calibration

A known-memorized model (PPO_30a/final_model, confirmed 2 unique scores in all non-sticky checks) was run through the MemorizationCheckCallback with `sticky_actions=True` to measure the noise baseline.

| Condition | Mean Unique | Range | P95 |
|-----------|------------|-------|-----|
| Non-sticky (p=0.0) | 2.0 | 2-2 | 2 |
| Sticky (p=0.25) | **11.3** | **8-14** | **14** |

A dead memorized policy + p=0.25 sticky noise produces 8-14 unique scores per 20-game batch, averaging 11.3. Both PPO_30b (10-19) and PPO_31b (10-19) fall partially within this noise baseline. **The GENERALIZING verdict is not reliable for sticky-action models.**

Full calibration data: `recordings/memorization_calibration.csv`

#### Sticky-Off Verification (Both Models)

| Model | Sticky Off Unique Scores | Verdict |
|-------|--------------------------|---------|
| PPO_30b | 2 (0, 69) | **MEMORIZED** — 99.8% zero-score |
| PPO_31b | 2 (29, 31) | **MEMORIZED** — all games 31.0 points, 178 frames |

**Both models collapse to fixed scripts without sticky actions.** PPO_31b's 31-point script is particularly revealing: its "low zero-score rate" (2.4% in the 10k eval) is not robustness — it's an artifact of a memorized script that happens to score 31 from every ALE state. PPO_30b's script zeros 99.8% of the time without sticky, explaining its 23.2% zero-score rate even with sticky (sticky noise rescues it from zeros in ~77% of games).

#### Sticky Probability Sweep

Both models evaluated at p ∈ {0.0, 0.05, 0.10, 0.15, 0.20, 0.25}, 500 games each.

| p | PPO_30b uniq | PPO_30b avg | PPO_30b zero% | PPO_31b uniq | PPO_31b avg | PPO_31b zero% |
|---|-------------|------------|--------------|-------------|------------|--------------|
| 0.00 | **2** | 0.1 | 99.8% | **2** | 31.0 | 0.0% |
| 0.05 | **55** | 17.2 | 76.8% | **63** | 27.1 | 0.4% |
| 0.10 | **90** | 24.6 | 58.6% | **63** | 25.7 | 0.8% |
| 0.15 | **92** | 28.5 | 40.6% | **65** | 24.9 | 0.6% |
| 0.20 | **88** | 27.5 | 33.2% | **60** | 23.3 | 1.4% |
| 0.25 | **88** | 28.5 | 20.0% | **64** | 23.5 | 1.4% |

**At just p=0.05 — 5% action-repeat probability — both models jump from 2 to 55-63 unique scores.** The "GENERALIZING" signal appears at the smallest possible noise level. Unique-score count in sticky environments is primarily a function of sticky probability, not policy reactivity.

PPO_30b's right-tail performance (best=392 at p=0.05) is real — sticky noise unlocks genuine scoring capability from its memorized scripts. PPO_31b's scores are tightly capped (best=78 at p=0.25, P99=67) — sticky noise produces small variations around its 31-point script but reveals very little genuine scoring capability.

#### Variance Decomposition (PPO_30b only)

| Condition | Unique | Avg | Median | Zero% | P99 |
|-----------|--------|-----|--------|-------|-----|
| det=True, sticky=0.25 | 93 | 30.1 | 23 | 18.8% | 278 |
| det=False, sticky=0.25 | 84 | 26.6 | 21 | 15.6% | 106 |
| det=True, sticky=0.0 | **2** | 0.1 | 0 | 99.8% | 0 |
| det=False, sticky=0.0 | **43** | 23.5 | 0 | 60.4% | 88 |

**Key finding: With stochastic sampling (det=False) but NO sticky actions, PPO_30b produces 43 unique scores and averages 23.5 — dramatically better than the deterministic collapse (2 unique, 0.1 avg).** The policy has non-zero action entropy — it has learned action preferences that stochastic sampling can exploit. The argmax decision rule (deterministic=True) produces the memorized-collapse behavior; the policy itself is not completely dead. Sticky noise and policy stochasticity are alternative ways to escape the memorized attractor, producing different score distributions.

#### Revised Conclusions for Experiment 3

1. **Both PPO_30b and PPO_31b are memorized policies.** The Phase 2 "GENERALIZING" verdicts were sticky-action noise, not genuine reactive behavior. This is confirmed by: (a) nosticky collapse to ≤2 unique scores, (b) calibration showing the noise baseline matches observed unique-score counts, and (c) the sticky sweep showing the "generalization" signal appears at p=0.05.

2. **The "trade-off" between pretraining duration and sticky steps is a trade-off between which memorized script each model learned.** PPO_30b's script produces zeros 99.8% of the time but contains latent high-score trajectories that sticky noise occasionally unlocks. PPO_31b's script produces 31 points reliably but has very limited upside. The trade-off is real but the mechanism is different from what was originally concluded.

3. **The policies are not completely dead.** Stochastic sampling (det=False) reveals the policies have learnable action preferences that the argmax decision rule suppresses. This is an important nuance: the collapse is in the argmax, not in the policy's learned distribution.

4. **The MemorizationCheckCallback threshold (≤2 unique scores) is valid for non-sticky environments but meaningless for sticky environments.** The calibrated threshold would need to be >14 unique scores for p=0.25, and different sticky probabilities require different thresholds.

5. **PPO_26 and PPO_27 are both CONFIRMED non-generalizing (2026-07-14).** Nosticky verification of all four sticky-trained models reveals two distinct failure modes: (a) **Script memorization** — non-sticky-pretrained models (PPO_26: 60 pts, PPO_31b: 31 pts, PPO_30b: 0 pts) play fixed deterministic scripts. Deeper pretraining produces higher-scoring scripts but never reactive policies. (b) **Noise coupling** — PPO_27 (p=0.25 from scratch) learned a policy dependent on sticky noise to function; without it, every game is 0 points in 19 frames. Neither training regime produces ball-tracking. **No model in this project has ever genuinely generalized.** This independently confirms Zhang et al. (2018), who found that *"stochasticity could neither prevent deep RL agents from serious overfitting nor detect overfitted agents effectively"* — ConvNets are naturally noise-robust and memorize through sticky perturbations.

---

### Reference

- Machado, M. C., Bellemare, M. G., Talvitie, E., Veness, J., Hausknecht, M., & Bowling, M. (2018). *Revisiting the Arcade Learning Environment: Evaluation Protocols and Open Problems for General Agents.* Journal of Artificial Intelligence Research, 61, 523-562.
- Zhang, C., Vinyals, O., Munos, R., & Bengio, S. (2018). *A Study on Overfitting in Deep Reinforcement Learning.* arXiv:1804.06893. — **Key finding:** sticky actions do not prevent memorization in deep ConvNet RL agents; CNNs are naturally noise-robust. This project independently confirmed this across 5 PPO models.

---

### Experiment 3.5: Continue PPO_30b/PPO_31b Past 400M (Considered and Rejected, 2026-07-14)

**Proposal:** Extend both PPO_30b and PPO_31b by 100-200M additional sticky steps (to 500-600M total) to test whether memorization can be escaped with enough sticky training.

**Reasons considered:**
- 400M was an arbitrary budget — PPO_26 succeeded at ~1.8B total. The memorization→generalization phase transition might require more steps.
- The sticky sweep showed PPO_30b has residual policy entropy (43 unique scores with det=False, sticky=off) — more training might strengthen alternative action paths.
- Near-zero marginal cost — scripts exist, checkpoints exist, just resume.
- Answers a permanent question: can sticky fine-tuning ever rescue a memorized foundation?

**Why rejected:**
1. **Evidence points to permanent memorization.** Experiment 2 showed sticky removal causes collapse within ~30M steps. Sticky actions prevent collapse but don't build generalization from a memorized base. There's no mechanistic reason to expect more steps changes this dynamic.
2. **The F-002 confound is permanent.** No amount of additional training can separate pretraining depth from sticky-step count in Experiment 3. Even if both models improve, the attribution remains underdetermined.
3. **The memorization track shows stability, not progress.** PPO_30b has been in the 10-19 unique score band for 33 consecutive checks spanning 290M sticky steps. No upward trend. The signal is sticky probability, not policy improvement.
4. **Calibration is damning.** Both models' 10-19 unique scores are indistinguishable from the dead-policy + noise baseline (8-14 unique, mean 11.3).
5. **PPO_26 nosticky verification is higher priority.** If PPO_26 also collapses without sticky, the entire "both ingredients" framework needs revision. That single data point changes direction more than another 200M steps on confirmed-memorized models.
6. **A clean experiment produces cleaner conclusions.** Low-sticky single-phase training (Experiment 4 Option A) tests one variable with no confounds. It's better science.

**Verdict:** REJECTED. Move to Experiment 4 with a clean design.

---

## Experiment 4: Sticky Probability — Single-Phase Training

### Context (Revised 2026-07-14)

The post-hoc analysis of Experiment 3 established three facts that reset the experimental direction:

1. **Non-sticky pretraining causes permanent memorization.** PPO_30a, PPO_31a, PPO_25 — every model trained without sticky actions collapsed to ≤2 unique scores. Phase 2 sticky fine-tuning masks this with noise but does not cure it.

2. **The sticky probability sweep revealed p=0.05 as transformative.** At inference time, just 5% action-repeat probability takes a memorized model from 2 unique scores to 55-63. The "generalization" signal is primarily a function of sticky probability, not policy reactivity.

3. **Full-sticky from scratch (p=0.25, PPO_27) prevents memorization but produces fragility.** PPO_27 had the worst single-env metrics (21.3% zero-score) despite holding the eval record (147.02). Too much stochasticity during training prevents the policy from building reliable skills.

These three facts point to the same unexplored dimension: **sticky probability itself.** Every experiment so far has used only p=0.0 or p=0.25. The intermediate range — where the sticky sweep showed the largest marginal effect — has never been tested during training.

The core question for Experiment 4: **Is there a sticky probability that prevents memorization (like p=0.25) while producing robust single-env policies (unlike p=0.25)?**

### Option A: Low-Sticky Single-Phase Training (RECOMMENDED)

**Design:** Train one model from scratch with `repeat_action_probability=0.05` (or 0.10), single phase, 400M total steps. No pretraining/fine-tuning split. Hyperparameters: n_envs=32, LR 2.5e-4→1e-5, clip 0.2→0.05, ent_coef=0.006.

**Hypothesis:** p=0.05 provides enough stochasticity to prevent deterministic script formation (unlike p=0.0) but not so much that the policy fails to build reliable reactive foundations (unlike p=0.25). This is the Goldilocks hypothesis — the stochasticity sweet spot where the policy *must* learn to react because memorized sequences are unreliably executed, but *can* learn to react because the environment is still mostly predictable.

**Why this is the right next experiment:**
- **Tests one variable.** Single phase, one sticky probability, no confounded phase transitions. The only thing being tested is "does low-sticky training produce better policies than either extreme?"
- **The sticky sweep provides a strong prior.** We already know p=0.05 produces behavioral diversity at inference time. The open question is whether training at p=0.05 builds a policy that generalizes under *deterministic* inference.
- **Simplifies the recipe.** If it works, the two-phase recipe (non-sticky pretrain → sticky fine-tune) is replaced with a single-phase run. That's a significant practical improvement.
- **PPO_26 nosticky verification can run in parallel on CPU.**

**Comparison groups (all at 400M total steps):**
| Model | Phase 1 | Phase 2 | Known Issue |
|-------|---------|---------|-------------|
| PPO_32 (new) | p=0.05 × 400M | — | Tested here |
| PPO_30b | p=0.0 × 100M | p=0.25 × 300M | Confirmed memorized + noise |
| PPO_31b | p=0.0 × 300M | p=0.25 × 100M | Confirmed memorized + noise |

PPO_27 (p=0.25 from scratch, ~1B steps) provides a loose upper bound on full-sticky performance but isn't a matched-total-steps comparison.

**Prediction table:**
| Outcome | Interpretation |
|---------|---------------|
| Nosticky unique scores ≥10 at 400M | Low-sticky training prevented memorization — recipe works |
| Nosticky unique scores 3-9 at 400M | Partial success — prevents total collapse but doesn't build full reactivity |
| Nosticky unique scores ≤2 at 400M | p=0.05 is still deterministic enough to allow memorization — need p≥0.10 |
| Sticky-on performance worse than PPO_30b | Low noise during training produces a weaker policy, even if less memorized |
| Sticky-on performance better than PPO_30b | Low-sticky is strictly better — prevents memorization AND builds stronger skills |

**Risk:** p=0.05 might still be low enough that the policy can memorize. The sticky sweep showed inference-time diversity at p=0.05, but that's testing a *trained* model through a noise lens — training dynamics at p=0.05 could still converge to deterministic scripts. If this happens, the next step is p=0.10.

**Estimated time:** ~2 weeks for 400M steps.

### Option B: Two-Model Sticky Probability Comparison

**Design:** Train two models from scratch: PPO_32 at p=0.05, PPO_33 at p=0.25. Both single-phase, both 400M total steps, identical hyperparameters. Direct head-to-head comparison of the only variable that matters: sticky probability.

**Why this is stronger than Option A:**
- Directly answers "does low-sticky beat full-sticky?" with matched total steps
- Controls for everything except sticky probability
- PPO_33 (p=0.25 from scratch at 400M) fills a missing data point — we have no full-sticky model at this budget

**Why this is weaker:**
- **2× the GPU time.** Two models × 400M = ~4 weeks sequential, or train PPO_33 on a second machine.
- PPO_27 (p=0.25 from scratch) already exists at higher step counts — we have a rough idea what full-sticky looks like.
- The most novel question is whether low-sticky works at all — Option A answers that first. If p=0.05 also memorizes, there's no point comparing it to p=0.25.

**Estimated time:** ~4 weeks sequential, ~2 weeks if both can run in parallel.

### Option C: Curriculum Sticky — Progressive Probability Increase

**Design:** Single model, single phase, but sticky probability increases with training progress:
- 0-100M steps: p=0.05
- 100M-200M steps: p=0.10
- 200M-400M steps: p=0.25

**Hypothesis:** The policy builds robust reactive foundations during the low-noise early phase (when it's learning basic paddle control and ball tracking), then adapts to increasing stochasticity as skills consolidate. This mirrors the non-sticky→sticky two-phase structure but replaces memorization-inducing p=0.0 with memorization-resistant p=0.05.

**Why this is interesting:**
- Addresses the concern that p=0.05 throughout might produce a policy that can't handle p=0.25 at inference time
- Progressive difficulty is a well-established curriculum learning principle
- The sticky sweep showed p=0.05 → p=0.25 is a smooth transition in behavioral diversity

**Why this is weaker than Option A:**
- **Confounds probability with training phase.** If it works, you don't know whether p=0.05 throughout would have worked just as well.
- More complex implementation (need to modify the env during training, or swap envs at phase boundaries).
- Tests two things at once: low-sticky early phase AND progressive probability increase.

**Estimated time:** ~2 weeks for 400M steps (same as Option A).

### Option D: PPO_26 Nosticky Verification First

**Design:** Before launching any new training, run `funnel_recorder_ppo_26_nosticky.py` — 500 games with sticky=off, persistent env, seed=None. Tests whether PPO_26 (avg 54.3, 0% zero-score) is genuinely generalizing or is also a memorized policy + sticky noise.

**Why this might change everything:**
- If PPO_26 IS memorized: the best model in the project is also noise-masked. The entire "both ingredients" framework collapses. The experimental direction pivots to "what actually prevents memorization?" rather than "how do we optimize the two-ingredient recipe?"
- If PPO_26 IS generalizing: we have a proven recipe (838M non-sticky + 1B sticky) that works. Experiment 4 should attempt to replicate it at smaller scale or with cleaner controls.

**This is not mutually exclusive with Options A-C.** It runs on CPU while the GPU trains. But the result is so consequential that it makes sense to get it before committing to a 2-week training run.

**Estimated time:** ~1-2 hours (CPU only, 500 games).

### Recommendation

**Do Option D + Option A in parallel.** PPO_26 nosticky verification runs on CPU (hours). PPO_32 (p=0.05 single-phase) trains on GPU (~2 weeks). Neither blocks the other.

If PPO_26 nosticky confirms generalization, Option A still answers the important question of whether a simpler single-phase recipe can work. If PPO_26 nosticky reveals memorization, Option A becomes even more important — it tests the leading candidate for a recipe that actually prevents memorization.

**Don't do Options B or C yet.** Option B (two-model comparison) is premature until we know whether low-sticky works at all. Option C (curriculum) confounds two variables and should only be attempted if Option A shows partial success (e.g., nosticky unique scores of 3-9 — not memorized but not fully generalizing).

### Measurement Protocol (for Experiment 4)

1. **MemorizationCheckCallback throughout training** — every 10M steps, 20 games, sticky OFF (so the verdict is actually meaningful). The calibration data now lets us interpret the results correctly.
2. **At 200M midpoint:** Run 500-game nosticky eval. If the model is already memorized (≤2 unique), we know early and can abort or adjust.
3. **At 400M completion:**
   - 10k-game single-env eval, sticky on (gold standard)
   - 10k-game single-env eval, sticky off (memorization verification)
   - 500-game eval at p=0.05, p=0.10, p=0.15, p=0.20 (sensitivity sweep)
   - Both deterministic and stochastic inference at each sticky level
4. **Compare against PPO_30b and PPO_31b** at matched 400M total steps
5. **Compute bootstrap CIs** on all mean/median/zero-score comparisons
6. **Update FLAWS.md** with any new issues discovered

---

## Experiment 5: Dynamics Randomization — Breaking Memorization Through Unpredictable Physics

**Status: DESIGNING (2026-07-14).** PPO_32 (Experiment 4) still training — results will determine whether Experiment 5 is needed and which direction it takes.

### Motivation

The central finding from Experiments 1-3: **no model in this project has ever genuinely generalized.** Every sticky-trained model tested without sticky actions collapsed to a deterministic script (PPO_26: 60 pts, PPO_31b: 31 pts, PPO_30b: 0 pts) or a noise-dependent degenerate policy (PPO_27: 100% zeros). Sticky actions do not prevent memorization — they mask it.

The root cause, per Zhang et al. (2018) and independently confirmed here: **CNNs are naturally robust to action-level perturbations.** Translation invariance is built into convolutional architectures. A 25% chance of action-repeat doesn't disrupt a memorized trajectory enough — the CNN's feature representations are smooth, and the noisy frame sequence maps to the same features as the clean one.

### The Design Insight (RL_REFERENCE.md Lesson #40)

Sticky actions perturb the **agent's output** (action choice). What matters for forcing reactivity is perturbing the **environment's dynamics** — the physics the agent must respond to.

- **Perceptual perturbations** (data augmentation, observation noise): CNNs are built to be robust to these. The pixel pattern changes, but the feature representation doesn't. The agent learns to see through the noise and arrives at the same memorizable features.
- **Dynamics perturbations** (variable ball speed, variable paddle width, variable frame skip): the ball is literally in a different place than a timed script expects. No amount of CNN invariance can compensate — the agent MUST observe and react.

This is domain randomization, the standard sim-to-real technique: randomize simulator parameters during training so the policy learns features that generalize across parameter ranges. Applied to Breakout: randomize the physics parameters a memorized script depends on, and the policy must develop ball-tracking because no fixed timed sequence works across all parameter values.

Later Breakout-style games (powerups, speed changes, challenge modes) implicitly use this principle — what was once a game-design feature is also a memorization countermeasure.

### Dependency Chain

```
PPO_32 finishes (p=0.05 from scratch)
├─ Generalizes? → DONE. Goldilocks hypothesis confirmed.
└─ Memorized? → Continue to Experiment 5

Experiment 5 Option A: Frame Skip Randomization (~1 week)
├─ Generalizes? → DONE. Cheap fix works.
└─ Memorized? → Continue to Option B

Experiment 5 Option B: RAM-Parameterized Physics (~2-3 weeks)
├─ Generalizes? → DONE. Domain randomization confirmed.
└─ Memorized? → Experiment 5 Option C: Custom ROM or PyGame clone
```

Each step is gated on the previous one's failure. No point building the complex solution if the simple one works.

### Technical Foundation: What CAN Be Randomized

#### Available Right Now (no ROM modification)

**Frame skip randomization** — apply the agent's action for a random number of ALE frames (e.g., 2-8 frames) rather than a fixed 4. This varies:
- **Effective ball speed**: 2-frame skip = ball moves 2 pixels per decision; 8-frame skip = ball moves 8 pixels — effectively 4× faster
- **Effective paddle responsiveness**: 2 frames of LEFT = paddle moves 2 pixels; 8 frames = 8 pixels — paddle moves further per decision

A wrapper implementation is ~30 lines:

```python
class RandomFrameSkip(gym.Wrapper):
    def __init__(self, env, min_frames=2, max_frames=8):
        super().__init__(env)
        self.min = min_frames
        self.max = max_frames
    
    def step(self, action):
        k = np.random.randint(self.min, self.max + 1)
        total_reward = 0
        for _ in range(k):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info
```

Cumulative position uncertainty: after 100 agent decisions at frameskip 2-8, ball position uncertainty is ~±300 pixels — far beyond what any timed script can accommodate.

**Limitation:** Frame skip randomization changes the *pace* of the game, not its fundamental invariants. Reflection angles, brick layout, and paddle-to-ball speed ratio remain constant. The agent might learn a speed-conditioned script (detect speed from first few frames, select from a small lookup table of speed-appropriate timed sequences) rather than genuine ball-tracking.

#### Requires Investigation (ALE RAM manipulation)

The project has confirmed these RAM addresses (RL_REFERENCE.md Part 2):

| Address | Value | Range |
|---------|-------|-------|
| 70 | Paddle x position | 0-191 |
| 72 | Ball x position | 0-191 |
| 90 | Ball y position | increments each step |

`env.unwrapped.ale.setRAM(address, value)` can write to any of the 128 bytes of Atari 2600 RAM. But many game parameters are not stored in RAM — they're in the ROM code and TIA hardware registers.

**Paddle width:** Controlled by the TIA NUSIZx register (Number-Size). Standard Breakout uses an 8-pixel-wide player sprite. NUSIZ can double (16px) or quadruple (32px) it. The ROM *may* cache this value in a RAM variable before writing it to the TIA each frame — if so, `setRAM()` works. If the ROM loads it as an immediate constant (`LDA #$05; STA NUSIZ0`), it's not writable without ROM modification. Unknown without disassembly inspection.

**Ball speed:** Determined by ROM code that moves the ball object every N frames. If the speed divider is stored in RAM (some games do this for progressive difficulty), it's modifiable. If hardcoded (likely for Breakout's simple design), it's not. Unknown.

**Ball size:** The ball is typically a missile sprite (1-8 pixels wide). Its size, like the paddle, is controlled by TIA registers. Same uncertainty as paddle width.

**Paddle speed:** The paddle moves based on joystick input polling each frame. Speed is effectively 1 pixel per frame. To vary it, we'd need to vary the frameskip — which is what Option A does.

**Proposed investigation:** A probe script that dumps all 128 RAM bytes every frame during gameplay and identifies which bytes correlate with which game parameters. This has been partially done for paddle_x/ball_x/ball_y (addresses 70/72/90), but we should systematically identify the remaining addresses. This is a few hours of work that pays off by telling us what's actually possible.

#### Fallbacks If RAM Manipulation is Insufficient

1. **Custom ROM:** The Atari homebrew community has tools for modifying 2600 ROMs. A "Breakout with variable paddle width" ROM could be created by patching the NUSIZ constant in the original ROM. This is more work but gives full control.
2. **PyGame Breakout clone:** Full control over every physics parameter. Zero literature comparability but maximum experimental flexibility. Best as a separate project if ALE-level approaches are exhausted.

### Measurement Protocol

Every Experiment 5 variant uses the same verification:

1. **Pre-training probe:** If pursuing RAM manipulation, run a probe script to catalog all accessible game parameters
2. **During training:** MemorizationCheckCallback every 10M steps, sticky OFF (verdict now validated for non-sticky environments)
3. **At completion:**
   - 10k-game eval with sticky OFF (nosticky verification — the only reliable behavioral test)
   - 10k-game eval with sticky ON at p=0.25 (for literature comparability)
   - Both deterministic and stochastic inference
4. **Control:** Compare against PPO_26 (best memorized script, 60 pts nosticky) — even a reactive policy scoring 40 pts genuinely beats a script scoring 60
5. **Nosticky ≤2 unique scores = MEMORIZED** regardless of sticky-on performance

### Prediction Table

| Training | Nosticky result | Interpretation |
|----------|----------------|----------------|
| Frame skip 2-8 | ≥10 unique, varied scores | Frame skip randomization forces reactivity. Cheap fix confirmed. |
| Frame skip 2-8 | ≤2 unique, one script | Speed-conditioned scripts beat frame skip. Move to Option B. |
| RAM-parameter paddle width | ≥10 unique, varied scores | Domain randomization confirmed. Parameter sweep next. |
| RAM-parameter paddle width | ≤2 unique, one script | Even physics randomization > perceptual randomization alone, but the CNN finds a way. Consider multi-parameter randomization or custom ROM. |
| Multi-parameter (width + speed) | ≤2 unique | The CNN's memorization capacity exceeds expectations at every turn. Investigate network architecture changes (dropout, smaller CNN) combined with dynamics randomization. |

### Interim Observation (2026-07-14, ~15M steps)

PPO_33 showed a phase-transition breakthrough between 12.3M and 13.1M steps: `ep_rew_mean` jumped from ~9 to ~90, eval score from 16 to 98, both in under 1M steps. At comparable step counts (~12-15M), PPO_33's eval score (98) nearly doubles PPO_32's (53).

Critically, PPO_33 achieved this WITHOUT the early markers of memorization:
- `explained_variance` = 0.50 (PPO_32: 0.96 — approaching the 1.0 danger threshold per Lesson #30)
- `entropy_loss` = -0.79 (PPO_32: -0.29 — much closer to collapse at 0)
- The value function is still learning and the policy retains exploration entropy

PPO_32 at 12M is already showing the early signatures that preceded memorization in Experiments 1-3: near-maximum EV and collapsing entropy. PPO_33 at 15M looks healthier on both metrics while also scoring higher.

Caveats: the memorization check at 12.8M still returned MEMORIZED (2 unique scores, avg 3.8), but this was pre-breakthrough random play. The 20M check will be the first post-breakthrough behavioral test. And we've been burned by early metrics before (PPO_26, PPO_27). The real question is whether PPO_33 sustains healthy entropy and EV through 400M steps without collapsing.

### Open Questions

1. **Does the agent detect frame skip from a single observation?** If the ball hasn't moved yet in the first frame after reset, can the agent infer "this is a slow-speed episode" from visual cues alone? If so, it can select a speed-matched script rather than tracking.
2. **Is dynamics randomization enough, or do we need perceptual randomization too?** The ideal solution likely combines both: the agent can't memorize pixels AND can't memorize timed sequences. But one variable at a time — test dynamics first.
3. **What RAM addresses actually exist for Breakout physics parameters?** The probe script needs to happen before committing to Option B. Don't design around capabilities we haven't confirmed.
4. **Is there already a "variable Breakout" ROM?** The homebrew community has been hacking Atari games for 40+ years. Worth searching before building our own.
