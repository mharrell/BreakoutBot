# Experiments

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
| PPO_31a | 400M steps, no sticky, fresh agent | — | Non-sticky baseline at 400M |
| PPO_31b | 300M steps, sticky on, loaded from PPO_31a | PPO_31a → PPO_31b | Hypothesis B — does 400M work? |

PPO_30a and PPO_31a run simultaneously (Phase 1 in parallel). PPO_30b starts as soon as PPO_30a completes (~4-8 hrs), without waiting for PPO_31a (~16-32 hrs).

**Key design difference from PPO_28/29:** the LR schedule at the Phase 1→2 switch restarts at 1e-4 (not 2.5e-4). This is deliberately conservative — high LR at phase transition was identified as a likely contributor to PPO_28/29's fast collapse.

**Preserved checkpoints:** `PPO_30a/final_model` and `PPO_31a/final_model` are saved at the phase transition and can be independently evaluated via funnel recorder to answer a bonus question: what does non-sticky performance look like at 100M vs 400M steps?

---

### Results

*Pending. PPO_30a and PPO_31a currently in Phase 1 training.*

**What to watch for:**

When Phase 2 (sticky) begins for each run, the key early warning sign from Experiment 2 was `ep_rew_mean` doubling or tripling within the first 10-30M steps while `explained_variance` → 1.0 and `value_loss` → ~0. If either PPO_30b or PPO_31b shows this pattern, it should be verified behaviorally immediately (run `diagnostic_ppo28.py` pattern against the checkpoint) rather than waiting for the full run to complete.

**Predicted outcome table** (to be confirmed by data):

| Outcome | Interpretation |
|---------|----------------|
| PPO_30b works (low zero-score, varied funnel data), PPO_31b works | Basic competency is sufficient. 100M non-sticky steps is enough. The recipe is cheap. |
| PPO_30b fails (memorization), PPO_31b works | Depth matters. 400M is the right ballpark. PPO_26 needed ~1B because it started from PPO_24's 300M — not because 1B is inherently necessary. |
| Both fail | Something more fundamental is going on. The "non-sticky pretraining" hypothesis may be wrong, or the phase switch LR/config needs more work. |
| PPO_30b works, PPO_31b fails | Unexpected. Would suggest a Goldilocks effect — too much non-sticky pretraining creates habits that sticky-action training can't adapt. Worth investigating but low prior probability. |

---

### Status

**IN PROGRESS.** PPO_30a (100M non-sticky) and PPO_31a (400M non-sticky) running simultaneously. PPO_30b and PPO_31b will start once their respective Phase 1 runs complete.

---

### Reference

- Machado, M. C., Bellemare, M. G., Talvitie, E., Veness, J., Hausknecht, M., & Bowling, M. (2018). *Revisiting the Arcade Learning Environment: Evaluation Protocols and Open Problems for General Agents.* Journal of Artificial Intelligence Research, 61, 523-562.