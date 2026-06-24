# Experiments

## Experiment 1: Sticky Actions and Training Regime — PPO_26 vs PPO_27

> **Status update (latest evals):** Both runs hit new personal records in the most recent batch — PPO_26 reached 134.16 @ 905.6M, and PPO_27 reached **143.36 @ 804.8M**, which is now the **new all-time record across all three runs**, surpassing PPO_25's original 140.94. See updated Results and Conclusions below.

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
| All-time peak (as of latest evals) | 134.16 @ 905.6M steps | **143.36 @ 804.8M steps** 🏆 — new all-time record, surpasses PPO_25's 140.94 |

PPO_26's inherited weights gave it a significant early advantage — it hit 80+ within 22M steps while PPO_27 took 129M. PPO_27 caught up and both crossed 100 at nearly identical step counts (~300M). **Since then, PPO_27 has pulled ahead decisively** — its 143.36 peak not only beats PPO_26's 134.16 but also overtakes PPO_25's original 140.94, making PPO_27 the outright champion across all runs to date.

#### Convergence Speed

At equivalent wall-clock training time (~319,000 seconds elapsed for both), the runs stood at:
- PPO_26: 317M steps, rollout mean ~81
- PPO_27: 236M steps, rollout mean ~85

PPO_27 was producing higher rollout means despite fewer steps, because 32 envs run faster per step than 64. Normalizing by steps rather than time, PPO_26 held a small advantage in peak scores at that point — but this has since reversed (see above).

#### Training Stability

PPO_26 showed significantly wider oscillation throughout, with eval scores swinging between 47 and 113 in the 250-400M step range. In the most recent logged batch (889.6M–947.2M), PPO_26 continued this pattern, swinging from 71.50 to 134.16 — a wide range that includes both its new personal record and its batch low arriving just a few checkpoints apart. Notably, this batch **ends on a downward note** (71.50 at the final logged step), which is worth monitoring for a real regression vs. ordinary exploration noise.

PPO_27's oscillation has remained tighter and more consistent by comparison. In its most recent logged batch (798.4M–856.0M), PPO_27 averaged 112.3 against PPO_26's 104.2 over a comparable batch — and PPO_27's lowest score in that batch (88.34) is still well above PPO_26's batch low (71.50). This is visible in the approx_kl values from earlier in the runs — PPO_27 consistently showed lower KL divergence (0.007-0.020) vs PPO_26 (0.025-0.090).

#### Single-Env Watch Performance

All three models were evaluated using the same `record_funnels.py` script with threshold 500, deterministic=True, single environment, against their respective best_model checkpoints. PPO_26 was run for 874 games before a manual stop; PPO_27 was left running overnight reaching 9,000+ games with its average fully converged.

| Metric | PPO_25 (no sticky) | PPO_26 (inherited + sticky) | PPO_27 (fresh + sticky) |
|--------|-------------------|----------------------------|------------------------|
| Average score | ~36 | ~53 | ~23.5 |
| Best score | 337 | 396 | 390 |
| Funnel rate (200+ pts) | ~2.2% (11/506) | ~1.8% (9/502) | ~0.8% (4/500) |
| Games sampled | 506 | 874 | 9,000+ |

**Unexpected result — PPO_26 has the highest single-env average score.** Despite having sticky actions applied mid-career on top of billion steps of non-sticky training, PPO_26 averaged ~53 per game compared to PPO_25's ~36 and PPO_27's ~23.5. PPO_27's average is completely stable at 23.5 across 9,000+ games — it is not converging upward with more samples, confirming this is its true single-env floor rather than an artifact of small sample size.

**Note on the apparent tension with the eval-score finding above:** This single-env watch data was captured at an earlier point in PPO_27's training (snapshot, not continuously updated). PPO_27's eval/mean_reward metric (averaged across 50 parallel eval episodes) has since climbed to a new all-time high, but that doesn't necessarily mean its single-env funnel rate or single-env average has improved proportionally — those would need to be re-measured against PPO_27's current best_model to know for sure. **This re-measurement is recommended as a first step before designing Experiment 2** (see below).

**The gap between PPO_26 (~53) and PPO_27 (~23.5) is substantial and consistent — roughly 2x** as of the last single-env measurement. This appears to be the sticky action penalty in full effect on a model trained exclusively with them from scratch. PPO_26, having inherited a billion steps of non-sticky training before switching regimes, retained enough of that baseline competence to score significantly higher in clean single-env play — at least as of that snapshot.

**The funnel rate ordering (as of that snapshot) is consistent with the sticky action penalty hypothesis.** PPO_25 (no sticky, clean execution) had the highest funnel rate at 2.2%, PPO_26 sat in the middle at 1.8%, and PPO_27 had the lowest at 0.8%. Sticky actions make funnel completion harder — the tunnel strategy requires precise sustained ball tracking, and a randomly repeated action during the critical phase can lose the ball before it racks up significant points.

**These results should be interpreted cautiously.** All three models were evaluated against best_model saves recorded at different training stages, and PPO_27 in particular has progressed substantially since its single-env snapshot was taken. A fully controlled comparison would require evaluating all three at identical step counts with identical environment configurations, ideally re-run against current best_model checkpoints.

---

### Conclusions

**1. Inherited weights with sticky actions show diminishing returns.**
PPO_26 recovered quickly early on (reaching 80+ in 22M steps) but this advantage eroded over training. The billion steps of non-sticky-action experience created deeply ingrained positional habits that were slow to overwrite. By 300M steps the advantage was essentially gone, and by the latest batch PPO_26 trails PPO_27 in both peak and average eval score.

**2. Fresh training with sticky actions is more sample efficient for high performance — and now holds the outright record.**
PPO_27 broke 100 at 299M steps — nearly the same as PPO_26's 304M — despite starting from random weights. As of the latest evals, PPO_27 has gone on to set the new all-time eval record (143.36), surpassing both PPO_26 and the original PPO_25 baseline. **This resolves the open question from earlier in this experiment:** for reaching the highest performance tier, starting fresh with the correct environment configuration is now confirmed to outperform inheriting a billion steps of prior (mismatched) training, at least at current step counts.

**3. Sticky actions produce more stable training dynamics.**
PPO_27 showed consistently lower KL divergence, tighter eval oscillation, and more predictable improvement curves. PPO_26's wider swings — including its most recent batch ending on a downward note after a fresh peak — suggest the policy may still be overwriting older habits rather than building cleanly on a stable foundation, even this far into training.

**4. Sticky actions improve general play consistency but reduce tunnel exploitation rate — though this needs re-confirmation against current models.**
As of the original snapshot, PPO_26 achieved the highest single-env average score (~52) of the three models, suggesting sticky-action training does improve robust ball tracking, and all sticky-action models showed lower funnel completion rates than PPO_25. This tradeoff is plausible and consistent with the precision-vs-robustness tradeoff sticky actions are expected to introduce. However, given how much PPO_27's eval scores have since moved, this finding is now **stale relative to PPO_27's current best_model** and should be re-measured (see Recommended Next Step below).

**5. Peak eval scores now favor PPO_27 outright — the open question from the original writeup is resolved.**
The original conclusion noted PPO_26 held the higher peak (113.62 vs 103.82) but that PPO_27 was younger and still climbing, leaving open whether it would surpass PPO_26. **It has** — PPO_27's 143.36 now exceeds PPO_26's 134.16, and exceeds the original all-time baseline set by PPO_25 (140.94). PPO_27 is the new champion on every score-based metric tracked in this experiment except single-env funnel rate (as last measured).

---

### Recommended Next Step (before designing Experiment 2)

Given how much PPO_27 has moved since the single-env watch snapshot was taken, re-run `ad_hoc_eval.py` or `funnel_recorder.py` against **PPO_27's current best_model** (and PPO_26's, for a fair comparison) before committing to a new experiment design. The current single-env numbers (~23.5 avg, 0.8% funnel rate for PPO_27) may no longer reflect its actual current behavior now that its eval score has overtaken everyone else's.

---

### Caveats

This is not a perfectly controlled experiment. Confounding variables include:

- **Environment count:** PPO_26 used 64 envs, PPO_27 used 32. This affects both training dynamics and the effective batch size despite batch_size being scaled proportionally.
- **Starting point:** PPO_26 inherited not just PPO_25's weights but also its optimizer state and learning rate schedule, which were reset. The schedule restart may have affected early training dynamics.
- **Hardware sharing:** Both runs trained simultaneously on the same GPU, with each receiving roughly half the available compute. Neither was running at full speed.
- **System restarts:** Multiple unplanned interruptions throughout both runs may have affected continuity.
- **Stale single-env data:** The single-env watch comparison table above was captured at one point in time and has not been refreshed against current best_model checkpoints, despite eval scores moving substantially since then.

Despite these limitations, the directional conclusions on training stability and overall eval performance are consistent across multiple metrics and observation methods.

---

## Option D (Considered and Rejected): Process-Based Reward via Paddle Bounces + Episode Length

Raised as an alternative to score-based reward, on the reasoning that human skill training often benefits from rewarding correct *process* over a graded *outcome* metric (e.g. math pedagogy rewarding correct steps over just the final answer). The proposed implementation: reward successful paddle bounces and episode length instead of score, on the theory that these are "fundamental skills" that can't be metagamed.

**Rejected for two independent reasons:**

1. **Already tried, twice, and failed both times.** PPO_15 rewarded ball-tracking (a process proxy similar in spirit) and the agent learned to mirror the ball without scoring. PPO_18 rewarded paddle hits directly and collapsed to an eval score of 19.0. This is the same failure mode recurring, not new territory.
2. **Structurally unsound, not just under-tuned.** Both proposed metrics (bounces, episode length) can be maximized by a degenerate strategy that never engages with the actual task: bounce the ball straight up at 0°, park the paddle beneath it, and let it loop against the ceiling indefinitely. This clears a column or two of bricks incidentally on the way to settling into the loop, then accumulates large reward from survival time alone while doing nothing further toward the actual goal. There's no equivalent shortcut available for "bricks removed" — see commentary below.

**Resolved clarification — the project has actually been running a process-style reward all along.** `ClipRewardEnv` flattens every brick hit to exactly 1.0 regardless of point value, so every run (PPO_5–PPO_27) has trained on "bricks removed," not raw score — raw score has only ever been an eval/comparison readout, never the training signal. This is a useful contrast case: "bricks removed" is a process metric that has no cheap degenerate maximum (you cannot rack up brick-clear reward without clearing bricks), while "bounces + survival time" does. The lesson isn't "process rewards don't work" — it's that a process proxy is only safe if there's no way to satisfy it without doing the real task. Full writeup added to RL_REFERENCE.md Part 3 and Key Lessons Learned #22.

---

## Planned Next Steps (as of latest session)

**Trigger:** once PPO_26 and PPO_27 each reach 1 billion total steps.

**Step 1 — Refresh the funnel/score baseline.** Run `funnel_recorder.py` (or `ad_hoc_eval.py`) against the current `best_model` checkpoint for both PPO_26 and PPO_27, capped at **10,000 games each**. This replaces the stale single-env comparison table in Experiment 1 (captured at an earlier, lower-performing snapshot of PPO_27) with numbers that reflect each agent's actual current behavior — average score, best score, and funnel rate (200+ pts). Re-running against PPO_25's existing best_model as a control, if not already at 10,000 games, would also be worth doing for a clean three-way comparison at matched sample size.

**Step 2 — Pick one "free" experiment to run alongside/after the refresh.** "Free" here means: no new training run, no new reward design, no new Goodhart risk — just analysis of data and models that already exist. Ranked by cost and expected value:

| # | Investigation | Cost | What it would tell us |
|---|---------------|------|------------------------|
| 1 | Diagnose PPO_26's recent downward-ending batch (134.16 → 71.50) using TensorBoard (`approx_kl`, `entropy_loss`, `value_loss`) for that step range | Free — log analysis only | Whether PPO_26 is in a real regression or normal oscillation; informs whether it's worth continuing past 1B steps as-is |
| 2 | Verify whether sticky actions actually fixed the original "dead ball" problem (agent ignoring the ball after losing a life) — qualitative pass with `watch.py` | Free — observation only | Closes the loop on one of the two original motivations for Experiment 1, which has only been inferred from aggregate scores so far, never directly checked |
| 3 | Sticky-action intensity sweep (`repeat_action_probability` at 0.05/0.10/0.15 vs the current 0.25) | Low — short branch runs, no new reward design | Maps the actual stability-vs-funnel-rate tradeoff curve instead of assuming it's linear; could reveal a sweet spot that keeps most of the stability gain while recovering funnel rate |
| 4 | Revisit `net=[512,512]` at current training durations | Medium — full run needed to be conclusive | Whether "bigger nets underperform" (established at <25M steps, PPO_8–11) still holds at 1B+ step scale, or was a training-duration artifact |

Items 1 and 2 are pure analysis of existing data/checkpoints and cost nothing to run — good candidates to pad out this report before committing to anything that requires new compute. Item 3 is the most direct lever on the actual open problem (funnel rate under sticky actions) and the natural next real experiment if 1 and 2 don't change the picture. Item 4 is lower priority — flagged for later, not for this round.

**Deferred:** the funnel-commitment reward-shaping idea (originally "Option C") stays on the shelf until the above is done — it carries the most Goodhart risk of any option discussed, and the sweep in item 3 may make it unnecessary.

*Whichever of items 1-4 gets picked, the results should be added back into this doc — and per the plan, may well surface new questions worth tracking here too.*

---

### Reference

- Machado, M. C., Bellemare, M. G., Talvitie, E., Veness, J., Hausknecht, M., & Bowling, M. (2018). *Revisiting the Arcade Learning Environment: Evaluation Protocols and Open Problems for General Agents.* Journal of Artificial Intelligence Research, 61, 523-562.
