# Experiments

## Experiment 1: Sticky Actions and Training Regime — PPO_26 vs PPO_27

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
| All-time peak (so far) | 113.62 @ 412M steps | 103.82 @ 299M steps |

PPO_26's inherited weights gave it a significant early advantage — it hit 80+ within 22M steps while PPO_27 took 129M. However, PPO_27 caught up rapidly and both crossed 100 at nearly identical step counts (~300M). From that point forward their trajectories are comparable, with PPO_26 holding a slight peak advantage.

#### Convergence Speed

At equivalent wall-clock training time (~319,000 seconds elapsed for both), the runs stood at:
- PPO_26: 317M steps, rollout mean ~81
- PPO_27: 236M steps, rollout mean ~85

PPO_27 was producing higher rollout means despite fewer steps, because 32 envs run faster per step than 64. Normalizing by steps rather than time, PPO_26 held a small advantage in peak scores but PPO_27 showed more stable, less oscillatory behavior.

#### Training Stability

PPO_26 showed significantly wider oscillation throughout, with eval scores swinging between 47 and 113 in the 250-400M step range. PPO_27's oscillation was tighter and more consistent, rarely dropping below 70 after the first 150M steps. This is visible in the approx_kl values — PPO_27 consistently showed lower KL divergence (0.007-0.020) vs PPO_26 (0.025-0.090).

#### Single-Env Watch Performance

All three models were evaluated using the same `record_funnels.py` script with threshold 500, deterministic=True, single environment, against their respective best_model checkpoints. PPO_26 was run for 874 games before a manual stop; PPO_27 was left running overnight reaching 9,000+ games with its average fully converged.

| Metric | PPO_25 (no sticky) | PPO_26 (inherited + sticky) | PPO_27 (fresh + sticky) |
|--------|-------------------|----------------------------|------------------------|
| Average score | ~36 | ~53 | ~23.5 |
| Best score | 337 | 396 | 390 |
| Funnel rate (200+ pts) | ~2.2% (11/506) | ~1.8% (9/502) | ~0.8% (4/500) |
| Games sampled | 506 | 874 | 9,000+ |

**Unexpected result — PPO_26 has the highest average score.** Despite having sticky actions applied mid-career on top of billion steps of non-sticky training, PPO_26 averaged ~53 per game compared to PPO_25's ~36 and PPO_27's ~23.5. PPO_27's average is completely stable at 23.5 across 9,000+ games — it is not converging upward with more samples, confirming this is its true single-env floor rather than an artifact of small sample size.

**The gap between PPO_26 (~53) and PPO_27 (~23.5) is substantial and consistent — roughly 2x.** This appears to be the sticky action penalty in full effect on a model trained exclusively with them from scratch. PPO_26, having inherited a billion steps of non-sticky training before switching regimes, retained enough of that baseline competence to score significantly higher in clean single-env play.

**The funnel rate ordering is consistent with the sticky action penalty hypothesis.** PPO_25 (no sticky, clean execution) has the highest funnel rate at 2.2%, PPO_26 sits in the middle at 1.8%, and PPO_27 has the lowest at 0.8%. Sticky actions make funnel completion harder — the tunnel strategy requires precise sustained ball tracking, and a randomly repeated action during the critical phase can lose the ball before it racks up significant points.

**The average score and funnel rate tell different stories.** PPO_26's higher average despite lower funnel rate than PPO_25 suggests it scores more consistently on non-funnel games — better general play — while PPO_25 may be more bimodal. This is consistent with the theoretical benefit of sticky actions producing more robust reactive policies, even if the tunnel exploitation specifically suffers.

**These results should be interpreted cautiously.** All three models were evaluated against best_model saves recorded at different training stages. A fully controlled comparison would require evaluating all three at identical step counts with identical environment configurations.

---

### Conclusions

**1. Inherited weights with sticky actions show diminishing returns.**
PPO_26 recovered quickly early on (reaching 80+ in 22M steps) but this advantage eroded over training. The billion steps of non-sticky-action experience created deeply ingrained positional habits that were slow to overwrite. By 300M steps the advantage was essentially gone.

**2. Fresh training with sticky actions is more sample efficient for high performance.**
PPO_27 broke 100 at 299M steps — nearly the same as PPO_26's 304M — despite starting from random weights. For reaching the 100+ performance tier, starting fresh with the correct environment is at least as good as inheriting a billion steps of prior training.

**3. Sticky actions produce more stable training dynamics.**
PPO_27 showed consistently lower KL divergence, tighter eval oscillation, and more predictable improvement curves. PPO_26's wider swings suggest the policy was still overwriting older habits rather than building cleanly on a stable foundation.

**4. Sticky actions improve general play consistency but reduce tunnel exploitation rate.**
PPO_26 achieved the highest single-env average score (~52) of the three models, suggesting sticky action training does improve robust ball tracking. However all sticky action models showed lower funnel completion rates than PPO_25, confirming that clean action execution is advantageous for the precise tunnel strategy specifically. This is an unresolved tension — sticky actions appear to trade tunnel exploitation frequency for more consistent general play.

**5. Peak eval scores still favor more total training time.**
PPO_26 holds the higher peak (113.62 vs 103.82) but PPO_27 is younger and still climbing. Whether PPO_27 ultimately surpasses PPO_26's peak is an open question as both runs are ongoing.

---

### Caveats

This is not a perfectly controlled experiment. Confounding variables include:

- **Environment count:** PPO_26 used 64 envs, PPO_27 used 32. This affects both training dynamics and the effective batch size despite batch_size being scaled proportionally.
- **Starting point:** PPO_26 inherited not just PPO_25's weights but also its optimizer state and learning rate schedule, which were reset. The schedule restart may have affected early training dynamics.
- **Hardware sharing:** Both runs trained simultaneously on the same GPU, with each receiving roughly half the available compute. Neither was running at full speed.
- **System restarts:** Multiple unplanned interruptions throughout both runs may have affected continuity.

Despite these limitations, the directional conclusions are consistent across multiple metrics and observation methods.

---

### Reference

- Machado, M. C., Bellemare, M. G., Talvitie, E., Veness, J., Hausknecht, M., & Bowling, M. (2018). *Revisiting the Arcade Learning Environment: Evaluation Protocols and Open Problems for General Agents.* Journal of Artificial Intelligence Research, 61, 523-562.