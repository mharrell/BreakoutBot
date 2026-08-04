# Three Lines of Code Fixed 123 Failed PPO Experiments on Atari Breakout

## After two years of trying to force PPO to generalize — sticky actions, cursor wrappers, adversarial bumpers, dynamics randomization — every single model memorized a script. The fix wasn't more environment engineering. It was directly rewarding the behavior we actually wanted.

---

I started this project with a simple question: can PPO learn to play Atari Breakout reactively — tracking the ball and positioning the paddle accordingly — or does it always converge to a memorized action sequence?

After 124 controlled experiments, I have an answer. The first 123 failed. Every model, across every approach I could think of, learned a script. The 124th worked. And the fix was three lines of code.

```python
distance = abs(paddle_x - ball_x)
bonus = 0.05 * max(0.0, 1.0 - distance / 80.0)
reward += bonus
```

That's it. A tiny dense reward for keeping the paddle horizontally close to the ball during descent. 0.05 per frame — about 50 bonus points over a full game, equivalent to roughly 7 yellow bricks. Applied during training only. The behavior transferred to clean eval without the bonus.

---

## The 123 Failures

I tried everything. Literally.

| Approach | Hypothesis | Why It Failed |
|---|---|---|
| **Sticky actions** (p=0.25) | Random noise prevents overfitting | Breakout is forgiving — scripts survive noise. Dead policy + noise = 8–14 "unique" scores |
| **Cursor wrapper** | Penalize paddle-ball distance | PPO hedged: reactive *distribution*, memorized *argmax* |
| **Entropy bonus** (up to 0.10) | Reward diverse actions | Entropy can come from anywhere — doesn't require ball-tracking |
| **Frame skip** | Unpredictable timing | CNN conditions on skip pattern; PPO finds skip-conditioned script |
| **Randomized bricks** | Different layout each episode | CNN conditions on first-frame pixels; PPO finds layout-conditioned script |
| **Dynamics randomization** | Varying physics per episode | Same — conditioned on first frames, not reactive |
| **Adversarial bumpers** | Moving threats force adaptation | Timing variance masked memorization |
| **Random ball bounce** | Non-conditionable perturbation | Breakout is still forgiving enough for scripts |
| **Brick pre-clearing** (1-life) | Remove scoring, force survival | Scripts still viable in 1-life mode |

Every approach shared the same assumption: **make scripts non-viable and PPO will be forced to generalize.** PPO consistently proved us wrong. It found scripts that survived sticky noise, adapted to different layouts, tolerated timing offsets, and worked around adversarial entities. The optimum was always a script — only the shape changed.

---

## The Diagnostic Blind Spot

For most of this project, I was measuring the wrong thing.

Every diagnostic I built — intervention probes, gradient tests, SCAD, memorization checks — measured the **policy distribution**: the probability PPO assigns to each action. But evaluation uses the **argmax**: the single action with the highest probability. A model can have a rich, entropy-laden distribution that shifts in response to the ball's position, while its argmax sits in the same place every frame.

I discovered this blind spot on July 30, 2026, with a test I call the **split-watcher**. Run the same model on two different brick layouts side-by-side. Give it independent predictions per side — no frame-stacking cross-contamination. Different bricks → different ball bounces → a reactive policy MUST produce different paddle positions. Compute the Pearson correlation of paddle trajectories: **px_corr > 0.99 = definitive memorization.** It is physically impossible for a reactive policy to produce identical paddle positions on different brick layouts.

Every model I'd been excited about failed this test. The BeamRider models I'd publicly claimed as reactive? Memorized. The cursor models with 33–50% intervention reversal rates? Memorized. The models with "MULTIPLE_SCRIPTS" on the memorization check? Memorized — false positives from timing variance.

The distribution-vs-argmax confound was universal.

---

## The Breakthrough: Reward What You Want

The insight, in retrospect, is embarrassingly simple.

Every previous approach was a **penalty on scripts**. We were trying to make the undesirable behavior impossible, hoping PPO would stumble into the desirable behavior by elimination. But PPO's objective is `argmax_π E[Σ rewards]` — it optimizes for maximum expected return. In a deterministic environment, a well-tuned script maximizes return. Changing the environment changes *what script* is optimal, not whether the optimum is a script.

Proximity reward changes what the objective *is*.

```python
# ProximityRewardWrapper — applied during training only
# Reads three Atari RAM addresses every step:
#   RAM[72] = paddle_x
#   RAM[99] = ball_x
#   RAM[101] = ball_y

if ball_y > 100:  # ball is descending toward paddle
    distance = abs(paddle_x - ball_x)
    bonus = 0.05 * max(0.0, 1.0 - distance / 80.0)
    reward += bonus
```

A center-hold script gets incidental proximity reward when the ball happens to pass near center — maybe 20% of the maximum. A reactive tracker that follows the ball gets the full bonus on every descent frame. The optimization pressure is unambiguous: **track the ball, get more reward.**

The bonus is tiny compared to brick breaks (0.05 vs 1.0–7.0), but it's dense — it fires every frame the ball is descending, roughly 2,000 frames per game. Dense, consistent signal beats sparse, large rewards for shaping behavior. And because the bonus is only applied during training (eval uses clean Breakout), any ball-tracking observed in evaluation is genuine transferred behavior.

---

## Verification: The Split-Watcher

PPO_124 was trained for 25 million steps on ALE/Breakout-v5 with the proximity reward wrapper. Standard NatureCNN architecture, 32 environments, no sticky actions, no special tricks. The only addition was those three lines.

Then I ran it through the split-watcher: 240 games across three altered brick layouts (left half, right half, random 50%), with and without NoopResetEnv timing offsets.

**0 out of 240 games showed perfect transfer.** Zero. On every previous model tested, at least one game produced identical paddle positions on different layouts. PPO_124 had none.

In the cleanest variant (no NoopResetEnv — zero timing confound), the model achieved **100% score retention** on altered layouts. It cleared every brick, every game, regardless of which bricks were present. A memorized script trained on the full brick wall cannot clear a half-wall — the ball bounces differently, and the script's paddle positions no longer match. PPO_124 adapted.

The intervention gradient — which measures distribution shifts when the ball is artificially teleported — showed a clean dose-response curve peaking at **60% reversal** at 15px displacement, with an AUC of 0.421 (classified as STRONG). Previous "reactive-looking" models had flat, noisy curves because their distributions reacted without their argmax changing. PPO_124's curve has the shape you'd expect from genuine ball-tracking: strongest response at small, physically plausible displacements, tapering off at extreme (physically impossible) offsets.

---

## PPO_126: The Regression

I continued training to 50 million steps to see if more training improved reactivity further. It didn't.

By 50M steps, PPO_126 had regressed to a memorized script — px_corr = 0.95, single 401-point script on all layouts. The best checkpoint was at 47.4M, showing partial layout-specific decoupling (reactive on right-half layouts, scripted on left-half). But by 50M, even that was gone.

**More training does not monotonically improve reactivity.** PPO eventually finds a script that maximizes the *combined* game + proximity objective. The proximity reward makes ball-tracking the optimal behavior for ~25M steps, but given enough training, PPO eventually discovers a script that maximizes both — clearing every brick while also happening to position the paddle near where the ball will be.

Checkpoint selection is critical. Save frequently, verify with the split-watcher.

---

## What This Means

**PPO's objective function was the root cause of universal memorization all along.** In a deterministic environment, `argmax_π E[Σ rewards]` converges to a script because scripts maximize expected return. Every environment modification — sticky actions, cursor wrappers, dynamics randomization — changed what script was optimal, not whether the optimum was a script.

Changing the *reward function* changes what the optimum *is*. Proximity reward made ball-tracking the reward-maximizing behavior.

The principle generalizes: **reward what you want, don't penalize what you don't want.** Dense rewards for intermediate steps of the desired behavior provide better gradients than sparse rewards for outcomes, even when the dense rewards are an order of magnitude smaller.

---

## Key Results (Clean Eval, No Proximity Reward)

| Metric | Result |
|---|---|
| Split-watcher perfect transfers | **0/240** (every previous model: ≥1/9) |
| No-timing ALT score retention | **100%** (clears every layout, every game) |
| Intervention AUC | **0.421** (STRONG dose-response) |
| Peak intervention reversal | **60%** at 15px |
| Stochastic best score | **216** (highest on clean Breakout in project history) |
| det=True memcheck | **MULTIPLE_SCRIPTS** sustained (10/12 checkpoints from 14M–25M) |

---

## Reproducibility

All code is open source:

- **Training script + proximity reward wrapper:** [github.com/mharrell/breakout-reactive-ppo](https://github.com/mharrell/breakout-reactive-ppo)
- **Full experiment history (all 124 runs):** [github.com/mharrell/BreakoutBot](https://github.com/mharrell/BreakoutBot)
- **Verification tools:** `verify_split_watcher.py`, `verify_split_watcher_notiming.py`, `intervention_gradient.py`

Training takes ~6.5 hours on an RTX 3060 Ti (avg 1,084 FPS, 32 environments).

---

*I'm an independent researcher. If you found this useful and have arXiv endorsement privileges in cs.AI or cs.LG, you can endorse me here: [arxiv.org/auth/endorse?x=MUM8BP](https://arxiv.org/auth/endorse?x=MUM8BP)*

*Email: mikey.harrell@gmail.com*
