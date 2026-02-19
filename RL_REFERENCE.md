# PPO Training Reference Guide
### Breakout RL Agent — Levers & Outputs

---

## Part 1: Training Levers

These are the parameters you control in `train.py`. Each one affects a different aspect of how the agent learns. The key principle: **change one thing at a time** so you can interpret the results.

| Parameter | What It Does | Effect on Training |
|-----------|-------------|-------------------|
| `ent_coef` | Entropy coefficient — adds bonus reward for taking varied actions | Higher = more exploration, prevents early convergence. Too high = unstable training |
| `learning_rate` | How big each update step is | Lower = more stable but slower learning. Higher = faster but can overshoot and destabilize |
| `clip_range` | Limits how much the policy can change in one update | Lower = more conservative updates, more stable. Higher = allows bigger changes per step |
| `n_steps` | How many steps to collect before each update | Higher = more data per update, smoother gradients but slower iteration |
| `batch_size` | How much data to use per gradient update | Higher = more stable updates but more memory usage |
| `n_epochs` | How many times to reuse collected data per update | Higher = more efficient data use but risk of overfitting to recent experience |
| `gamma` | Discount factor — how much to value future vs immediate rewards | Higher = agent thinks further ahead. Lower = more focused on immediate rewards |
| `net_arch` | Size of fully connected layers after the CNN | Larger = more capacity to learn complex strategies but needs more careful tuning |
| `vf_coef` | How much to weight value function loss vs policy loss | Higher = more focus on accurate reward prediction |
| `n_envs` | Number of parallel game environments | More = faster experience collection, more diverse training data |
| `total_timesteps` | Total training duration | More = more time to learn but diminishing returns after convergence |

---

## Experiment History

| Run | TensorBoard | Key Changes | Peak Reward | Notes |
|-----|-------------|-------------|-------------|-------|
| Run 1 | PPO_5 | Default params, ent_coef=0.0, lr=2.5e-4, net=[64,64] | ~30 | Entropy collapsed ~2M steps |
| Run 2 | PPO_6 | ent_coef=0.01, clip_range=0.1 | ~26 | Unstable, tanked at 2M steps |
| Run 3 | PPO_7 | ent_coef=0.003, clip_range=0.2 | ~31 | Best so far, declined after 2M |
| Run 4 | PPO_8 | Added net=[512,512] | ~22 | Large network needs lower lr |
| Run 5 | PPO_9 | net=[512,512], lr=1.25e-4 | ~21 | Entropy still collapsed early |
| Run 6 | PPO_10 | net=[512,512], lr=1.25e-4, ent_coef=0.006 | In progress | — |

---

## Part 2: Training Outputs

These are the metrics TensorBoard and the terminal show during training. Understanding them helps you decide when to intervene and what to change.

| Output | What It Means | Healthy Range / What to Watch |
|--------|--------------|-------------------------------|
| `ep_rew_mean` | Average score per episode during training (noisy — agent is still exploring) | Should climb over time. Dips are normal during exploration phases |
| `ep_len_mean` | Average game length in frames | Longer = agent keeping ball alive longer. Sudden drops signal a problem |
| `eval/mean_reward` | Average score during deterministic evaluation — most reliable metric | Your ground truth. Should trend upward. Plateau or decline = time to intervene |
| `fps` | Frames processed per second across all environments | Higher is better. Sudden drops suggest GPU throttling or background processes |
| `entropy_loss` | How much the agent is exploring vs exploiting | Early: -1.0 to -1.5. Mid: -0.3 to -0.5. Late: -0.1 to -0.3. Near 0 = collapsed |
| `explained_variance` | How accurately the agent predicts future rewards | Closer to 1.0 is better. Below 0.5 = value function is struggling |
| `approx_kl` | How much the policy changed this update | Should stay below 0.05. Spikes above 0.1 indicate unstable updates |
| `clip_fraction` | How often PPO's safety clamp triggered | 0.05-0.15 is healthy. Consistently high = learning rate may be too aggressive |
| `loss` | Combined training loss | Should generally decrease over time. Spikes can indicate instability |
| `value_loss` | How wrong the reward predictions were | Should decrease over time as predictions improve |
| `policy_gradient_loss` | How much the policy is being pushed to change | Should be small and stable. Large values indicate aggressive updates |
| `n_updates` | Total number of gradient updates performed | Useful for tracking how much training has actually happened |

---

## Healthy Entropy Loss by Training Stage

| Stage | Steps | Healthy Range | What It Means |
|-------|-------|--------------|---------------|
| Early | 0 - 500k | -1.0 to -1.5 | Agent exploring freely, not committed to any strategy |
| Mid | 500k - 2M | -0.3 to -0.5 | Narrowing down good strategies while still exploring |
| Late | 2M+ | -0.1 to -0.3 | Some confidence is fine as long as reward is still climbing |
| 🚨 Red flag | Any | Near 0.0 before 1M | Premature convergence — agent got confident too early |

---

## Decision Framework

| What You See | What It Means | What To Do |
|-------------|--------------|-----------|
| Reward climbing steadily | Healthy training | Keep going, don't touch anything |
| Reward plateau, healthy entropy | May need more time | Wait it out, check again in 500k steps |
| Reward plateau, collapsed entropy | Premature convergence | Increase `ent_coef` or restart with new params |
| Reward declining after peak | Instability or collapse | Check if both rollout AND eval are declining. If yes, stop and adjust |
| `approx_kl` consistently above 0.05 | Updates too aggressive | Reduce `learning_rate` |
| fps suddenly drops | Hardware throttling | Close background apps, check GPU temp with `nvidia-smi` |
| Both rollout and eval declining | Real regression | Stop training, diagnose before continuing |
| Rollout dips but eval holds | Exploration phase | Normal, wait it out |
