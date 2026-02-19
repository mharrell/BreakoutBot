
# Breakout RL Agent

A deep reinforcement learning agent trained to play Atari Breakout using PPO.

## Technologies
- Python
- Stable-Baselines3
- Gymnasium
- PyTorch
- CUDA

## How to Run
1. Install dependencies: `pip install stable-baselines3[extra] gymnasium[atari] ale-py autorom`
2. Accept ROM licenses: `autorom --accept-license`
3. Train: `python train.py`
4. Watch: `python watch.py`


## Experiments
- Run 1: Default params, peaked at ~30, entropy collapsed around 2.5M steps
- Run 2: ent_coef=0.01, clip_range=0.1, peaked at 26.5, unstable exploration
- Run 3: ent_coef=0.003, clip_range=0.2 peaked at 30.1 then dipped
- Run 4: (PPO_8) Added policy_kwargs: net_arch=[512, 512]
- Rationale: Runs 1-3 all plateaued and declined around 2 million timesteps 
  regardless of exploration tuning, suggesting the default [64, 64] network 
  lacked capacity to learn more complex strategies. Kept ent_coef=0.003, clip_range=0.2 from run 3
- Run 5: (PPO_9) run 4 stalled out around 25 after 2.5 million timesteps. Decreased learning rate to 1.25e-4
- Run 6: entropy loss dropped quickly in run 5. changed ent_coef=0.006
- 

