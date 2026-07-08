<<<<<<< HEAD
Start of Breakout bot: 2.17.26 with out of the box params.

Added checkpoint to save progress.

Added watch file to run a game on the fly using the most recent checkpoint.

Changed reporting params to include if checkpoint was saved.


2.18.26
continued training. At 4 million timesteps rew_mean dropped to 23 from 30 earlier in the day. 

Stopped training. Renamed current best to best_model_marki.zip

Restarted training with new params.

```
model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log="./tensorboard/",
    n_steps=128,
    batch_size=256,
    n_epochs=4,
    gamma=0.99,
    learning_rate=2.5e-4,
    ent_coef=0.01,        # encourages exploration, default is 0.0
    vf_coef=0.5,          # how much to weight value function loss
    clip_range=0.1,       # tighter clipping for more stable updates
)
```

Progress was going well but then declined again. Going to split the difference. 

```
model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log="./tensorboard/",
    n_steps=128,
    batch_size=256,
    n_epochs=4,
    gamma=0.99,
    learning_rate=2.5e-4,
    ent_coef=0.003,
    vf_coef=0.5,
    clip_range=0.2,
)
```
=======
Start of Breakout bot: 2.17.26 with out of the box params.

Added checkpoint to save progress.

Added watch file to run a game on the fly using the most recent checkpoint.

Changed reporting params to include if checkpoint was saved.


2.18.26
continued training. At 4 million timesteps rew_mean dropped to 23 from 30 earlier in the day. 

Stopped training. Renamed current best to best_model_marki.zip

Restarted training with new params.

```
model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log="./tensorboard/",
    n_steps=128,
    batch_size=256,
    n_epochs=4,
    gamma=0.99,
    learning_rate=2.5e-4,
    ent_coef=0.01,        # encourages exploration, default is 0.0
    vf_coef=0.5,          # how much to weight value function loss
    clip_range=0.1,       # tighter clipping for more stable updates
)
```

>>>>>>> 1b0536408e2c5b23e1fd0aa7fa002620a1b19fbc
