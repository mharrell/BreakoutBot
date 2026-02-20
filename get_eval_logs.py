import numpy as np

log_path = "./logs/evaluations.npz"
data = np.load(log_path)

timesteps = data["timesteps"]
results = data["results"]

for i, (step, scores) in enumerate(zip(timesteps, results)):
    mean_score = np.mean(scores)
    print(f"Step {step:>10,} | Mean Reward: {mean_score:.2f}")