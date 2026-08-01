"""Batch run split-watcher verification across multiple models."""
import subprocess
import sys

MODELS = {
    "PPO_111": "./models/PPO_111/best_model.zip",
    "PPO_112": "./models/PPO_112/best_model.zip",
    "PPO_113": "./models/PPO_113/best_model.zip",
    "PPO_114": "./models/PPO_114/best_model.zip",
    "PPO_116": "./models/PPO_116/best_model.zip",
    "PPO_117": "./models/PPO_117/best_model.zip",
}

# PPO_115 already done

for name, path in MODELS.items():
    print(f"\n{'#'*70}")
    print(f"# {name}")
    print(f"{'#'*70}")
    subprocess.run([
        sys.executable, "verify_split_watcher.py",
        "--model", path,
        "--games", "3",
    ])
