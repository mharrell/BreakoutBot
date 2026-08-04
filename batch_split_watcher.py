"""
Batch split-watcher runner — runs verify_split_watcher_notiming.py on all
PPO_124 and PPO_126 checkpoints and saves structured results.

Usage:
    python batch_split_watcher.py
    python batch_split_watcher.py --games 10  (faster, fewer games)
"""
import subprocess
import sys
import os
import json
import re
from datetime import datetime
from pathlib import Path

CHECKPOINTS = [
    # PPO_124 — emergence
    ("PPO_124", "models/PPO_124/checkpoint/latest_checkpoint_5000000_steps.zip", "5M"),
    ("PPO_124", "models/PPO_124/checkpoint/latest_checkpoint_10000000_steps.zip", "10M"),
    ("PPO_124", "models/PPO_124/checkpoint/latest_checkpoint_15000000_steps.zip", "15M"),
    ("PPO_124", "models/PPO_124/best_model.zip", "19.2M (best)"),
    ("PPO_124", "models/PPO_124/checkpoint/latest_checkpoint_20000000_steps.zip", "20M"),
    ("PPO_124", "models/PPO_124/final_model.zip", "25M (final)"),
    # PPO_126 — regression
    ("PPO_126", "models/PPO_126/checkpoint/latest_checkpoint_30001984_steps.zip", "30M"),
    ("PPO_126", "models/PPO_126/checkpoint/latest_checkpoint_35001984_steps.zip", "35M"),
    ("PPO_126", "models/PPO_126/checkpoint/latest_checkpoint_40001984_steps.zip", "40M"),
    ("PPO_126", "models/PPO_126/checkpoint/latest_checkpoint_45001984_steps.zip", "45M"),
    ("PPO_126", "models/PPO_126/best_model.zip", "47.4M (best)"),
    ("PPO_126", "models/PPO_126/final_model.zip", "50M (final)"),
]

OUTPUT_DIR = "recordings/split_watcher_batch"
SCRIPT = "verify_split_watcher_notiming.py"

def parse_output(output: str):
    """Extract structured results from split-watcher output."""
    results = {
        "games": [],
        "verdict": None,
        "verdict_detail": None,
        "n_perfect": None,
        "n_total": None,
        "avg_divergence": None,
        "avg_retention": None,
        "avg_full_score": None,
    }

    # Parse individual game lines
    for line in output.split("\n"):
        # Match: "  RIGHT_HALF game 1: 6000f  |  FULL=403  ALT=223 (55%)  |  ..."
        match = re.match(
            r"\s+(RIGHT_HALF|LEFT_HALF|RANDOM_50)\s+game\s+(\d+):\s+(\d+)f\s+\|\s+"
            r"FULL=([\d.]+)\s+ALT=([\d.]+)\s+\(([\d.]+)%\)\s+\|\s+"
            r"actions diverged:\s+(\d+)/(\d+)\s+\(([\d.]+)%\)\s+"
            r"px_corr=([\d.]+)(.*)",
            line,
        )
        if match:
            results["games"].append({
                "layout": match.group(1),
                "game": int(match.group(2)),
                "frames": int(match.group(3)),
                "full_score": float(match.group(4)),
                "alt_score": float(match.group(5)),
                "score_retention_pct": float(match.group(6)),
                "diverged_frames": int(match.group(7)),
                "compared_frames": int(match.group(8)),
                "divergence_pct": float(match.group(9)),
                "px_corr": float(match.group(10)),
                "perfect_transfer": "PERFECT TRANSFER" in match.group(11),
            })

    # Parse verdict section
    verdict_match = re.search(r"VERDICT:\s*(.+)", output)
    if verdict_match:
        results["verdict"] = verdict_match.group(1).strip()

    # Parse summary stats
    perfect_match = re.search(
        r"Games with perfect transfer.*?:\s*(\d+)/(\d+)", output
    )
    if perfect_match:
        results["n_perfect"] = int(perfect_match.group(1))
        results["n_total"] = int(perfect_match.group(2))

    div_match = re.search(r"Avg action divergence:\s*([\d.]+)%", output)
    if div_match:
        results["avg_divergence"] = float(div_match.group(1))

    ret_match = re.search(r"Avg ALT score retention:\s*([\d.]+)%", output)
    if ret_match:
        results["avg_retention"] = float(ret_match.group(1))

    # Collect verdict detail lines
    verdict_lines = []
    in_verdict = False
    for line in output.split("\n"):
        if "OVERALL VERDICT" in line:
            in_verdict = True
            continue
        if in_verdict:
            stripped = line.strip()
            if stripped.startswith("How to read"):
                break
            if stripped:
                verdict_lines.append(stripped)
    results["verdict_detail"] = verdict_lines

    # Compute aggregate stats
    full_scores = [g["full_score"] for g in results["games"]]
    results["avg_full_score"] = sum(full_scores) / len(full_scores) if full_scores else 0

    return results


def main():
    games = 20
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--games":
            games = int(args[i + 1])
            i += 2
        else:
            i += 1

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_results = []
    summary_rows = []

    for run_name, model_path, label in CHECKPOINTS:
        if not os.path.exists(model_path):
            print(f"SKIP {run_name} {label}: model not found at {model_path}")
            continue

        print(f"\n{'='*70}")
        print(f"RUNNING: {run_name} {label}  ({model_path})")
        print(f"{'='*70}")

        log_file = os.path.join(OUTPUT_DIR, f"{run_name}_{label.replace(' ', '_')}.log")

        try:
            result = subprocess.run(
                [sys.executable, SCRIPT, "--model", model_path, "--games", str(games)],
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour max per checkpoint
                cwd=os.getcwd(),
            )

            output = result.stdout + result.stderr

            # Save raw output
            with open(log_file, "w", encoding="utf-8") as f:
                f.write(output)

            # Parse structured results
            parsed = parse_output(output)
            parsed["run_name"] = run_name
            parsed["label"] = label
            parsed["model_path"] = model_path
            all_results.append(parsed)

            # Print one-line summary
            n_perfect = parsed.get("n_perfect", "?")
            n_total = parsed.get("n_total", "?")
            avg_ret = parsed.get("avg_retention", 0)
            avg_div = parsed.get("avg_divergence", 0)
            verdict = parsed.get("verdict", "PARSE_ERROR")
            print(f"  -> {verdict} | perfect={n_perfect}/{n_total} | "
                  f"retention={avg_ret:.0f}% | divergence={avg_div:.1f}%")

            if result.returncode != 0:
                print(f"  WARNING: exit code {result.returncode}")

        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT after 1 hour")
            all_results.append({
                "run_name": run_name, "label": label, "model_path": model_path,
                "verdict": "TIMEOUT", "error": "Exceeded 1 hour",
            })
        except Exception as e:
            print(f"  ERROR: {e}")
            all_results.append({
                "run_name": run_name, "label": label, "model_path": model_path,
                "verdict": "ERROR", "error": str(e),
            })

    # -------------------------------------------------------------------
    # Write structured JSON
    # -------------------------------------------------------------------
    json_path = os.path.join(OUTPUT_DIR, f"batch_results_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nFull results saved to {json_path}")

    # -------------------------------------------------------------------
    # Print summary table
    # -------------------------------------------------------------------
    print(f"\n{'='*90}")
    print("BATCH SUMMARY — Split-Watcher Reactivity Curve")
    print(f"{'='*90}")
    print(f"{'Model':<10} {'Label':<14} {'Verdict':<20} {'Perfect':>8} {'Retention':>10} {'Divergence':>11} {'Avg FULL':>9}")
    print("-" * 90)

    for r in all_results:
        verdict = r.get("verdict", "ERROR") or "UNKNOWN"
        n_p = r.get("n_perfect", "-")
        n_t = r.get("n_total", "-")
        perfect_str = f"{n_p}/{n_t}" if isinstance(n_p, int) else str(n_p)
        ret = r.get("avg_retention")
        ret_str = f"{ret:.0f}%" if ret is not None else "-"
        div = r.get("avg_divergence")
        div_str = f"{div:.1f}%" if div is not None else "-"
        avg_f = r.get("avg_full_score")
        full_str = f"{avg_f:.0f}" if avg_f else "-"

        # Pad verdict to 20 chars
        print(f"{r['run_name']:<10} {r['label']:<14} {verdict:<20} {perfect_str:>8} {ret_str:>10} {div_str:>11} {full_str:>9}")

    print("-" * 90)
    print(f"\nAll per-checkpoint logs: {OUTPUT_DIR}/")
    print(f"Structured results: {json_path}")


if __name__ == "__main__":
    main()
