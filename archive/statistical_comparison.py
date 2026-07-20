"""
Statistical comparison utility for BreakoutBot funnel evaluations.

Computes bootstrap confidence intervals and significance tests for
comparing two models' 10k-game single-env evaluations.

Addresses FLAWS.md F-010 (funnel rate CIs) and provides a reusable
tool for future experiment comparisons.

Usage:
    python statistical_comparison.py recordings/PPO_30b_funnel_log.csv recordings/PPO_31b_funnel_log.csv

Outputs:
    - Bootstrap 95% CI on mean difference
    - Bootstrap 95% CI on median difference
    - Fisher's exact test on zero-score rates
    - Binomial 95% CIs on funnel rates
    - Score distribution comparison table
"""
import csv
import sys
import numpy as np
from scipy import stats as scipy_stats


def load_scores(csv_path):
    """Load scores from a funnel log CSV. Returns list of floats."""
    scores = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Empty file: {csv_path}")

        # Find the 'real_score' column
        try:
            score_col = header.index("real_score")
        except ValueError:
            # Try 'score' as fallback
            try:
                score_col = header.index("score")
            except ValueError:
                raise ValueError(
                    f"Cannot find 'real_score' or 'score' column in {csv_path}. "
                    f"Columns: {header}"
                )

        for row in reader:
            if row and len(row) > score_col:
                try:
                    scores.append(float(row[score_col]))
                except (ValueError, IndexError):
                    continue

    return scores


def bootstrap_ci(data, statistic=np.mean, n_bootstrap=10000, alpha=0.05):
    """Compute bootstrap confidence interval for a statistic."""
    n = len(data)
    bootstrapped = np.array([
        statistic(np.random.choice(data, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])
    lower = np.percentile(bootstrapped, 100 * alpha / 2)
    upper = np.percentile(bootstrapped, 100 * (1 - alpha / 2))
    return lower, upper


def bootstrap_diff_ci(data_a, data_b, statistic=np.mean, n_bootstrap=10000, alpha=0.05):
    """Bootstrap CI on the difference in a statistic between two samples."""
    n_a, n_b = len(data_a), len(data_b)
    diffs = np.array([
        statistic(np.random.choice(data_a, size=n_a, replace=True)) -
        statistic(np.random.choice(data_b, size=n_b, replace=True))
        for _ in range(n_bootstrap)
    ])
    lower = np.percentile(diffs, 100 * alpha / 2)
    upper = np.percentile(diffs, 100 * (1 - alpha / 2))
    return lower, upper


def binomial_ci(count, n, alpha=0.05):
    """Clopper-Pearson exact binomial confidence interval."""
    from scipy.stats import beta as beta_dist
    if count == 0:
        lower = 0.0
    else:
        lower = beta_dist.ppf(alpha / 2, count, n - count + 1)
    if count == n:
        upper = 1.0
    else:
        upper = beta_dist.ppf(1 - alpha / 2, count + 1, n - count)
    return lower, upper


def summarize(scores, name):
    """Compute summary statistics for a set of scores."""
    arr = np.array(scores)
    n = len(arr)
    zero_count = int(np.sum(arr == 0))
    n_400 = int(np.sum(arr >= 400))

    return {
        "name": name,
        "n": n,
        "avg": np.mean(arr),
        "median": np.median(arr),
        "std": np.std(arr),
        "min": np.min(arr),
        "max": np.max(arr),
        "zero_count": zero_count,
        "zero_rate": zero_count / n,
        "n_400": n_400,
        "funnel_rate": n_400 / n,
    }


def main():
    if len(sys.argv) != 3:
        print("Usage: python statistical_comparison.py <funnel_log_a.csv> <funnel_log_b.csv>")
        print("Example: python statistical_comparison.py recordings/PPO_30b_funnel_log.csv recordings/PPO_31b_funnel_log.csv")
        sys.exit(1)

    path_a, path_b = sys.argv[1], sys.argv[2]

    print("=" * 65)
    print("Statistical Comparison -- BreakoutBot Funnel Evaluations")
    print("=" * 65)

    # Load data
    scores_a = load_scores(path_a)
    scores_b = load_scores(path_b)

    # Derive model names from paths
    name_a = path_a.split("PPO_")[1].split("_funnel")[0] if "PPO_" in path_a else "Model A"
    name_b = path_b.split("PPO_")[1].split("_funnel")[0] if "PPO_" in path_b else "Model B"

    print(f"\nModel A (PPO_{name_a}): {len(scores_a)} games from {path_a}")
    print(f"Model B (PPO_{name_b}): {len(scores_b)} games from {path_b}")

    # Summaries
    s_a = summarize(scores_a, f"PPO_{name_a}")
    s_b = summarize(scores_b, f"PPO_{name_b}")

    print(f"\n{'-' * 50}")
    print("DESCRIPTIVE STATISTICS")
    print(f"{'-' * 50}")

    print(f"\n{'Metric':<20} {'PPO_' + name_a:<20} {'PPO_' + name_b:<20}")
    print(f"{'-' * 60}")
    print(f"{'N games':<20} {s_a['n']:<20} {s_b['n']:<20}")
    print(f"{'Average':<20} {s_a['avg']:<20.1f} {s_b['avg']:<20.1f}")
    print(f"{'Median':<20} {s_a['median']:<20.0f} {s_b['median']:<20.0f}")
    print(f"{'Std Dev':<20} {s_a['std']:<20.1f} {s_b['std']:<20.1f}")
    print(f"{'Min':<20} {s_a['min']:<20.0f} {s_b['min']:<20.0f}")
    print(f"{'Max':<20} {s_a['max']:<20.0f} {s_b['max']:<20.0f}")
    print(f"{'Zero-score':<20} {s_a['zero_count']}/{s_a['n']} ({s_a['zero_rate']*100:.1f}%)    "
          f"{s_b['zero_count']}/{s_b['n']} ({s_b['zero_rate']*100:.1f}%)")
    print(f"{'Funnel (400+)':<20} {s_a['n_400']}/{s_a['n']} ({s_a['funnel_rate']*100:.2f}%)    "
          f"{s_b['n_400']}/{s_b['n']} ({s_b['funnel_rate']*100:.2f}%)")

    # --- Inferential Statistics ---
    print(f"\n{'-' * 50}")
    print("INFERENTIAL STATISTICS")
    print(f"{'-' * 50}")

    # Mean difference
    mean_diff_ci = bootstrap_diff_ci(scores_a, scores_b, statistic=np.mean)
    mean_diff = s_a["avg"] - s_b["avg"]
    print(f"\nMean difference (A - B): {mean_diff:+.1f}")
    print(f"  95% bootstrap CI: [{mean_diff_ci[0]:+.1f}, {mean_diff_ci[1]:+.1f}]")
    if mean_diff_ci[0] > 0:
        print(f"  -> A significantly higher (CI entirely above zero)")
    elif mean_diff_ci[1] < 0:
        print(f"  -> B significantly higher (CI entirely below zero)")
    else:
        print(f"  -> No significant difference (CI includes zero)")

    # Median difference
    median_diff_ci = bootstrap_diff_ci(scores_a, scores_b, statistic=np.median)
    median_diff = s_a["median"] - s_b["median"]
    print(f"\nMedian difference (A - B): {median_diff:+.0f}")
    print(f"  95% bootstrap CI: [{median_diff_ci[0]:+.0f}, {median_diff_ci[1]:+.0f}]")
    if median_diff_ci[0] > 0:
        print(f"  -> A significantly higher")
    elif median_diff_ci[1] < 0:
        print(f"  -> B significantly higher")
    else:
        print(f"  -> No significant difference")

    # Zero-score rate comparison (Fisher's exact test)
    table = [[s_a["zero_count"], s_a["n"] - s_a["zero_count"]],
             [s_b["zero_count"], s_b["n"] - s_b["zero_count"]]]
    odds_ratio, fisher_p = scipy_stats.fisher_exact(table)
    print(f"\nZero-score rate comparison:")
    print(f"  A: {s_a['zero_rate']*100:.1f}%  B: {s_b['zero_rate']*100:.1f}%")
    print(f"  Fisher's exact p = {fisher_p:.4f}")
    if fisher_p < 0.05:
        print(f"  -> Significant difference in zero-score rates")
    else:
        print(f"  -> No significant difference in zero-score rates")

    # Funnel rate confidence intervals
    fa_ci = binomial_ci(s_a["n_400"], s_a["n"])
    fb_ci = binomial_ci(s_b["n_400"], s_b["n"])
    print(f"\nFunnel rate (400+) binomial 95% CIs:")
    print(f"  A ({s_a['n_400']}/{s_a['n']}): [{fa_ci[0]*100:.3f}%, {fa_ci[1]*100:.3f}%]")
    print(f"  B ({s_b['n_400']}/{s_b['n']}): [{fb_ci[0]*100:.3f}%, {fb_ci[1]*100:.3f}%]")

    # Check CI overlap
    if fa_ci[1] < fb_ci[0]:
        print(f"  -> B has significantly higher funnel rate (CIs don't overlap)")
    elif fb_ci[1] < fa_ci[0]:
        print(f"  -> A has significantly higher funnel rate (CIs don't overlap)")
    else:
        print(f"  -> No significant difference (CIs overlap)")

    # --- Score Distribution ---
    print(f"\n{'-' * 50}")
    print("SCORE DISTRIBUTION")
    print(f"{'-' * 50}")

    thresholds = [0, 5, 10, 20, 30, 40, 50, 60, 100]
    print(f"\n{'<=Threshold':<12} {'PPO_' + name_a + ' %':<15} {'PPO_' + name_b + ' %':<15}")
    print(f"{'-' * 45}")
    for t in thresholds:
        pct_a = np.mean(np.array(scores_a) <= t) * 100
        pct_b = np.mean(np.array(scores_b) <= t) * 100
        print(f"{'<=' + str(t):<12} {pct_a:<15.1f} {pct_b:<15.1f}")

    percentiles = [5, 10, 25, 50, 75, 90, 95, 99]
    print(f"\n{'Percentile':<12} {'PPO_' + name_a:<15} {'PPO_' + name_b:<15}")
    print(f"{'-' * 45}")
    for p in percentiles:
        val_a = np.percentile(scores_a, p)
        val_b = np.percentile(scores_b, p)
        print(f"{'P' + str(p):<12} {val_a:<15.0f} {val_b:<15.0f}")

    # --- Conditional stats ---
    nonzero_a = [s for s in scores_a if s > 0]
    nonzero_b = [s for s in scores_b if s > 0]
    print(f"\n{'-' * 50}")
    print("CONDITIONAL STATS (non-zero games only)")
    print(f"{'-' * 50}")
    print(f"\n{'Metric':<20} {'PPO_' + name_a:<20} {'PPO_' + name_b:<20}")
    print(f"{'-' * 60}")
    print(f"{'Non-zero count':<20} {len(nonzero_a):<20} {len(nonzero_b):<20}")
    print(f"{'Non-zero avg':<20} {np.mean(nonzero_a):<20.1f} {np.mean(nonzero_b):<20.1f}")
    print(f"{'Non-zero median':<20} {np.median(nonzero_a):<20.0f} {np.median(nonzero_b):<20.0f}")

    zero_rate_diff = abs(s_a["zero_rate"] - s_b["zero_rate"]) * 100
    if zero_rate_diff > 5:
        print(f"\n[!] Zero-score rates differ by {zero_rate_diff:.1f} percentage points. "
              f"Non-zero averages are not directly comparable -- the model with more zeros "
              f"gets a larger mechanical boost from excluding its worst games.")

    print(f"\n{'=' * 65}")
    print("Note: Funnel rate CIs use Clopper-Pearson exact binomial.")
    print("Bootstrap CIs use 10,000 resamples.")
    print("For rare events (funnel rate), differences of <5 events at N=10,000")
    print("are not statistically significant regardless of CI method.")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
