"""Run the Day 8 evaluation and write tables and plots to evaluation_output/.

Usage (from the project root):
    python run_evaluation.py
    python run_evaluation.py --seeds 1000 --horizon 300
"""
import argparse
import csv
import os

try:
    import matplotlib

    matplotlib.use("Agg")  # write image files; no window needed
    import matplotlib.pyplot as plt
except ImportError:  # plots are optional; tables and CSV curves are still written
    plt = None

from evaluation import (POLICY_FACTORIES, SCENARIOS, best_arm_rate_curve,
                        cumulative_regret_curve, evaluate, format_table, summarize)
from ai_core.thompson import ThompsonBandit

OUTPUT_DIR = "evaluation_output"

# Thompson sampling with a noise assumption that matches the simulated student (0.15)
SENSITIVITY_POLICIES = {
    "Thompson (obs_std 0.3, default)": lambda arms, seed: ThompsonBandit(arms),
    "Thompson (obs_std 0.15)": lambda arms, seed: ThompsonBandit(arms, obs_std=0.15),
    "Epsilon-greedy": POLICY_FACTORIES["Epsilon-greedy"],
}


def plot_scenario(name, results, path):
    """Save a two-panel figure: cumulative regret and best-option rate per policy."""
    fig, (ax_regret, ax_rate) = plt.subplots(1, 2, figsize=(12, 4.5))
    for policy, result in results.items():
        ax_regret.plot(range(1, len(result["regret"][0]) + 1),
                       cumulative_regret_curve(result), label=policy)
        ax_rate.plot(range(1, len(result["hits"][0]) + 1),
                     best_arm_rate_curve(result), label=policy)
    ax_regret.set_title("Cumulative regret (lower is better)")
    ax_regret.set_xlabel("Session")
    ax_regret.set_ylabel("Regret")
    ax_rate.set_title("How often the best option is picked (higher is better)")
    ax_rate.set_xlabel("Session")
    ax_rate.set_ylabel("Share of students picking it")
    ax_rate.set_ylim(0, 1.02)
    for ax in (ax_regret, ax_rate):
        ax.grid(alpha=0.3)
    ax_rate.legend(loc="center right")
    fig.suptitle(f"Scenario: {name}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_curves_csv(results, path):
    """Save the curves as CSV (one row per session) so they can be charted in Excel."""
    names = list(results)
    regret = {n: cumulative_regret_curve(results[n]) for n in names}
    rate = {n: best_arm_rate_curve(results[n]) for n in names}
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["session"] + [f"{n} regret" for n in names]
                        + [f"{n} best-option rate" for n in names])
        for i in range(len(regret[names[0]])):
            writer.writerow([i + 1] + [round(regret[n][i], 4) for n in names]
                            + [round(rate[n][i], 4) for n in names])


def main():
    """Evaluate every scenario, print the tables and save plots and a Markdown report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=500, help="simulated students per policy")
    parser.add_argument("--horizon", type=int, default=200, help="sessions per student")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report = [f"# Evaluation results\n\n{args.seeds} simulated students per policy, "
              f"{args.horizon} sessions each.\n"]

    for name, make_student in SCENARIOS.items():
        results = evaluate(make_student, horizon=args.horizon, n_seeds=args.seeds)
        table = format_table(summarize(results))
        stem = name.replace(" ", "_")
        write_curves_csv(results, os.path.join(OUTPUT_DIR, stem + ".csv"))
        print(f"\n{name}\n{table}")
        report.append(f"## {name}\n\n{table}\n")
        if plt is not None:
            plot_scenario(name, results, os.path.join(OUTPUT_DIR, stem + ".png"))
            report.append(f"![{name}]({stem}.png)\n")

    report.append("## Sensitivity to the noise assumption\n")
    for name, make_student in SCENARIOS.items():
        results = evaluate(make_student, SENSITIVITY_POLICIES,
                           horizon=args.horizon, n_seeds=args.seeds)
        table = format_table(summarize(results))
        print(f"\nSensitivity: {name}\n{table}")
        report.append(f"### {name}\n\n{table}\n")

    with open(os.path.join(OUTPUT_DIR, "results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"\nSaved plots and results.md to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()