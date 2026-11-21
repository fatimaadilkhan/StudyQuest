"""Compare the annealing planner with the naive equal-split plan on random weeks.

Usage (from the project root):
    python run_planner_evaluation.py
    python run_planner_evaluation.py --weeks 500
Writes the table to evaluation_output/planner_comparison.md as well as printing it.
"""
import argparse
import os

from ai_core.planner_eval import compare_planners, format_comparison, summarize_comparison


def main():
    """Run the comparison, print the table and save it as Markdown."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weeks", type=int, default=200, help="random weeks to plan")
    parser.add_argument("--iterations", type=int, default=4000, help="annealing steps per week")
    args = parser.parse_args()

    results = compare_planners(n_weeks=args.weeks, iterations=args.iterations)
    table = format_comparison(summarize_comparison(results))
    print(table)

    os.makedirs("evaluation_output", exist_ok=True)
    path = os.path.join("evaluation_output", "planner_comparison.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Planner comparison\n\n{args.weeks} random weeks, "
                f"{args.iterations} annealing steps per week.\n\n{table}\n")
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()