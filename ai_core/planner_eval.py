"""Compare the annealing planner with the naive equal-split plan on many random weeks.

The two independent measures are missed deadlines and overload. The objective score is
also reported, but the planner optimises exactly that score, so it is expected to win it.
"""
import random

from ai_core.planner import (PlannerConfig, StudyPlanProblem, Task,
                             equal_split_timetable, plan_metrics, plan_week)

DEFAULT_QUALITY = {"morning": 0.45, "afternoon": 0.55, "evening": 0.70, "night": 0.40}


def random_week(rng, config=None):
    """Build a random busy week: 2 or more tasks, 12 to 20 blocks in total, deadlines on days 1 to 6."""
    config = config if config is not None else PlannerConfig(slot_effectiveness=DEFAULT_QUALITY)
    target = rng.randint(12, 20)
    tasks, blocks = [], 0
    while blocks < target or len(tasks) < 2:
        size = min(rng.randint(2, 8), max(1, 20 - blocks))
        tasks.append(Task(f"Task {len(tasks) + 1}", size * config.block_minutes,
                          rng.randint(1, config.n_days - 1), rng.choice([1.0, 1.0, 1.5, 2.0])))
        blocks += size
        if blocks >= 20:
            break
    return StudyPlanProblem(tasks, config)


def deadlines_feasible(problem):
    """True if every deadline can be met at all: by each day, the blocks due fit in the slots available."""
    cfg = problem.config
    return all(sum(1 for block in problem.blocks if block.deadline_day <= d) <= (d + 1) * len(cfg.slots)
               for d in range(cfg.n_days))


def compare_planners(n_weeks=100, seed=0, iterations=4000):
    """Plan the same random weeks with both methods.

    Returns {"Simulated annealing": [metrics per week], "Equal split": [metrics per week]}.
    """
    if n_weeks < 1:
        raise ValueError("n_weeks must be at least 1")
    rng = random.Random(seed)
    results = {"Simulated annealing": [], "Equal split": []}
    for week in range(n_weeks):
        problem = random_week(rng)
        planned = plan_week(problem, iterations=iterations, seed=seed + week)
        feasible = deadlines_feasible(problem)
        for name, timetable in (("Simulated annealing", planned.timetable),
                                ("Equal split", equal_split_timetable(problem))):
            metrics = plan_metrics(problem, timetable)
            metrics["feasible"] = feasible
            results[name].append(metrics)
    return results


def summarize_comparison(results):
    """One row per method: averages, plus the share of weeks with no missed deadline and no overload.

    Some random weeks have deadlines that cannot all be met by any plan, so deadlines are also
    reported for the weeks where it is possible.
    """
    rows = []
    for name, weeks in results.items():
        n = len(weeks)
        possible = [w for w in weeks if w["feasible"]]
        rows.append({
            "method": name,
            "tasks_missed": sum(w["tasks_missed"] for w in weeks) / n,
            "late_blocks": sum(w["late_blocks"] for w in weeks) / n,
            "plans_clash_free": sum(w["clashes"] == 0 for w in weeks) / n,
            "on_time_when_possible": (sum(w["late_blocks"] == 0 for w in possible) / len(possible)
                                      if possible else None),
            "overload_minutes": sum(w["overload_minutes"] for w in weeks) / n,
            "weeks_no_overload": sum(w["overload_minutes"] == 0 for w in weeks) / n,
            "score": sum(w["total"] for w in weeks) / n,
        })
    return rows


def format_comparison(rows):
    """Format the summary as an aligned Markdown table."""
    header = ["Method", "Tasks missed per week", "Deadlines met when possible",
              "Overload minutes per week", "Weeks with no overload", "Clash-free plans",
              "Objective score"]
    body = [[r["method"], f"{r['tasks_missed']:.2f}",
             "n/a" if r["on_time_when_possible"] is None else f"{r['on_time_when_possible'] * 100:.0f}%",
             f"{r['overload_minutes']:.0f}", f"{r['weeks_no_overload'] * 100:.0f}%",
             f"{r['plans_clash_free'] * 100:.0f}%", f"{r['score']:.0f}"] for r in rows]
    widths = [max(len(row[c]) for row in [header] + body) for c in range(len(header))]

    def line(cells):
        return "| " + " | ".join(cell.ljust(widths[c]) for c, cell in enumerate(cells)) + " |"

    return "\n".join([line(header), "| " + " | ".join("-" * w for w in widths) + " |"]
                     + [line(row) for row in body])