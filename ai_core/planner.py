"""Weekly study planner: timetable representation, objective function and simulated annealing.

A timetable is a list with one (day, slot) pair per study block, in the same order as
StudyPlanProblem.blocks. Days and slots are 0-based indexes (slot names live in
PlannerConfig.slots). The objective is a score to maximise:

    score = urgency-weighted effective minutes
            - clash penalty      (two blocks in the same day and slot; set highest so a
                                  double-booked plan is never preferred to a late one)
            - overload penalty   (minutes above the daily cap)
            - rest penalty       (too few completely free days)
            - late penalty       (blocks placed after their deadline, which also earn nothing;
                                  set above the others so a missed deadline costs more than a long day)

Also here: a naive equal-split baseline, per-block reasons, and replanning after missed sessions.
"""
import math
import random
from collections import Counter
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Task:
    """A piece of work to schedule: total study minutes, last day it can be done, and importance.

    deadline_day is a 0-based day index: 0 is the first day of the plan.
    """

    name: str
    minutes: int
    deadline_day: int
    importance: float = 1.0


@dataclass
class PlannerConfig:
    """Planning horizon, block size, limits and penalty weights.

    slot_effectiveness maps each slot name to the expected fruitfulness (0 to 1) of studying then;
    None means 0.5 for every slot. effectiveness_from_optimizer() fills it from learned data.
    """

    n_days: int = 7
    slots: tuple = ("morning", "afternoon", "evening", "night")
    block_minutes: int = 45
    max_daily_minutes: int = 135
    min_rest_days: int = 1
    slot_effectiveness: dict = None
    clash_penalty: float = 300.0
    overload_penalty_per_minute: float = 1.0
    rest_penalty: float = 60.0
    late_penalty: float = 200.0


@dataclass
class PlanResult:
    """Outcome of an annealing run: best timetable, its score and breakdown, and run statistics."""

    timetable: list
    score: float
    breakdown: dict
    iterations: int
    accepted: int
    history: list


class StudyPlanProblem:
    """Tasks expanded into study blocks, plus the objective function over timetables."""

    def __init__(self, tasks, config=None):
        self.config = config if config is not None else PlannerConfig()
        cfg = self.config
        if cfg.n_days < 1 or not cfg.slots or cfg.block_minutes < 1:
            raise ValueError("n_days, slots and block_minutes must be positive")
        if not tasks:
            raise ValueError("tasks must not be empty")

        names = [t.name for t in tasks]
        if len(set(names)) != len(names):
            raise ValueError("task names must be unique")
        for t in tasks:
            if not t.name or t.minutes <= 0 or t.importance <= 0:
                raise ValueError(f"invalid task: {t!r}")
            if not 0 <= t.deadline_day < cfg.n_days:
                raise ValueError(f"deadline_day of {t.name!r} must be between 0 and {cfg.n_days - 1}")

        self.tasks = list(tasks)
        self.blocks = [t for t in self.tasks for _ in range(math.ceil(t.minutes / cfg.block_minutes))]
        if len(self.blocks) > cfg.n_days * len(cfg.slots):
            raise ValueError(f"{len(self.blocks)} blocks cannot fit in "
                             f"{cfg.n_days * len(cfg.slots)} slots")

        effectiveness = cfg.slot_effectiveness
        self.effectiveness = ({s: 0.5 for s in cfg.slots} if effectiveness is None
                              else {s: float(effectiveness[s]) for s in cfg.slots})
        # Earlier deadline means higher urgency: 1 + (days left after the deadline) / n_days.
        self.urgency = {t.name: t.importance * (1 + (cfg.n_days - 1 - t.deadline_day) / cfg.n_days)
                        for t in self.tasks}

    def random_timetable(self, rng):
        """Place every block in a random day and slot (clashes allowed)."""
        return [(rng.randrange(self.config.n_days), rng.randrange(len(self.config.slots)))
                for _ in self.blocks]

    def is_valid(self, timetable):
        """True if the timetable has one in-range position per block and no two blocks clash."""
        cfg = self.config
        if len(timetable) != len(self.blocks):
            return False
        if any(not (0 <= d < cfg.n_days and 0 <= s < len(cfg.slots)) for d, s in timetable):
            return False
        return len(set(timetable)) == len(timetable)

    def violations(self, timetable):
        """Count clashes, late blocks, overloaded days and missing rest days."""
        cfg = self.config
        occupied = Counter(timetable)
        minutes = Counter()
        late = 0
        for task, (day, _slot) in zip(self.blocks, timetable):
            minutes[day] += cfg.block_minutes
            if day > task.deadline_day:
                late += 1
        rest_days = cfg.n_days - len(minutes)
        return {
            "clashes": sum(c - 1 for c in occupied.values() if c > 1),
            "late_blocks": late,
            "overload_minutes": sum(max(0, m - cfg.max_daily_minutes) for m in minutes.values()),
            "missing_rest_days": max(0, cfg.min_rest_days - rest_days),
        }

    def score(self, timetable):
        """Return the objective breakdown; the 'total' entry is the number to maximise."""
        cfg = self.config
        effective = 0.0
        for task, (day, slot) in zip(self.blocks, timetable):
            if day <= task.deadline_day:  # a late block earns nothing
                effective += (cfg.block_minutes * self.effectiveness[cfg.slots[slot]]
                              * self.urgency[task.name])
        v = self.violations(timetable)
        parts = {
            "effective_minutes": effective,
            "clash_penalty": v["clashes"] * cfg.clash_penalty,
            "overload_penalty": v["overload_minutes"] * cfg.overload_penalty_per_minute,
            "rest_penalty": v["missing_rest_days"] * cfg.rest_penalty,
            "late_penalty": v["late_blocks"] * cfg.late_penalty,
        }
        parts["total"] = (parts["effective_minutes"] - parts["clash_penalty"]
                          - parts["overload_penalty"] - parts["rest_penalty"]
                          - parts["late_penalty"])
        return parts

    def neighbour(self, timetable, rng):
        """Propose a small change: move one block somewhere else, or swap two blocks."""
        candidate = list(timetable)
        if len(candidate) >= 2 and rng.random() < 0.5:
            i, j = rng.sample(range(len(candidate)), 2)
            candidate[i], candidate[j] = candidate[j], candidate[i]
        else:
            i = rng.randrange(len(candidate))
            candidate[i] = (rng.randrange(self.config.n_days),
                            rng.randrange(len(self.config.slots)))
        return candidate


def anneal(problem, iterations=6000, start_temp=30.0, end_temp=0.1, seed=None,
           initial=None, record_every=100):
    """Search for a high-scoring timetable by simulated annealing.

    Each step proposes a neighbouring timetable and always accepts an improvement, or
    accepts a worse one with probability exp(change / temperature). The temperature
    cools geometrically from start_temp to end_temp. Returns the best timetable seen.
    """
    if iterations < 1 or record_every < 1:
        raise ValueError("iterations and record_every must be at least 1")
    if not 0 < end_temp < start_temp:
        raise ValueError("need 0 < end_temp < start_temp")
    rng = random.Random(seed)
    if initial is not None and len(initial) != len(problem.blocks):
        raise ValueError("initial timetable has the wrong number of blocks")

    current = list(initial) if initial is not None else problem.random_timetable(rng)
    current_score = problem.score(current)["total"]
    best, best_score = list(current), current_score
    cooling = (end_temp / start_temp) ** (1 / max(1, iterations - 1))
    temperature = start_temp
    accepted = 0
    history = []

    for step in range(iterations):
        candidate = problem.neighbour(current, rng)
        candidate_score = problem.score(candidate)["total"]
        change = candidate_score - current_score
        if change >= 0 or rng.random() < math.exp(change / temperature):
            current, current_score = candidate, candidate_score
            accepted += 1
            if current_score > best_score:
                best, best_score = list(current), current_score
        temperature *= cooling
        if step % record_every == 0:
            history.append(best_score)

    return PlanResult(best, best_score, problem.score(best), iterations, accepted, history)


def plan_week(problem, iterations=6000, seed=None, **anneal_options):
    """Plan a week: anneal starting from the equal-split plan.

    Starting from the naive plan means the result is never worse than it, and keeps deadlines
    met whenever the naive plan met them, while annealing removes overload and uses the best slots.
    """
    return anneal(problem, iterations=iterations, seed=seed,
                  initial=equal_split_timetable(problem), **anneal_options)


def random_baseline(problem, n=300, seed=None):
    """Score `n` random timetables; returns {'mean', 'best', 'valid_share'}."""
    rng = random.Random(seed)
    scores, valid = [], 0
    for _ in range(n):
        timetable = problem.random_timetable(rng)
        scores.append(problem.score(timetable)["total"])
        valid += problem.is_valid(timetable)
    return {"mean": sum(scores) / n, "best": max(scores), "valid_share": valid / n}


def effectiveness_from_optimizer(optimizer, slots=("morning", "afternoon", "evening", "night")):
    """Use what the time-slot bandit has learned: expected fruitfulness (0 to 1) for each slot."""
    return {s: min(1.0, max(0.0, optimizer.slot_bandit.posterior(s)[0])) for s in slots}


def format_timetable(problem, timetable, day_names=None):
    """Draw the timetable as a text grid: one row per day, one column per slot.

    Rows are labelled "Day 0", "Day 1", ... unless day_names (one label per day) is given.
    """
    cfg = problem.config
    grid = {pos: task.name for task, pos in zip(problem.blocks, timetable)}
    clashes = Counter(timetable)
    width = max([len(t.name) for t in problem.tasks] + [len(s) for s in cfg.slots]) + 2
    labels = list(day_names) if day_names is not None else [f"Day {d}" for d in range(cfg.n_days)]
    if len(labels) != cfg.n_days:
        raise ValueError("day_names needs one label per day")
    label_width = max(len(label) for label in labels) + 2
    lines = ["".ljust(label_width) + "".join(s.ljust(width) for s in cfg.slots)]
    for day in range(cfg.n_days):
        cells = []
        for slot in range(len(cfg.slots)):
            text = grid.get((day, slot), "-")
            if clashes[(day, slot)] > 1:
                text += " (clash)"
            cells.append(text.ljust(width))
        lines.append(labels[day].ljust(label_width) + "".join(cells))
    return "\n".join(lines)


def plan_metrics(problem, timetable):
    """Measure a timetable on the things a student cares about.

    Returns late_blocks, tasks_missed (tasks with at least one late block), overload_minutes,
    overloaded_days, clashes, rest_days, effective_minutes and the objective total.
    """
    cfg = problem.config
    minutes = Counter(day for day, _slot in timetable)
    late_tasks = {task.name for task, (day, _slot) in zip(problem.blocks, timetable)
                  if day > task.deadline_day}
    v = problem.violations(timetable)
    scored = problem.score(timetable)
    return {
        "late_blocks": v["late_blocks"],
        "tasks_missed": len(late_tasks),
        "overload_minutes": v["overload_minutes"],
        "overloaded_days": sum(1 for d in minutes if minutes[d] * cfg.block_minutes > cfg.max_daily_minutes),
        "clashes": v["clashes"],
        "rest_days": cfg.n_days - len(minutes),
        "effective_minutes": scored["effective_minutes"],
        "total": scored["total"],
    }


def equal_split_timetable(problem):
    """The naive baseline: spread each task's blocks evenly over the days up to its deadline.

    Tasks are handled in deadline order. A block goes in the first free slot (morning first)
    on its target day, or on the next days with space, wrapping around the week. It ignores
    how good each slot is, the daily limit and rest days.
    """
    cfg = problem.config
    n_slots = len(cfg.slots)
    taken = set()
    timetable = [None] * len(problem.blocks)

    def first_free(day):
        for offset in range(cfg.n_days):
            d = (day + offset) % cfg.n_days
            for s in range(n_slots):
                if (d, s) not in taken:
                    taken.add((d, s))
                    return (d, s)
        raise ValueError("no free slot left")  # cannot happen: capacity is checked up front

    for task in sorted(problem.tasks, key=lambda t: t.deadline_day):
        indexes = [i for i, block in enumerate(problem.blocks) if block.name == task.name]
        for k, i in enumerate(indexes):
            timetable[i] = first_free(k * (task.deadline_day + 1) // len(indexes))
    return timetable


def _ordinal(n):
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(n if n < 20 else n % 10, "th")
    return f"{n}{suffix}"


def explain_plan(problem, timetable, day_names=None):
    """Give every block a short, factual reason, sorted by day and slot.

    Each row has task, day, slot, minutes and reason. The reason names how good the slot is
    for this student, how far the block is from its deadline, and flags double-booked or
    overloaded days.
    """
    cfg = problem.config
    labels = list(day_names) if day_names is not None else [f"day {d}" for d in range(cfg.n_days)]
    if len(labels) != cfg.n_days:
        raise ValueError("day_names needs one label per day")
    ranking = sorted(cfg.slots, key=lambda s: -problem.effectiveness[s])
    occupied = Counter(timetable)
    blocks_per_day = Counter(day for day, _slot in timetable)

    rows = []
    for task, (day, slot) in zip(problem.blocks, timetable):
        slot_name = cfg.slots[slot]
        rank = ranking.index(slot_name) + 1
        quality = problem.effectiveness[slot_name]
        if rank == 1:
            parts = [f"{slot_name} is your best slot ({quality:.2f})"]
        else:
            text = f"{slot_name} is your {_ordinal(rank)} best slot ({quality:.2f})"
            better = ranking[:rank - 1]
            if all(occupied[(day, cfg.slots.index(s))] > 0 for s in better):
                text += "; the better slots that day are already booked"
            parts = [text]

        gap = task.deadline_day - day
        deadline = labels[task.deadline_day]
        if gap > 0:
            parts.append(f"{gap} day{'s' if gap != 1 else ''} before the {task.name} deadline ({deadline})")
        elif gap == 0:
            parts.append(f"on the {task.name} deadline day ({deadline})")
        else:
            parts.append(f"after the {task.name} deadline ({deadline}), so it earns no credit")

        if occupied[(day, slot)] > 1:
            parts.append("double-booked with another block")
        if blocks_per_day[day] * cfg.block_minutes > cfg.max_daily_minutes:
            parts.append("this day is over the daily study limit")

        reason = "; ".join(parts)
        rows.append({"task": task.name, "day": day, "slot": slot_name,
                     "minutes": cfg.block_minutes, "reason": reason[0].upper() + reason[1:] + "."})
    rows.sort(key=lambda r: (r["day"], cfg.slots.index(r["slot"])))
    return rows


@dataclass
class ReplanResult:
    """A replanned remainder of the week.

    `problem` and `result` describe the remaining days only: day 0 there is day `from_day`
    of the original plan. `overdue_tasks` lists tasks whose deadline had already passed.
    """

    problem: object
    result: object
    from_day: int
    overdue_tasks: list


def replan(problem, completed, from_day, iterations=6000, seed=None, **anneal_options):
    """Re-plan the rest of the week after missed sessions.

    `completed` maps task name to how many of its blocks were actually done (missing names
    count as 0). Planning restarts at the beginning of day `from_day`. Overdue tasks get
    deadline 0 in the new plan, which makes them the most urgent. Raises ValueError if
    nothing is left to plan or the remaining blocks cannot fit in the remaining slots.
    """
    cfg = problem.config
    if not 0 <= from_day < cfg.n_days:
        raise ValueError(f"from_day must be between 0 and {cfg.n_days - 1}")
    totals = Counter(block.name for block in problem.blocks)
    for name, done in completed.items():
        if name not in totals:
            raise ValueError(f"Unknown task in completed: {name!r}")
        if not 0 <= done <= totals[name]:
            raise ValueError(f"completed blocks for {name!r} must be between 0 and {totals[name]}")

    tasks, overdue = [], []
    for task in problem.tasks:
        left = totals[task.name] - completed.get(task.name, 0)
        if left == 0:
            continue
        if task.deadline_day < from_day:
            overdue.append(task.name)
        tasks.append(Task(task.name, left * cfg.block_minutes,
                          max(0, task.deadline_day - from_day), task.importance))
    if not tasks:
        raise ValueError("every block is already completed; nothing to replan")

    days_left = cfg.n_days - from_day
    new_config = replace(cfg, n_days=days_left, min_rest_days=min(cfg.min_rest_days, days_left - 1))
    new_problem = StudyPlanProblem(tasks, new_config)
    result = plan_week(new_problem, iterations=iterations, seed=seed, **anneal_options)
    return ReplanResult(new_problem, result, from_day, overdue)