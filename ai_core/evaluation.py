"""Evaluation harness: compares bandit policies on a simulated student.

A policy is anything with select_arm() and update(arm, reward). A student is
anything with arms, true_means, best_arm and reward(arm); FakeStudent is the
built-in one, and a wrapper around the team simulator only needs those four.

Regret is measured against the best option's true mean, so a policy that always
picks the best option has zero regret. The harness seeds Python's global random
generator before each trial so results are reproducible.
"""
import random
import statistics
from itertools import accumulate

from ai_core.bandit import EpsilonGreedyBandit
from ai_core.thompson import ThompsonBandit


class FakeStudent:
    """A simulated student whose sessions have a known average fruitfulness per option.

    true_means are the averages before rewards are clipped into [0, 1].
    """

    def __init__(self, true_means, noise_std=0.15, seed=None):
        if not true_means:
            raise ValueError("true_means must not be empty")
        if noise_std < 0:
            raise ValueError("noise_std cannot be negative")
        self.true_means = dict(true_means)
        self.arms = list(self.true_means)
        self.noise_std = noise_std
        self._rng = random.Random(seed)

    @property
    def best_arm(self):
        """The option with the highest true mean."""
        return max(self.true_means, key=lambda name: self.true_means[name])

    def reward(self, arm):
        """Return one noisy fruitfulness score in [0, 1] for a session using this option."""
        if arm not in self.true_means:
            raise ValueError(f"Unknown arm: {arm!r}. Valid arms: {self.arms}")
        value = self._rng.gauss(self.true_means[arm], self.noise_std)
        return min(1.0, max(0.0, value))


class RandomPolicy:
    """Baseline: picks an option uniformly at random every time."""

    def __init__(self, arms, seed=None):
        self.arms = list(arms)
        self._rng = random.Random(seed)

    def select_arm(self):
        """Pick any option with equal probability."""
        return self._rng.choice(self.arms)

    def update(self, arm, reward):
        """Ignore the feedback; this policy never learns."""


class EqualSplitPolicy:
    """Baseline: cycles through the options in order, giving each an equal share of sessions."""

    def __init__(self, arms):
        self.arms = list(arms)
        self._next = 0

    def select_arm(self):
        """Return the next option in the cycle."""
        arm = self.arms[self._next % len(self.arms)]
        self._next += 1
        return arm

    def update(self, arm, reward):
        """Ignore the feedback; this policy never learns."""


# name -> function(arms, seed) that builds a fresh policy
POLICY_FACTORIES = {
    "Thompson sampling": lambda arms, seed: ThompsonBandit(arms),
    "Epsilon-greedy": lambda arms, seed: EpsilonGreedyBandit(arms),
    "Random": lambda arms, seed: RandomPolicy(arms, seed),
    "Equal split": lambda arms, seed: EqualSplitPolicy(arms),
}


def _scenario(means, noise_std):
    return lambda seed: FakeStudent(means, noise_std, seed)


# name -> function(seed) that builds a fresh simulated student (four time slots)
SCENARIOS = {
    "clear best slot": _scenario(
        {"morning": 0.45, "afternoon": 0.55, "evening": 0.70, "night": 0.40}, 0.15),
    "close slots": _scenario(
        {"morning": 0.60, "afternoon": 0.65, "evening": 0.70, "night": 0.55}, 0.15),
    "noisy sessions": _scenario(
        {"morning": 0.45, "afternoon": 0.55, "evening": 0.70, "night": 0.40}, 0.30),
}


def run_trial(make_policy, make_student, horizon, seed):
    """Run one simulated student for `horizon` sessions.

    Returns (regrets, hits): per-session regret and 1 or 0 for whether the best option was picked.
    """
    student = make_student(seed)
    random.seed(seed + 1)
    policy = make_policy(student.arms, seed + 2)
    best_mean = max(student.true_means.values())
    best_arm = student.best_arm
    regrets, hits = [], []
    for _ in range(horizon):
        arm = policy.select_arm()
        policy.update(arm, student.reward(arm))
        regrets.append(best_mean - student.true_means[arm])
        hits.append(1 if arm == best_arm else 0)
    return regrets, hits


def evaluate(make_student, policies=None, horizon=200, n_seeds=500):
    """Run every policy on `n_seeds` different simulated students.

    Returns {policy name: {"regret": [per-seed lists], "hits": [per-seed lists]}}.
    """
    if horizon < 1 or n_seeds < 1:
        raise ValueError("horizon and n_seeds must be at least 1")
    policies = policies if policies is not None else POLICY_FACTORIES
    results = {}
    for name, make_policy in policies.items():
        regret_runs, hit_runs = [], []
        for seed in range(n_seeds):
            regrets, hits = run_trial(make_policy, make_student, horizon, seed)
            regret_runs.append(regrets)
            hit_runs.append(hits)
        results[name] = {"regret": regret_runs, "hits": hit_runs}
    return results


def mean_curve(runs):
    """Average several equal-length runs session by session."""
    return [sum(column) / len(column) for column in zip(*runs)]


def cumulative_regret_curve(result):
    """Mean cumulative regret after each session, averaged over seeds."""
    return mean_curve([list(accumulate(run)) for run in result["regret"]])


def best_arm_rate_curve(result, window=10):
    """Share of seeds picking the best option, smoothed over a trailing window of sessions."""
    raw = mean_curve(result["hits"])
    smoothed = []
    for i in range(len(raw)):
        chunk = raw[max(0, i - window + 1): i + 1]
        smoothed.append(sum(chunk) / len(chunk))
    return smoothed


def sessions_to_reach(curve, target=0.8, window=10):
    """First session at which a smoothed best-option rate reaches `target`, or None if never.

    Only checked once a full window of sessions exists.
    """
    for i in range(window - 1, len(curve)):
        if curve[i] >= target:
            return i + 1
    return None


def summarize(results, early=30, late=50, target=0.8):
    """Turn results into one row per policy.

    Each row has: policy, regret_mean, regret_std (final cumulative regret across seeds),
    early_best_rate (first `early` sessions), late_best_rate (last `late` sessions)
    and sessions_to_target (None if the policy never gets there).
    """
    rows = []
    for name, result in results.items():
        finals = [sum(run) for run in result["regret"]]
        hits = result["hits"]
        rows.append({
            "policy": name,
            "regret_mean": statistics.fmean(finals),
            "regret_std": statistics.stdev(finals) if len(finals) > 1 else 0.0,
            "early_best_rate": statistics.fmean(sum(h[:early]) / len(h[:early]) for h in hits),
            "late_best_rate": statistics.fmean(sum(h[-late:]) / len(h[-late:]) for h in hits),
            "sessions_to_target": sessions_to_reach(best_arm_rate_curve(result), target),
        })
    return rows


def format_table(rows, early=30, late=50, target=0.8):
    """Format summary rows as an aligned Markdown table."""
    header = ["Policy", "Cumulative regret", f"Best-option rate (first {early})",
              f"Best-option rate (last {late})", f"Sessions to {round(target * 100)}%"]
    body = []
    for r in rows:
        reached = r["sessions_to_target"]
        body.append([r["policy"],
                     f"{r['regret_mean']:.1f} ± {r['regret_std']:.1f}",
                     f"{r['early_best_rate'] * 100:.0f}%",
                     f"{r['late_best_rate'] * 100:.0f}%",
                     "never" if reached is None else str(reached)])
    widths = [max(len(row[c]) for row in [header] + body) for c in range(len(header))]

    def line(cells):
        return "| " + " | ".join(cell.ljust(widths[c]) for c, cell in enumerate(cells)) + " |"

    divider = "| " + " | ".join("-" * w for w in widths) + " |"
    return "\n".join([line(header), divider] + [line(row) for row in body])