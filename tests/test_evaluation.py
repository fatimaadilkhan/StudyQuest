import unittest
from collections import Counter

from ai_core.bandit import EpsilonGreedyBandit
from ai_core.evaluation import (SCENARIOS, EqualSplitPolicy, FakeStudent, RandomPolicy,
                                best_arm_rate_curve, cumulative_regret_curve, evaluate,
                                format_table, run_trial, sessions_to_reach, summarize)
from ai_core.thompson import ThompsonBandit

MEANS = {"morning": 0.45, "afternoon": 0.55, "evening": 0.70, "night": 0.40}


class TestFakeStudent(unittest.TestCase):
    def test_best_arm(self):
        self.assertEqual(FakeStudent(MEANS).best_arm, "evening")

    def test_rewards_are_clipped_to_unit_interval(self):
        student = FakeStudent({"a": 0.95, "b": 0.05}, noise_std=0.5, seed=0)
        rewards = [student.reward(arm) for arm in ("a", "b") for _ in range(300)]
        self.assertTrue(all(0.0 <= r <= 1.0 for r in rewards))

    def test_average_reward_is_close_to_true_mean(self):
        student = FakeStudent(MEANS, noise_std=0.15, seed=1)
        rewards = [student.reward("evening") for _ in range(4000)]
        self.assertAlmostEqual(sum(rewards) / len(rewards), 0.70, delta=0.02)

    def test_same_seed_gives_same_rewards(self):
        a, b = FakeStudent(MEANS, seed=5), FakeStudent(MEANS, seed=5)
        self.assertEqual([a.reward("night") for _ in range(10)],
                         [b.reward("night") for _ in range(10)])

    def test_invalid_input_raises(self):
        with self.assertRaises(ValueError):
            FakeStudent({})
        with self.assertRaises(ValueError):
            FakeStudent(MEANS, noise_std=-1)
        with self.assertRaises(ValueError):
            FakeStudent(MEANS).reward("midnight")


class TestBaselinePolicies(unittest.TestCase):
    def test_equal_split_cycles_evenly(self):
        policy = EqualSplitPolicy(["a", "b", "c"])
        picks = Counter(policy.select_arm() for _ in range(30))
        self.assertEqual(set(picks.values()), {10})

    def test_random_policy_uses_every_arm_and_ignores_feedback(self):
        policy = RandomPolicy(["a", "b", "c"], seed=0)
        for _ in range(50):
            policy.update("a", 1.0)
        picks = Counter(policy.select_arm() for _ in range(600))
        self.assertEqual(set(picks), {"a", "b", "c"})
        for count in picks.values():
            self.assertGreater(count, 140)


class TestRunTrial(unittest.TestCase):
    def make_student(self, seed):
        return FakeStudent(MEANS, 0.15, seed)

    def test_regret_is_zero_for_the_best_arm_and_positive_otherwise(self):
        regrets, hits = run_trial(lambda arms, s: EqualSplitPolicy(arms),
                                  self.make_student, 8, seed=0)
        self.assertEqual(len(regrets), 8)
        for regret, hit in zip(regrets, hits):
            self.assertGreaterEqual(regret, 0.0)
            self.assertEqual(regret == 0.0, hit == 1)

    def test_same_seed_is_reproducible(self):
        make = lambda arms, s: ThompsonBandit(arms)
        self.assertEqual(run_trial(make, self.make_student, 50, seed=3),
                         run_trial(make, self.make_student, 50, seed=3))


class TestEvaluate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = evaluate(SCENARIOS["clear best slot"], horizon=150, n_seeds=60)
        cls.rows = {r["policy"]: r for r in summarize(cls.results, early=20, late=40)}

    def test_all_policies_present_with_the_right_shape(self):
        self.assertEqual(set(self.results),
                         {"Thompson sampling", "Epsilon-greedy", "Random", "Equal split"})
        for result in self.results.values():
            self.assertEqual(len(result["regret"]), 60)
            self.assertEqual(len(result["regret"][0]), 150)

    def test_learning_policies_beat_the_baselines(self):
        for learner in ("Thompson sampling", "Epsilon-greedy"):
            for baseline in ("Random", "Equal split"):
                self.assertLess(self.rows[learner]["regret_mean"],
                                self.rows[baseline]["regret_mean"] / 2)

    def test_thompson_finds_the_best_slot(self):
        self.assertGreater(self.rows["Thompson sampling"]["late_best_rate"], 0.8)
        self.assertIsNotNone(self.rows["Thompson sampling"]["sessions_to_target"])

    def test_baselines_never_reach_the_target(self):
        self.assertIsNone(self.rows["Random"]["sessions_to_target"])
        self.assertIsNone(self.rows["Equal split"]["sessions_to_target"])

    def test_cumulative_regret_never_decreases(self):
        curve = cumulative_regret_curve(self.results["Thompson sampling"])
        self.assertTrue(all(b >= a for a, b in zip(curve, curve[1:])))

    def test_best_arm_rate_curve_is_a_proportion(self):
        curve = best_arm_rate_curve(self.results["Thompson sampling"])
        self.assertEqual(len(curve), 150)
        self.assertTrue(all(0.0 <= x <= 1.0 for x in curve))

    def test_evaluate_rejects_bad_sizes(self):
        with self.assertRaises(ValueError):
            evaluate(SCENARIOS["clear best slot"], horizon=0)
        with self.assertRaises(ValueError):
            evaluate(SCENARIOS["clear best slot"], n_seeds=0)

    def test_custom_policy_set(self):
        results = evaluate(SCENARIOS["clear best slot"],
                           {"EG": lambda arms, s: EpsilonGreedyBandit(arms)},
                           horizon=20, n_seeds=5)
        self.assertEqual(list(results), ["EG"])


class TestSummaryHelpers(unittest.TestCase):
    def test_sessions_to_reach(self):
        self.assertEqual(sessions_to_reach([0.1] * 5 + [0.9] * 5, window=3), 6)
        self.assertIsNone(sessions_to_reach([0.1] * 10, window=3))

    def test_format_table_has_header_divider_and_a_row_per_policy(self):
        rows = [{"policy": "A", "regret_mean": 1.0, "regret_std": 0.5,
                 "early_best_rate": 0.5, "late_best_rate": 0.9, "sessions_to_target": 12},
                {"policy": "B", "regret_mean": 9.0, "regret_std": 0.0,
                 "early_best_rate": 0.2, "late_best_rate": 0.2, "sessions_to_target": None}]
        lines = format_table(rows).split("\n")
        self.assertEqual(len(lines), 4)
        self.assertTrue(lines[1].startswith("| -"))
        self.assertIn("never", lines[3])
        self.assertIn("12", lines[2])


if __name__ == "__main__":
    unittest.main()