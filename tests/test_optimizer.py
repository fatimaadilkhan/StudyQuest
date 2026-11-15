import random
import unittest
from collections import Counter
from ai_core.bandit import EpsilonGreedyBandit
from ai_core.thompson import ThompsonBandit

from ai_core.optimizer import StudyOptimizer

ARMS = ["quiz", "flashcard", "summary"]


class TestStudyOptimizer(unittest.TestCase):
    def setUp(self):
        self.opt = StudyOptimizer(ARMS)



    # --- reward conversion ---
    def test_pass_gives_reward_1(self):
        self.assertEqual(self.opt.process_student_feedback("quiz", 85), 1)

    def test_fail_gives_reward_0(self):
        self.assertEqual(self.opt.process_student_feedback("quiz", 40), 0)

    def test_threshold_boundary(self):
        self.assertEqual(self.opt.process_student_feedback("quiz", 70), 1)
        self.assertEqual(self.opt.process_student_feedback("quiz", 69), 0)

    def test_custom_threshold(self):
        self.assertEqual(
            self.opt.process_student_feedback("quiz", 60, passing_threshold=50), 1
        )

    # --- bandit state updates ---
    def test_counts_update(self):
        self.opt.process_student_feedback("quiz", 90)
        self.opt.process_student_feedback("quiz", 20)
        self.assertEqual(self.opt.bandit.counts["quiz"], 2)
        self.assertEqual(self.opt.bandit.counts["flashcard"], 0)

    def test_value_is_running_average(self):
        self.opt.process_student_feedback("quiz", 90)  # reward 1
        self.opt.process_student_feedback("quiz", 20)  # reward 0
        self.assertAlmostEqual(self.opt.bandit.values["quiz"], 0.5)

    def test_unknown_module_raises_and_changes_nothing(self):
        with self.assertRaises(ValueError):
            self.opt.process_student_feedback("nonexistent", 80)
        self.assertEqual(sum(self.opt.bandit.counts.values()), 0)

    # --- recommendations ---
    def test_recommendation_is_valid_arm(self):
        self.assertIn(self.opt.get_next_recommendation(), ARMS)

    def test_every_arm_tried_before_exploiting_epsilon_greedy(self):
        opt = StudyOptimizer(ARMS, EpsilonGreedyBandit(ARMS))
        seen = []
        for _ in range(len(ARMS)):
            arm = opt.get_next_recommendation()
            seen.append(arm)
            opt.process_student_feedback(arm, 50)
        self.assertEqual(set(seen), set(ARMS))
    def test_best_arm_dominates(self):
        random.seed(0)
        pass_rate = {"quiz": 0.9, "flashcard": 0.4, "summary": 0.2}
        picks = Counter()
        for _ in range(300):
            arm = self.opt.get_next_recommendation()
            picks[arm] += 1
            score = 80 if random.random() < pass_rate[arm] else 40
            self.opt.process_student_feedback(arm, score)
        self.assertEqual(picks.most_common(1)[0][0], "quiz")

    def test_ties_are_broken_randomly(self):
        random.seed(1)
        picks = set()
        for _ in range(100):
            opt = StudyOptimizer(ARMS)
            for a in ARMS:
                opt.process_student_feedback(a, 80)  # all values equal 1.0
            opt.bandit.epsilon = 0  # force the exploitation path
            picks.add(opt.get_next_recommendation())
        self.assertGreater(len(picks), 1)

    def test_checkin_updates_bandit_with_continuous_reward(self):
        reward = self.opt.process_checkin("quiz", 8, 10, 4, 1, 4)
        self.assertAlmostEqual(reward, 0.78)
        self.assertEqual(self.opt.bandit.counts["quiz"], 1)
        self.assertAlmostEqual(self.opt.bandit.values["quiz"], 0.78)
    def test_default_bandit_is_thompson(self):
        self.assertIsInstance(self.opt.bandit, ThompsonBandit)


if __name__ == "__main__":
    unittest.main()