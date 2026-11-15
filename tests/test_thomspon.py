import random
import unittest
from collections import Counter

from ai_core.thompson import ThompsonBandit

ARMS = ["quiz", "flashcard", "summary"]


class TestThompsonBandit(unittest.TestCase):
    def setUp(self):
        self.bandit = ThompsonBandit(ARMS, prior_mean=0.5, prior_std=0.5, obs_std=0.3)

    def test_untried_arm_has_prior_posterior(self):
        mean, std = self.bandit.posterior("quiz")
        self.assertAlmostEqual(mean, 0.5)
        self.assertAlmostEqual(std, 0.5)

    def test_update_tracks_counts_and_sample_mean(self):
        self.bandit.update("quiz", 0.8)
        self.bandit.update("quiz", 0.4)
        self.assertEqual(self.bandit.counts["quiz"], 2)
        self.assertAlmostEqual(self.bandit.values["quiz"], 0.6)
        self.assertEqual(self.bandit.counts["flashcard"], 0)

    def test_posterior_matches_normal_normal_formula(self):
        self.bandit.update("quiz", 0.9)
        prior_prec = 1 / 0.5 ** 2
        obs_prec = 1 / 0.3 ** 2
        precision = prior_prec + obs_prec
        expected_mean = (prior_prec * 0.5 + obs_prec * 0.9) / precision
        mean, std = self.bandit.posterior("quiz")
        self.assertAlmostEqual(mean, expected_mean)
        self.assertAlmostEqual(std, precision ** -0.5)

    def test_uncertainty_shrinks_and_mean_moves_to_data(self):
        _, std_before = self.bandit.posterior("quiz")
        for _ in range(20):
            self.bandit.update("quiz", 0.9)
        mean, std_after = self.bandit.posterior("quiz")
        self.assertLess(std_after, std_before)
        self.assertGreater(mean, 0.85)

    def test_unknown_arm_raises_and_changes_nothing(self):
        with self.assertRaises(ValueError):
            self.bandit.update("nonexistent", 0.5)
        self.assertEqual(sum(self.bandit.counts.values()), 0)

    def test_select_arm_returns_valid_arm(self):
        for _ in range(50):
            self.assertIn(self.bandit.select_arm(), ARMS)

    def test_uncertain_arms_still_get_explored(self):
        random.seed(0)
        for _ in range(20):
            self.bandit.update("quiz", 0.6)
        picks = Counter(self.bandit.select_arm() for _ in range(1000))
        self.assertGreater(picks["flashcard"], 100)
        self.assertGreater(picks["summary"], 100)

    def test_converges_to_best_arm(self):
        random.seed(0)
        true_mean = {"quiz": 0.8, "flashcard": 0.5, "summary": 0.3}
        picks = Counter()
        for _ in range(500):
            arm = self.bandit.select_arm()
            picks[arm] += 1
            reward = min(1.0, max(0.0, random.gauss(true_mean[arm], 0.15)))
            self.bandit.update(arm, reward)
        self.assertEqual(picks.most_common(1)[0][0], "quiz")
        self.assertGreater(picks["quiz"], 300)

    def test_invalid_parameters_raise(self):
        with self.assertRaises(ValueError):
            ThompsonBandit(ARMS, prior_std=0)
        with self.assertRaises(ValueError):
            ThompsonBandit(ARMS, obs_std=-1)


if __name__ == "__main__":
    unittest.main()