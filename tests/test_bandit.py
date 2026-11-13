import random
import unittest

from ai_core.bandit import EpsilonGreedyBandit


class TestEpsilonGreedyBandit(unittest.TestCase):
    def setUp(self):
        random.seed(42)  # makes the tests repeatable
        self.arms = ['quiz', 'flashcard', 'summary']

    def test_tries_every_arm_first(self):
        b = EpsilonGreedyBandit(self.arms)
        picked = set()
        for _ in range(3):
            arm = b.select_arm()
            picked.add(arm)
            b.update(arm, 0)
        self.assertEqual(picked, set(self.arms))

    def test_unknown_arm_raises(self):
        b = EpsilonGreedyBandit(self.arms)
        with self.assertRaises(ValueError):
            b.update('video', 1)

    def test_average_is_correct(self):
        b = EpsilonGreedyBandit(self.arms)
        for r in [1, 0, 1, 1]:
            b.update('quiz', r)
        self.assertAlmostEqual(b.values['quiz'], 0.75)
        self.assertEqual(b.counts['quiz'], 4)

    def test_finds_best_arm(self):
        b = EpsilonGreedyBandit(self.arms, epsilon=0.1)
        rates = {'quiz': 0.3, 'flashcard': 0.7, 'summary': 0.5}
        for _ in range(500):
            arm = b.select_arm()
            b.update(arm, 1 if random.random() < rates[arm] else 0)
        self.assertEqual(max(b.counts, key=b.counts.get), 'flashcard')


if __name__ == '__main__':
    unittest.main()