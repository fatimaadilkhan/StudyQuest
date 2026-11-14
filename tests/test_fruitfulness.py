import unittest

from ai_core.fruitfulness import fruitfulness_score, WEIGHTS


class TestFruitfulnessScore(unittest.TestCase):
    def test_weights_sum_to_one(self):
        self.assertAlmostEqual(sum(WEIGHTS.values()), 1.0)

    def test_perfect_session(self):
        self.assertAlmostEqual(fruitfulness_score(10, 10, 5, 0, 5), 1.0)

    def test_skipped_session(self):
        self.assertAlmostEqual(fruitfulness_score(0, 10, 1, 5, 1), 0.0)

    def test_distracted_session(self):
        # 0.40*1 + 0.30*0.25 + 0.20*0 + 0.10*0.5
        self.assertAlmostEqual(fruitfulness_score(10, 10, 2, 5, 3), 0.525)

    def test_good_session(self):
        # 0.40*0.8 + 0.30*0.75 + 0.20*0.8 + 0.10*0.75
        self.assertAlmostEqual(fruitfulness_score(8, 10, 4, 1, 4), 0.78)

    def test_half_done_average_session(self):
        # 0.40*0.5 + 0.30*0.5 + 0.20*0.6 + 0.10*0.5
        self.assertAlmostEqual(fruitfulness_score(5, 10, 3, 2, 3), 0.52)

    def test_tired_but_productive(self):
        # 0.40 + 0.30 + 0.20 + 0
        self.assertAlmostEqual(fruitfulness_score(10, 10, 5, 0, 1), 0.9)

    def test_overdone_is_capped(self):
        self.assertAlmostEqual(fruitfulness_score(15, 10, 5, 0, 5), 1.0)

    def test_always_between_0_and_1(self):
        for done in (0, 3, 10, 50):
            for focus in range(1, 6):
                for dist in (0, 2, 5, 20):
                    for energy in range(1, 6):
                        s = fruitfulness_score(done, 10, focus, dist, energy)
                        self.assertGreaterEqual(s, 0.0)
                        self.assertLessEqual(s, 1.0)

    def test_invalid_inputs_raise(self):
        with self.assertRaises(ValueError):
            fruitfulness_score(5, 0, 3, 0, 3)    # nothing planned
        with self.assertRaises(ValueError):
            fruitfulness_score(-1, 10, 3, 0, 3)  # negative units
        with self.assertRaises(ValueError):
            fruitfulness_score(5, 10, 6, 0, 3)   # focus off the scale
        with self.assertRaises(ValueError):
            fruitfulness_score(5, 10, 3, -1, 3)  # negative distractions


if __name__ == "__main__":
    unittest.main()