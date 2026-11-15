import math
import random
import unittest

from ai_core.calibration import TimeCalibrator
from ai_core.optimizer import StudyOptimizer


class TestTimeCalibrator(unittest.TestCase):
    def setUp(self):
        self.cal = TimeCalibrator(prior_std=0.3, obs_std=0.4)

    def test_no_data_leaves_estimate_unchanged(self):
        self.assertAlmostEqual(self.cal.calibrate(30), 30)
        self.assertAlmostEqual(self.cal.multiplier(), 1.0)

    def test_posterior_matches_formula_for_one_task(self):
        self.cal.record(30, 60)
        prior_prec = 1 / 0.3 ** 2
        obs_prec = 1 / 0.4 ** 2
        precision = prior_prec + obs_prec
        expected_mean = obs_prec * math.log(2) / precision
        mean, std = self.cal.posterior()
        self.assertAlmostEqual(mean, expected_mean)
        self.assertAlmostEqual(std, precision ** -0.5)

    def test_underestimator_gets_multiplier_above_one(self):
        random.seed(0)
        for _ in range(50):
            est = random.choice([20, 30, 45, 60])
            self.cal.record(est, est * 2 * math.exp(random.gauss(0, 0.2)))
        self.assertGreater(self.cal.multiplier(), 1.8)
        self.assertLess(self.cal.multiplier(), 2.2)

    def test_overestimator_gets_multiplier_below_one(self):
        random.seed(1)
        for _ in range(50):
            est = random.choice([20, 30, 45, 60])
            self.cal.record(est, est * 0.5 * math.exp(random.gauss(0, 0.2)))
        self.assertGreater(self.cal.multiplier(), 0.45)
        self.assertLess(self.cal.multiplier(), 0.56)

    def test_double_and_half_cancel_out(self):
        self.cal.record(30, 60)
        self.cal.record(60, 30)
        self.assertAlmostEqual(self.cal.multiplier(), 1.0)

    def test_uncertainty_shrinks_with_data(self):
        _, std_before = self.cal.posterior()
        for _ in range(10):
            self.cal.record(30, 45)
        _, std_after = self.cal.posterior()
        self.assertLess(std_after, std_before)

    def test_calibrate_scales_linearly_with_estimate(self):
        self.cal.record(30, 60)
        self.assertAlmostEqual(self.cal.calibrate(60), 2 * self.cal.calibrate(30))

    def test_categories_learn_separately(self):
        for _ in range(10):
            self.cal.record(30, 60, category="reading")
            self.cal.record(30, 15, category="coding")
        self.assertGreater(self.cal.calibrate(30, "reading"), 30)
        self.assertLess(self.cal.calibrate(30, "coding"), 30)

    def test_unseen_category_falls_back_to_overall(self):
        for _ in range(10):
            self.cal.record(30, 60, category="reading")
        self.assertAlmostEqual(self.cal.calibrate(30, "writing"), self.cal.calibrate(30))

    def test_invalid_input_raises_and_records_nothing(self):
        with self.assertRaises(ValueError):
            self.cal.record(0, 30)
        with self.assertRaises(ValueError):
            self.cal.record(30, -5)
        with self.assertRaises(ValueError):
            self.cal.calibrate(0)
        self.assertAlmostEqual(self.cal.multiplier(), 1.0)

    def test_invalid_parameters_raise(self):
        with self.assertRaises(ValueError):
            TimeCalibrator(prior_std=0)
        with self.assertRaises(ValueError):
            TimeCalibrator(obs_std=-1)


class TestOptimizerCalibration(unittest.TestCase):
    def setUp(self):
        self.opt = StudyOptimizer(["quiz", "flashcard", "summary"])

    def test_untrained_estimate_is_unchanged(self):
        self.assertAlmostEqual(self.opt.calibrated_estimate(40), 40)

    def test_recorded_times_change_the_estimate(self):
        for _ in range(10):
            self.opt.record_task_time(30, 60)
        self.assertGreater(self.opt.calibrated_estimate(30), 40)


if __name__ == "__main__":
    unittest.main()