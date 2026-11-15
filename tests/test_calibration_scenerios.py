import unittest

from ai_core.calibration import TimeCalibrator


def trained(n_sessions, estimated=60, actual=84, category="reading"):
    cal = TimeCalibrator()
    for _ in range(n_sessions):
        cal.record(estimated, actual, category)
    return cal


class TestCalibrationScenarios(unittest.TestCase):
    def test_reading_estimate_approaches_true_time(self):
        # True time for a 60-minute plan is 84 minutes (40% longer).
        errors = [abs(trained(n).calibrate(60, "reading") - 84) for n in (1, 3, 20)]
        self.assertGreater(errors[0], errors[1])
        self.assertGreater(errors[1], errors[2])
        self.assertLess(errors[2], 2)

    def test_one_session_shrinks_toward_no_correction(self):
        m = trained(1).multiplier("reading")
        self.assertGreater(m, 1.0)
        self.assertLess(m, 1.4)  # doesn't jump all the way to the raw ratio

    def test_confidence_grows_with_sessions(self):
        stds = [trained(n).posterior("reading")[1] for n in (1, 3, 20)]
        self.assertGreater(stds[0], stds[1])
        self.assertGreater(stds[1], stds[2])

    def test_sparse_category_shrinks_toward_overall(self):
        cal = TimeCalibrator()
        for _ in range(20):
            cal.record(30, 30, "coding")      # accurate on coding
        cal.record(30, 60, "reading")         # one reading task took twice as long
        reading = cal.multiplier("reading")
        self.assertLess(reading, 1.4)         # well below the raw ratio of 2.0
        self.assertGreater(reading, cal.multiplier())


if __name__ == "__main__":
    unittest.main()