import os
import tempfile
import unittest

from ai_core.bandit import EpsilonGreedyBandit
from ai_core.optimizer import StudyOptimizer
from ai_core.storage import load_or_create, load_state, save_state

ARMS = ["quiz", "flashcard", "summary"]


def trained_optimizer():
    opt = StudyOptimizer(ARMS)
    sessions = [
        ("quiz", "morning", 25, 8, 4, 1, 4),
        ("flashcard", "evening", 45, 5, 3, 3, 2),
        ("quiz", "evening", 45, 10, 5, 0, 5),
        ("summary", "night", 60, 2, 2, 5, 1),
        ("quiz", "morning", 25, 9, 5, 0, 4),
    ]
    for method, slot, length, done, focus, dist, energy in sessions:
        opt.process_checkin(method, done, 10, focus, dist, energy,
                            time_slot=slot, session_length=length)
    opt.record_task_time(60, 84, "reading")
    opt.record_task_time(30, 25, "coding")
    opt.record_task_time(45, 60)
    return opt


class TestStorage(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = os.path.join(self._tmp.name, "state.db")

    def test_bandit_state_round_trips(self):
        original = trained_optimizer()
        save_state(original, self.path)
        loaded = load_state(self.path)
        for attr in ("bandit", "slot_bandit", "length_bandit"):
            a, b = getattr(original, attr), getattr(loaded, attr)
            self.assertEqual(a.arms, b.arms)
            self.assertEqual(a.counts, b.counts)
            self.assertEqual(a.wins, b.wins)
            for arm in a.arms:
                self.assertAlmostEqual(a.sums[arm], b.sums[arm])
                self.assertAlmostEqual(a.values[arm], b.values[arm])
                self.assertAlmostEqual(a.posterior(arm)[0], b.posterior(arm)[0])
                self.assertAlmostEqual(a.posterior(arm)[1], b.posterior(arm)[1])

    def test_calibration_state_round_trips(self):
        original = trained_optimizer()
        save_state(original, self.path)
        loaded = load_state(self.path)
        for category in (None, "reading", "coding", "unseen"):
            self.assertAlmostEqual(original.calibrator.multiplier(category),
                                   loaded.calibrator.multiplier(category))
            self.assertAlmostEqual(original.calibrator.posterior(category)[1],
                                   loaded.calibrator.posterior(category)[1])
        self.assertAlmostEqual(original.calibrated_estimate(60, "reading"),
                               loaded.calibrated_estimate(60, "reading"))

    def test_arm_types_and_order_preserved(self):
        save_state(trained_optimizer(), self.path)
        loaded = load_state(self.path)
        self.assertEqual(loaded.length_bandit.arms, [25, 45, 60])
        self.assertTrue(all(isinstance(a, int) for a in loaded.length_bandit.arms))
        self.assertEqual(loaded.slot_bandit.arms,
                         ["morning", "afternoon", "evening", "night"])
        self.assertEqual(loaded.bandit.arms, ARMS)

    def test_custom_arms_preserved(self):
        opt = StudyOptimizer(["a", "b"], time_slots=["am", "pm"], session_lengths=[30, 90])
        save_state(opt, self.path)
        loaded = load_state(self.path)
        self.assertEqual(loaded.bandit.arms, ["a", "b"])
        self.assertEqual(loaded.slot_bandit.arms, ["am", "pm"])
        self.assertEqual(loaded.length_bandit.arms, [30, 90])

    def test_restored_optimizer_keeps_learning(self):
        save_state(trained_optimizer(), self.path)
        loaded = load_state(self.path)
        before = loaded.bandit.counts["quiz"]
        loaded.process_checkin("quiz", 8, 10, 4, 1, 4,
                               time_slot="morning", session_length=25)
        self.assertEqual(loaded.bandit.counts["quiz"], before + 1)
        loaded.record_task_time(60, 84, "reading")
        self.assertGreater(loaded.calibrated_estimate(60, "reading"), 60)

    def test_saving_again_replaces_old_state(self):
        opt = trained_optimizer()
        save_state(opt, self.path)
        opt.process_checkin("quiz", 8, 10, 4, 1, 4,
                            time_slot="morning", session_length=25)
        save_state(opt, self.path)
        loaded = load_state(self.path)
        self.assertEqual(loaded.bandit.counts, opt.bandit.counts)
        self.assertEqual(sum(loaded.bandit.counts.values()), 6)

    def test_empty_history_round_trips_and_cold_starts(self):
        save_state(StudyOptimizer(ARMS), self.path)
        loaded = load_state(self.path)
        self.assertEqual(sum(loaded.bandit.counts.values()), 0)
        s = loaded.recommend_session()
        self.assertEqual((s["method"], s["time_slot"], s["length"]),
                         ("quiz", "morning", 25))
        self.assertAlmostEqual(loaded.calibrated_estimate(40), 40)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_state(os.path.join(self._tmp.name, "nope.db"))

    def test_non_state_file_raises_value_error(self):
        with open(self.path, "w") as f:
            f.write("this is not a database")
        with self.assertRaises(ValueError):
            load_state(self.path)

    def test_epsilon_greedy_method_bandit_rejected_and_nothing_written(self):
        opt = StudyOptimizer(ARMS, bandit=EpsilonGreedyBandit(ARMS))
        with self.assertRaises(TypeError):
            save_state(opt, self.path)
        self.assertFalse(os.path.exists(self.path))

    def test_load_or_create(self):
        fresh = load_or_create(self.path, ARMS)
        self.assertEqual(sum(fresh.bandit.counts.values()), 0)
        self.assertFalse(os.path.exists(self.path))  # creating doesn't save
        save_state(trained_optimizer(), self.path)
        loaded = load_or_create(self.path, ARMS)
        self.assertEqual(loaded.bandit.counts["quiz"], 3)


if __name__ == "__main__":
    unittest.main()