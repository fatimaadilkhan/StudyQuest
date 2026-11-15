import random
import unittest
from collections import Counter

from ai_core.optimizer import StudyOptimizer

ARMS = ["quiz", "flashcard", "summary"]


class TestSessionOptimizer(unittest.TestCase):
    def setUp(self):
        self.opt = StudyOptimizer(ARMS)

    def test_recommend_session_returns_valid_choices(self):
        for _ in range(30):
            s = self.opt.recommend_session()
            self.assertEqual(set(s), {"method", "time_slot", "length"})
            self.assertIn(s["method"], ARMS)
            self.assertIn(s["time_slot"], self.opt.slot_bandit.arms)
            self.assertIn(s["length"], self.opt.length_bandit.arms)

    def test_checkin_updates_all_three_bandits_with_same_reward(self):
        reward = self.opt.process_checkin(
            "quiz", 8, 10, 4, 1, 4, time_slot="evening", session_length=45)
        self.assertAlmostEqual(reward, 0.78)
        self.assertEqual(self.opt.bandit.counts["quiz"], 1)
        self.assertEqual(self.opt.slot_bandit.counts["evening"], 1)
        self.assertEqual(self.opt.length_bandit.counts[45], 1)
        self.assertAlmostEqual(self.opt.slot_bandit.values["evening"], 0.78)
        self.assertAlmostEqual(self.opt.length_bandit.values[45], 0.78)

    def test_checkin_without_context_updates_method_only(self):
        self.opt.process_checkin("quiz", 8, 10, 4, 1, 4)
        self.assertEqual(self.opt.bandit.counts["quiz"], 1)
        self.assertEqual(sum(self.opt.slot_bandit.counts.values()), 0)
        self.assertEqual(sum(self.opt.length_bandit.counts.values()), 0)

    def test_unknown_time_slot_raises_and_changes_nothing(self):
        with self.assertRaises(ValueError):
            self.opt.process_checkin(
                "quiz", 8, 10, 4, 1, 4, time_slot="midnight", session_length=45)
        self._assert_nothing_updated()

    def test_unknown_session_length_raises_and_changes_nothing(self):
        with self.assertRaises(ValueError):
            self.opt.process_checkin(
                "quiz", 8, 10, 4, 1, 4, time_slot="evening", session_length=999)
        self._assert_nothing_updated()

    def test_custom_slots_and_lengths(self):
        opt = StudyOptimizer(ARMS, time_slots=["am", "pm"], session_lengths=[30, 90])
        for _ in range(20):
            s = opt.recommend_session()
            self.assertIn(s["time_slot"], ["am", "pm"])
            self.assertIn(s["length"], [30, 90])

    def test_learns_best_method_slot_and_length(self):
        random.seed(0)
        method_bonus = {"quiz": 0.4, "flashcard": 0.2, "summary": 0.0}
        slot_bonus = {"morning": 0.1, "afternoon": 0.0, "evening": 0.3, "night": 0.0}
        length_bonus = {25: 0.0, 45: 0.2, 60: 0.0}
        picks = {"method": Counter(), "time_slot": Counter(), "length": Counter()}

        for _ in range(800):
            s = self.opt.recommend_session()
            for key in picks:
                picks[key][s[key]] += 1
            completion = (0.1 + method_bonus[s["method"]] + slot_bonus[s["time_slot"]]
                          + length_bonus[s["length"]] + random.gauss(0, 0.05))
            completion = min(1.0, max(0.0, completion))
            self.opt.process_checkin(
                s["method"], round(10 * completion), 10, 5, 0, 5,
                time_slot=s["time_slot"], session_length=s["length"])

        self.assertEqual(picks["method"].most_common(1)[0][0], "quiz")
        self.assertEqual(picks["time_slot"].most_common(1)[0][0], "evening")
        self.assertEqual(picks["length"].most_common(1)[0][0], 45)

    def _assert_nothing_updated(self):
        self.assertEqual(sum(self.opt.bandit.counts.values()), 0)
        self.assertEqual(sum(self.opt.slot_bandit.counts.values()), 0)
        self.assertEqual(sum(self.opt.length_bandit.counts.values()), 0)


if __name__ == "__main__":
    unittest.main()