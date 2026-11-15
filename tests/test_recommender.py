import random
import unittest

from ai_core.bandit import EpsilonGreedyBandit
from ai_core.optimizer import StudyOptimizer
from ai_core.thompson import ThompsonBandit

ARMS = ["quiz", "flashcard", "summary"]


class TestWins(unittest.TestCase):
    def test_wins_counted_at_threshold(self):
        b = ThompsonBandit(ARMS)  # default threshold 0.6
        b.update("quiz", 0.6)
        b.update("quiz", 0.59)
        self.assertEqual(b.wins["quiz"], 1)
        self.assertEqual(b.counts["quiz"], 2)

    def test_custom_win_threshold(self):
        b = ThompsonBandit(ARMS, win_threshold=0.8)
        b.update("quiz", 0.7)
        self.assertEqual(b.wins["quiz"], 0)
        b.update("quiz", 0.8)
        self.assertEqual(b.wins["quiz"], 1)


class TestRecommender(unittest.TestCase):
    def test_cold_start_uses_defaults(self):
        s = StudyOptimizer(ARMS).recommend_session()
        self.assertEqual(s["method"], "quiz")
        self.assertEqual(s["time_slot"], "morning")
        self.assertEqual(s["length"], 25)

    def test_cold_start_reason_mentions_default(self):
        s = StudyOptimizer(ARMS).recommend_session()
        self.assertIn("starting default", s["reason"])

    def test_custom_lists_default_to_first_option(self):
        opt = StudyOptimizer(ARMS, time_slots=["am", "pm"], session_lengths=[30, 90])
        s = opt.recommend_session()
        self.assertEqual(s["time_slot"], "am")
        self.assertEqual(s["length"], 30)

    def test_reason_covers_all_three_decisions(self):
        reason = StudyOptimizer(ARMS).recommend_session()["reason"]
        for label in ("Method", "Time slot", "Length"):
            self.assertIn(label, reason)

    def test_reason_reports_wins_out_of_sessions(self):
        random.seed(0)
        opt = StudyOptimizer(ARMS)
        for _ in range(15):
            s = opt.recommend_session()
            opt.process_checkin(
                s["method"], random.randint(0, 10), 10, random.randint(1, 5),
                random.randint(0, 5), random.randint(1, 5),
                time_slot=s["time_slot"], session_length=s["length"])
        s = opt.recommend_session()
        slot = s["time_slot"]
        n = opt.slot_bandit.counts[slot]
        w = opt.slot_bandit.wins[slot]
        expected = f"worked well in {w} of {n} sessions" if n else "not tried yet"
        self.assertIn(f"Time slot {slot}: {expected}", s["reason"])

    def test_reason_example_six_of_eight(self):
        random.seed(0)
        opt = StudyOptimizer(ARMS)
        for _ in range(30):
            for slot in ("morning", "afternoon", "night"):
                opt.slot_bandit.update(slot, 0.1)
        for reward in [0.9] * 6 + [0.1] * 2:
            opt.slot_bandit.update("evening", reward)
        s = opt.recommend_session()
        self.assertEqual(s["time_slot"], "evening")
        self.assertIn("6 of 8 sessions", s["reason"])

    def test_works_with_epsilon_greedy_method_bandit(self):
        opt = StudyOptimizer(ARMS, bandit=EpsilonGreedyBandit(ARMS))
        opt.process_student_feedback("quiz", 90)
        s = opt.recommend_session()
        self.assertIn("Method", s["reason"])


if __name__ == "__main__":
    unittest.main()