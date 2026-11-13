import unittest
from ai_core.optimizer import StudyOptimizer


class TestStudyOptimizer(unittest.TestCase):
    def setUp(self):
        self.opt = StudyOptimizer(['quiz', 'flashcard', 'summary'])

    def test_pass_gives_reward_1(self):
        self.assertEqual(self.opt.process_student_feedback('quiz', 70), 1)

    def test_fail_gives_reward_0(self):
        self.assertEqual(self.opt.process_student_feedback('quiz', 69), 0)

    def test_feedback_updates_bandit(self):
        self.opt.process_student_feedback('quiz', 90)
        self.assertEqual(self.opt.bandit.counts['quiz'], 1)

    def test_recommendation_is_valid_arm(self):
        self.assertIn(self.opt.get_next_recommendation(), ['quiz', 'flashcard', 'summary'])


if __name__ == '__main__':
    unittest.main()