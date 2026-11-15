from ai_core.thompson import ThompsonBandit
from ai_core.fruitfulness import fruitfulness_score


class StudyOptimizer:
    def __init__(self, study_modules, bandit=None):
        """
        :param study_modules: List of available module types (e.g., ['quiz', 'flashcard', 'summary'])
        :param bandit: Optional bandit instance; defaults to Thompson sampling
        """
        self.bandit = bandit if bandit is not None else ThompsonBandit(arms=study_modules)

    def get_next_recommendation(self):
        """Asks the bandit algorithm to choose the next best study module."""
        return self.bandit.select_arm()

    def process_student_feedback(self, chosen_module, score, passing_threshold=70):
        """Legacy Day 1 path: converts a quiz score to a 0/1 reward."""
        reward = 1 if score >= passing_threshold else 0
        self.bandit.update(chosen_module, reward)
        return reward

    def process_checkin(self, chosen_module, units_done, units_planned,
                        focus, distractions, energy):
        """Scores a session from check-in data and feeds it to the bandit as the reward."""
        reward = fruitfulness_score(units_done, units_planned, focus, distractions, energy)
        self.bandit.update(chosen_module, reward)
        return reward