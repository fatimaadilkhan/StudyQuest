from ai_core.bandit import EpsilonGreedyBandit


class StudyOptimizer:
    def __init__(self, study_modules):
        """
        :param study_modules: List of available module types (e.g., ['quiz', 'flashcard', 'summary'])
        """
        # Initialize our Epsilon-Greedy Bandit with the modules
        self.bandit = EpsilonGreedyBandit(arms=study_modules)

    def get_next_recommendation(self):
        """Asks the bandit algorithm to choose the next best study module."""
        return self.bandit.select_arm()

    def process_student_feedback(self, chosen_module, score, passing_threshold=70):
        """
        Processes a student's performance score, converts it into a reward (0 or 1),
        and updates the bandit's learning model.

        :param chosen_module: The module the student just completed
        :param score: The student's score (e.g., out of 100)
        :param passing_threshold: The score required to count as a 'success' (default 70)
        """
        # Convert raw score into a binary reward: 1 if passed, 0 if failed
        reward = 1 if score >= passing_threshold else 0

        # Feed the reward back into the bandit
        self.bandit.update(chosen_module, reward)

        return reward