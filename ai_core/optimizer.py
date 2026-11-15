from ai_core.thompson import ThompsonBandit
from ai_core.fruitfulness import fruitfulness_score

DEFAULT_TIME_SLOTS = ["morning", "afternoon", "evening", "night"]
DEFAULT_SESSION_LENGTHS = [25, 45, 60]  # minutes


class StudyOptimizer:
    def __init__(self, study_modules, bandit=None, time_slots=None, session_lengths=None):
        """
        :param study_modules: List of available study methods (e.g., ['quiz', 'flashcard', 'summary'])
        :param bandit: Optional bandit for the method decision; defaults to Thompson sampling
        :param time_slots: Optional list of time slots; defaults to DEFAULT_TIME_SLOTS
        :param session_lengths: Optional list of lengths in minutes; defaults to DEFAULT_SESSION_LENGTHS
        """
        self.bandit = bandit if bandit is not None else ThompsonBandit(arms=study_modules)
        self.slot_bandit = ThompsonBandit(
            arms=time_slots if time_slots is not None else DEFAULT_TIME_SLOTS)
        self.length_bandit = ThompsonBandit(
            arms=session_lengths if session_lengths is not None else DEFAULT_SESSION_LENGTHS)

    def get_next_recommendation(self):
        """Asks the bandit algorithm to choose the next best study method."""
        return self.bandit.select_arm()

    def recommend_session(self):
        """Recommends a full session: method, time slot and length (minutes)."""
        return {
            "method": self.bandit.select_arm(),
            "time_slot": self.slot_bandit.select_arm(),
            "length": self.length_bandit.select_arm(),
        }

    def process_student_feedback(self, chosen_module, score, passing_threshold=70):
        """Legacy Day 1 path: converts a quiz score to a 0/1 reward."""
        reward = 1 if score >= passing_threshold else 0
        self.bandit.update(chosen_module, reward)
        return reward

    def process_checkin(self, chosen_module, units_done, units_planned,
                        focus, distractions, energy,
                        time_slot=None, session_length=None):
        """Scores a session from check-in data and feeds the reward to every bandit
        whose choice is given. time_slot and session_length are optional."""
        reward = fruitfulness_score(units_done, units_planned, focus, distractions, energy)

        # Validate everything first so a bad value never leaves a half-updated state.
        if chosen_module not in self.bandit.counts:
            raise ValueError(f"Unknown method: {chosen_module!r}")
        if time_slot is not None and time_slot not in self.slot_bandit.counts:
            raise ValueError(f"Unknown time slot: {time_slot!r}")
        if session_length is not None and session_length not in self.length_bandit.counts:
            raise ValueError(f"Unknown session length: {session_length!r}")

        self.bandit.update(chosen_module, reward)
        if time_slot is not None:
            self.slot_bandit.update(time_slot, reward)
        if session_length is not None:
            self.length_bandit.update(session_length, reward)
        return reward