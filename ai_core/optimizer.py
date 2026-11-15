from ai_core.calibration import TimeCalibrator
from ai_core.fruitfulness import fruitfulness_score
from ai_core.thompson import ThompsonBandit

DEFAULT_TIME_SLOTS = ["morning", "afternoon", "evening", "night"]
DEFAULT_SESSION_LENGTHS = [25, 45, 60]  # minutes

# Cold-start defaults: used for a decision until it has any data.
DEFAULT_TIME_SLOT = "morning"
DEFAULT_LENGTH = 25  # the gentlest option


def _start_default(arms, preferred):
    return preferred if preferred in arms else arms[0]


def _reason_part(label, bandit, arm, unit=""):
    name = f"{arm}{unit}"
    if sum(bandit.counts.values()) == 0:
        return f"{label} {name}: starting default, no history yet"
    n = bandit.counts[arm]
    if n == 0:
        return f"{label} {name}: not tried yet, worth testing"
    wins = getattr(bandit, "wins", None)
    if wins is None:  # e.g. the epsilon-greedy baseline doesn't track wins
        return f"{label} {name}: tried in {n} sessions"
    return f"{label} {name}: worked well in {wins[arm]} of {n} sessions"


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
        self.calibrator = TimeCalibrator()

    def get_next_recommendation(self):
        """Asks the bandit algorithm to choose the next best study method."""
        return self.bandit.select_arm()

    @staticmethod
    def _pick(bandit, preferred_default):
        """Use a sensible default until the decision has any data, then learn."""
        if sum(bandit.counts.values()) == 0:
            return _start_default(bandit.arms, preferred_default)
        return bandit.select_arm()

    def recommend_session(self):
        """Recommends a full session: method, time slot, length (minutes) and a reason."""
        method = self._pick(self.bandit, self.bandit.arms[0])
        slot = self._pick(self.slot_bandit, DEFAULT_TIME_SLOT)
        length = self._pick(self.length_bandit, DEFAULT_LENGTH)
        reason = ". ".join([
            _reason_part("Method", self.bandit, method),
            _reason_part("Time slot", self.slot_bandit, slot),
            _reason_part("Length", self.length_bandit, length, unit=" min"),
        ]) + "."
        return {"method": method, "time_slot": slot, "length": length, "reason": reason}

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

    def record_task_time(self, estimated_minutes, actual_minutes, category=None):
        """Teach the calibrator how long a task really took."""
        self.calibrator.record(estimated_minutes, actual_minutes, category)

    def calibrated_estimate(self, estimate_minutes, category=None):
        """Return the student's estimate corrected by what we've learned about them."""
        return self.calibrator.calibrate(estimate_minutes, category)