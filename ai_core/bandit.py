import random


class EpsilonGreedyBandit:
    def __init__(self, arms, epsilon=0.1):
        """
        :param arms: List of str/int representing study modules or resource types
        :param epsilon: Probability of exploring a random arm instead of the best one
        """
        self.arms = list(arms)
        self.epsilon = epsilon

        self.counts = {arm: 0 for arm in self.arms}
        self.values = {arm: 0.0 for arm in self.arms}

    def select_arm(self):
        """Selects an arm using the Epsilon-Greedy strategy."""
        # Fix 1: try every arm at least once before exploiting
        untried = [arm for arm in self.arms if self.counts[arm] == 0]
        if untried:
            return random.choice(untried)

        # Exploration
        if random.random() < self.epsilon:
            return random.choice(self.arms)

        # Exploitation, Fix 2: break ties randomly instead of always taking the first arm
        best_value = max(self.values.values())
        best_arms = [arm for arm in self.arms if self.values[arm] == best_value]
        return random.choice(best_arms)

    def update(self, chosen_arm, reward):
        """Updates the average reward for the chosen arm."""
        # Fix 3: validate the arm
        if chosen_arm not in self.counts:
            raise ValueError(f"Unknown arm: {chosen_arm!r}. Valid arms: {self.arms}")

        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]

        current_value = self.values[chosen_arm]
        self.values[chosen_arm] = current_value + (reward - current_value) / n