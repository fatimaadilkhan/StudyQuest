import random


class ThompsonBandit:
    """Normal-Normal Thompson sampling bandit.

    Each arm's mean reward has a Normal prior (mean 0.5, std 0.5, i.e. variance
    0.25). Rewards are assumed Normal around the arm's mean with a known noise
    spread obs_std = 0.3 (variance 0.09). That value is a starting guess for how
    much a session's fruitfulness score varies for the same choice; re-estimate
    it from real check-ins when they exist.

    A "win" is a session whose reward is at least win_threshold. Wins are only
    used to explain recommendations; they don't affect the learning.

    Same interface as EpsilonGreedyBandit: select_arm() and update().
    """

    def __init__(self, arms, prior_mean=0.5, prior_std=0.5, obs_std=0.3,
                 win_threshold=0.6):
        if prior_std <= 0 or obs_std <= 0:
            raise ValueError("prior_std and obs_std must be positive")
        self.arms = list(arms)
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.obs_std = obs_std
        self.win_threshold = win_threshold

        self.counts = {arm: 0 for arm in self.arms}
        self.sums = {arm: 0.0 for arm in self.arms}
        self.wins = {arm: 0 for arm in self.arms}
        self.values = {arm: 0.0 for arm in self.arms}  # sample mean, as in the baseline

    def posterior(self, arm):
        """Return (mean, std) of the belief about this arm's average reward."""
        prior_prec = 1.0 / self.prior_std ** 2
        obs_prec = 1.0 / self.obs_std ** 2
        precision = prior_prec + self.counts[arm] * obs_prec
        mean = (prior_prec * self.prior_mean + obs_prec * self.sums[arm]) / precision
        return mean, precision ** -0.5

    def select_arm(self):
        """Draw one sample from each arm's posterior and pick the highest."""
        samples = {a: random.gauss(*self.posterior(a)) for a in self.arms}
        return max(samples, key=lambda name: samples[name])

    def update(self, chosen_arm, reward):
        if chosen_arm not in self.counts:
            raise ValueError(f"Unknown arm: {chosen_arm!r}. Valid arms: {self.arms}")
        self.counts[chosen_arm] += 1
        self.sums[chosen_arm] += reward
        if reward >= self.win_threshold:
            self.wins[chosen_arm] += 1
        self.values[chosen_arm] = self.sums[chosen_arm] / self.counts[chosen_arm]