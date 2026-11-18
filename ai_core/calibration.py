"""Learns how long tasks really take compared with a student's estimates."""
import math


class TimeCalibrator:
    """Learns how long tasks really take compared with a student's estimates.

    Works on log(actual / estimated). The mean log ratio has a Normal prior
    centred on 0 (estimates are accurate); observations are Normal with a known
    spread (obs_std). A category's prior is centred on the overall correction,
    so a new category starts from the student's general habit.
    """

    def __init__(self, prior_std=0.3, obs_std=0.4):
        if prior_std <= 0 or obs_std <= 0:
            raise ValueError("prior_std and obs_std must be positive")
        self.prior_std = prior_std
        self.obs_std = obs_std
        self._overall = (0, 0.0)   # (count, sum of log ratios)
        self._by_category = {}     # category -> (count, sum of log ratios)

    def record(self, estimated, actual, category=None):
        """Record one finished task (both durations in the same unit, > 0)."""
        if estimated <= 0 or actual <= 0:
            raise ValueError(f"estimated and actual must be positive, got {estimated}, {actual}")
        log_ratio = math.log(actual / estimated)

        n, total = self._overall
        self._overall = (n + 1, total + log_ratio)
        if category is not None:
            n, total = self._by_category.get(category, (0, 0.0))
            self._by_category[category] = (n + 1, total + log_ratio)

    def _posterior(self, n, total, prior_mean):
        prior_prec = 1.0 / self.prior_std ** 2
        obs_prec = 1.0 / self.obs_std ** 2
        precision = prior_prec + n * obs_prec
        mean = (prior_prec * prior_mean + obs_prec * total) / precision
        return mean, precision ** -0.5

    def posterior(self, category=None):
        """Return (mean, std) of the belief about the log of actual/estimated."""
        overall_mean, overall_std = self._posterior(*self._overall, prior_mean=0.0)
        if category is None:
            return overall_mean, overall_std
        n, total = self._by_category.get(category, (0, 0.0))
        return self._posterior(n, total, prior_mean=overall_mean)

    def multiplier(self, category=None):
        """Factor to multiply a student's estimate by (1.0 = estimates are accurate)."""
        mean, _ = self.posterior(category)
        return math.exp(mean)

    def calibrate(self, estimate, category=None):
        """Return the corrected duration for a student's raw estimate."""
        if estimate <= 0:
            raise ValueError(f"estimate must be positive, got {estimate}")
        return estimate * self.multiplier(category)


    def get_state(self):
        """Return the learned data: {'overall': (count, log_sum), 'by_category': {name: (count, log_sum)}}."""
        return {"overall": self._overall, "by_category": dict(self._by_category)}

    def set_state(self, overall, by_category):
        """Restore data returned by get_state()."""
        self._overall = (int(overall[0]), float(overall[1]))
        self._by_category = {c: (int(n), float(t)) for c, (n, t) in by_category.items()}