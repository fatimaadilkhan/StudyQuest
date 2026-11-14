"""Fruitfulness score: how productive a study session was, from 0 to 1.

Computed from the student's check-in, with no quiz needed.

Weights (sum to 1.0):
    completion   0.40  units done vs planned (capped at 100%)
    focus        0.30  self-rated focus, 1-5
    distractions 0.20  count of interruptions; MAX_DISTRACTIONS or more scores 0
    energy       0.10  self-rated energy, 1-5
"""

WEIGHTS = {
    "completion": 0.40,
    "focus": 0.30,
    "distractions": 0.20,
    "energy": 0.10,
}

RATING_MIN, RATING_MAX = 1, 5
MAX_DISTRACTIONS = 5


def _normalise_rating(value, name):
    if not RATING_MIN <= value <= RATING_MAX:
        raise ValueError(f"{name} must be between {RATING_MIN} and {RATING_MAX}, got {value}")
    return (value - RATING_MIN) / (RATING_MAX - RATING_MIN)


def fruitfulness_score(units_done, units_planned, focus, distractions, energy):
    """Return a score in [0, 1] for one study session."""
    if units_planned <= 0:
        raise ValueError(f"units_planned must be positive, got {units_planned}")
    if units_done < 0:
        raise ValueError(f"units_done cannot be negative, got {units_done}")
    if distractions < 0:
        raise ValueError(f"distractions cannot be negative, got {distractions}")

    completion = min(units_done / units_planned, 1.0)
    focus_score = _normalise_rating(focus, "focus")
    energy_score = _normalise_rating(energy, "energy")
    distraction_score = 1.0 - min(distractions, MAX_DISTRACTIONS) / MAX_DISTRACTIONS

    score = (
        WEIGHTS["completion"] * completion
        + WEIGHTS["focus"] * focus_score
        + WEIGHTS["distractions"] * distraction_score
        + WEIGHTS["energy"] * energy_score
    )
    return max(0.0, min(1.0, score))  # guard against float drift