"""Save and load the AI core's learned state through SQLite.

One state file holds one student's StudyOptimizer: the three bandits
(method, time slot, session length) and the time-estimate calibrator.
"""
import json
import os
import sqlite3
from contextlib import closing

from ai_core.calibration import TimeCalibrator
from ai_core.optimizer import StudyOptimizer
from ai_core.thompson import ThompsonBandit

SCHEMA_VERSION = 1

# bandit name in the file -> attribute on StudyOptimizer
_BANDITS = {"method": "bandit", "slot": "slot_bandit", "length": "length_bandit"}

_SCHEMA = [
    "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
    """CREATE TABLE IF NOT EXISTS bandit_params (
           bandit TEXT PRIMARY KEY, prior_mean REAL, prior_std REAL,
           obs_std REAL, win_threshold REAL)""",
    """CREATE TABLE IF NOT EXISTS bandit_arms (
           bandit TEXT, position INTEGER, arm TEXT, count INTEGER,
           reward_sum REAL, wins INTEGER, PRIMARY KEY (bandit, position))""",
    """CREATE TABLE IF NOT EXISTS calibration (
           scope TEXT, category TEXT, count INTEGER, log_sum REAL,
           PRIMARY KEY (scope, category))""",
]


def save_state(optimizer, path):
    """Write the optimizer's learned state to `path`, replacing any earlier state.

    The write is one transaction, so a crash can't leave a half-written file.
    """
    for attr in _BANDITS.values():
        if not isinstance(getattr(optimizer, attr), ThompsonBandit):
            raise TypeError("save_state only supports ThompsonBandit; "
                            f"{attr} is {type(getattr(optimizer, attr)).__name__}")

    with closing(sqlite3.connect(path)) as conn:
        with conn:  # commit on success, roll back on error
            for statement in _SCHEMA:
                conn.execute(statement)
            for table in ("meta", "bandit_params", "bandit_arms", "calibration"):
                conn.execute(f"DELETE FROM {table}")

            cal = optimizer.calibrator
            conn.executemany("INSERT INTO meta VALUES (?, ?)", [
                ("schema_version", str(SCHEMA_VERSION)),
                ("cal_prior_std", str(cal.prior_std)),
                ("cal_obs_std", str(cal.obs_std)),
            ])

            for name, attr in _BANDITS.items():
                b = getattr(optimizer, attr)
                conn.execute("INSERT INTO bandit_params VALUES (?, ?, ?, ?, ?)",
                             (name, b.prior_mean, b.prior_std, b.obs_std, b.win_threshold))
                conn.executemany(
                    "INSERT INTO bandit_arms VALUES (?, ?, ?, ?, ?, ?)",
                    [(name, i, json.dumps(arm), b.counts[arm], b.sums[arm], b.wins[arm])
                     for i, arm in enumerate(b.arms)])

            state = cal.get_state()
            conn.execute("INSERT INTO calibration VALUES ('overall', '', ?, ?)",
                         state["overall"])
            conn.executemany(
                "INSERT INTO calibration VALUES ('category', ?, ?, ?)",
                [(c, n, t) for c, (n, t) in state["by_category"].items()])


def _meta(conn, key):
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    if row is None:
        raise ValueError(f"State file is missing '{key}'")
    return row[0]


def load_state(path):
    """Rebuild a StudyOptimizer from a file written by save_state()."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No state file at {path!r}")

    with closing(sqlite3.connect(path)) as conn:
        try:
            if int(_meta(conn, "schema_version")) != SCHEMA_VERSION:
                raise ValueError("Unsupported state file version")

            bandits = {}
            for name in _BANDITS:
                params = conn.execute(
                    "SELECT prior_mean, prior_std, obs_std, win_threshold "
                    "FROM bandit_params WHERE bandit = ?", (name,)).fetchone()
                rows = conn.execute(
                    "SELECT arm, count, reward_sum, wins FROM bandit_arms "
                    "WHERE bandit = ? ORDER BY position", (name,)).fetchall()
                if params is None or not rows:
                    raise ValueError(f"State file has no data for the {name} bandit")

                arms = [json.loads(r[0]) for r in rows]
                bandit = ThompsonBandit(arms, prior_mean=params[0], prior_std=params[1],
                                        obs_std=params[2], win_threshold=params[3])
                for arm, (_, count, total, wins) in zip(arms, rows):
                    bandit.counts[arm] = count
                    bandit.sums[arm] = total
                    bandit.wins[arm] = wins
                    bandit.values[arm] = total / count if count else 0.0
                bandits[name] = bandit

            calibrator = TimeCalibrator(prior_std=float(_meta(conn, "cal_prior_std")),
                                        obs_std=float(_meta(conn, "cal_obs_std")))
            overall, by_category = (0, 0.0), {}
            for scope, category, n, total in conn.execute(
                    "SELECT scope, category, count, log_sum FROM calibration"):
                if scope == "overall":
                    overall = (n, total)
                else:
                    by_category[category] = (n, total)
            calibrator.set_state(overall, by_category)
        except sqlite3.DatabaseError as exc:  # not a database, or tables missing
            raise ValueError(f"{path!r} is not a valid StudyQuest state file") from exc

    optimizer = StudyOptimizer(bandits["method"].arms, bandit=bandits["method"])
    optimizer.slot_bandit = bandits["slot"]
    optimizer.length_bandit = bandits["length"]
    optimizer.calibrator = calibrator
    return optimizer


def load_or_create(path, study_modules, **optimizer_options):
    """Load saved state if `path` exists, otherwise start a fresh optimizer.

    `optimizer_options` (time_slots, session_lengths) only apply to a fresh one.
    """
    if os.path.exists(path):
        return load_state(path)
    return StudyOptimizer(study_modules, **optimizer_options)