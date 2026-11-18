"""Day 7: edge cases (empty history, single session, extreme values, unknown arms,
missing check-in fields), constructor validation, docstrings and plain-script use."""
import importlib
import inspect
import os
import pkgutil
import subprocess
import sys
import tempfile
import unittest

import ai_core
from ai_core.bandit import EpsilonGreedyBandit
from ai_core.fruitfulness import fruitfulness_score
from ai_core.optimizer import StudyOptimizer
from ai_core.thompson import ThompsonBandit

ARMS = ["quiz", "flashcard", "summary"]
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestEmptyHistory(unittest.TestCase):
    def test_bandit_posterior_is_the_prior_for_every_arm(self):
        bandit = ThompsonBandit(ARMS)
        for arm in ARMS:
            mean, std = bandit.posterior(arm)
            self.assertAlmostEqual(mean, 0.5)
            self.assertAlmostEqual(std, 0.5)

    def test_cold_start_recommendation_is_stable(self):
        opt = StudyOptimizer(ARMS)
        self.assertEqual(opt.recommend_session(), opt.recommend_session())

    def test_calibrator_with_no_data_has_prior_uncertainty(self):
        opt = StudyOptimizer(ARMS)
        for category in (None, "reading"):
            mean, std = opt.calibrator.posterior(category)
            self.assertAlmostEqual(mean, 0.0)
            self.assertAlmostEqual(std, 0.3)


class TestSingleSession(unittest.TestCase):
    def test_one_reward_moves_belief_part_of_the_way(self):
        bandit = ThompsonBandit(ARMS)
        bandit.update("quiz", 1.0)
        mean, std = bandit.posterior("quiz")
        self.assertGreater(mean, 0.5)
        self.assertLess(mean, 1.0)
        self.assertLess(std, 0.5)

    def test_recommendation_after_one_checkin_is_valid(self):
        opt = StudyOptimizer(ARMS)
        opt.process_checkin("quiz", 8, 10, 4, 1, 4, time_slot="morning", session_length=25)
        for _ in range(20):
            s = opt.recommend_session()
            self.assertIn(s["method"], ARMS)
            self.assertIn(s["time_slot"], opt.slot_bandit.arms)
            self.assertIn(s["length"], opt.length_bandit.arms)
            self.assertTrue(s["reason"].endswith("."))

    def test_one_slow_task_gives_only_a_partial_correction(self):
        opt = StudyOptimizer(ARMS)
        opt.record_task_time(60, 120)
        self.assertGreater(opt.calibrated_estimate(60), 60)
        self.assertLess(opt.calibrated_estimate(60), 120)


class TestExtremeValues(unittest.TestCase):
    def test_huge_progress_and_distractions_are_capped(self):
        # completion 1.0, focus 5, distractions maxed out, energy 5
        self.assertAlmostEqual(fruitfulness_score(1e9, 1, 5, 10 ** 6, 5), 0.8)

    def test_tiny_progress_stays_in_range(self):
        score = fruitfulness_score(0.0001, 1000, 1, 0, 1)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_bandit_accepts_rewards_outside_zero_to_one(self):
        bandit = ThompsonBandit(ARMS)
        bandit.update("quiz", 5.0)
        bandit.update("flashcard", -3.0)
        self.assertIn(bandit.select_arm(), ARMS)

    def test_many_updates_stay_numerically_stable(self):
        bandit = ThompsonBandit(ARMS)
        for _ in range(5000):
            bandit.update("quiz", 0.7)
        mean, std = bandit.posterior("quiz")
        self.assertAlmostEqual(mean, 0.7, places=3)
        self.assertGreater(std, 0)
        self.assertLess(std, 0.01)


class TestUnknownArms(unittest.TestCase):
    def test_both_bandits_reject_unknown_arms(self):
        for bandit in (ThompsonBandit(ARMS), EpsilonGreedyBandit(ARMS)):
            with self.assertRaises(ValueError):
                bandit.update("nonexistent", 0.5)

    def test_unknown_method_changes_nothing(self):
        opt = StudyOptimizer(ARMS)
        with self.assertRaises(ValueError):
            opt.process_checkin("nope", 8, 10, 4, 1, 4,
                                time_slot="morning", session_length=25)
        self._assert_untouched(opt)

    def test_session_length_of_wrong_type_is_rejected(self):
        opt = StudyOptimizer(ARMS)
        with self.assertRaises(ValueError):
            opt.process_checkin("quiz", 8, 10, 4, 1, 4, session_length="25")
        self._assert_untouched(opt)

    def _assert_untouched(self, opt):
        for bandit in (opt.bandit, opt.slot_bandit, opt.length_bandit):
            self.assertEqual(sum(bandit.counts.values()), 0)


class TestMissingCheckinFields(unittest.TestCase):
    GOOD = (8, 10, 4, 1, 4)

    def test_missing_argument_raises_type_error(self):
        opt = StudyOptimizer(ARMS)
        with self.assertRaises(TypeError):
            opt.process_checkin("quiz", 8, 10, 4, 1)  # no energy

    def test_none_in_any_field_raises_value_error(self):
        for i in range(5):
            fields = list(self.GOOD)
            fields[i] = None
            with self.assertRaises(ValueError, msg=f"field {i}"):
                fruitfulness_score(*fields)

    def test_non_numeric_nan_infinite_and_bool_are_rejected(self):
        for bad in ("5", float("nan"), float("inf"), True, [3]):
            with self.assertRaises(ValueError, msg=repr(bad)):
                fruitfulness_score(bad, 10, 4, 1, 4)

    def test_bad_checkin_changes_nothing(self):
        opt = StudyOptimizer(ARMS)
        with self.assertRaises(ValueError):
            opt.process_checkin("quiz", None, 10, 4, 1, 4,
                                time_slot="morning", session_length=25)
        for bandit in (opt.bandit, opt.slot_bandit, opt.length_bandit):
            self.assertEqual(sum(bandit.counts.values()), 0)


class TestConstructorValidation(unittest.TestCase):
    def test_empty_arms_rejected(self):
        with self.assertRaises(ValueError):
            ThompsonBandit([])

    def test_duplicate_arms_rejected(self):
        with self.assertRaises(ValueError):
            ThompsonBandit(["quiz", "quiz", "summary"])

    def test_optimizer_with_no_methods_rejected(self):
        with self.assertRaises(ValueError):
            StudyOptimizer([])

    def test_optimizer_with_empty_time_slots_rejected(self):
        with self.assertRaises(ValueError):
            StudyOptimizer(ARMS, time_slots=[])


class TestDocstrings(unittest.TestCase):
    def test_every_public_module_class_and_function_has_a_docstring(self):
        missing = []
        for info in pkgutil.iter_modules(ai_core.__path__):
            module = importlib.import_module(f"ai_core.{info.name}")
            if not module.__doc__:
                missing.append(module.__name__)
            for name, obj in vars(module).items():
                if name.startswith("_") or getattr(obj, "__module__", None) != module.__name__:
                    continue
                if inspect.isfunction(obj):
                    if not obj.__doc__:
                        missing.append(f"{module.__name__}.{name}")
                elif inspect.isclass(obj):
                    if not obj.__doc__:
                        missing.append(f"{module.__name__}.{name}")
                    for method_name, method in vars(obj).items():
                        if method_name.startswith("_"):
                            continue
                        func = method.__func__ if isinstance(method, (staticmethod, classmethod)) else method
                        if inspect.isfunction(func) and not func.__doc__:
                            missing.append(f"{module.__name__}.{name}.{method_name}")
        self.assertEqual(missing, [], f"Missing docstrings: {missing}")


class TestPlainScriptUsage(unittest.TestCase):
    SCRIPT = """
from ai_core.storage import load_or_create, save_state

opt = load_or_create("student.db", ["quiz", "flashcard", "summary"])
s = opt.recommend_session()
opt.process_checkin(s["method"], 4, 5, 4, 1, 3,
                    time_slot=s["time_slot"], session_length=s["length"])
opt.record_task_time(60, 84, "reading")
save_state(opt, "student.db")
print(sum(opt.bandit.counts.values()))
"""

    def test_script_works_and_state_survives_restarts(self):
        env = dict(os.environ, PYTHONPATH=PROJECT_ROOT)
        with tempfile.TemporaryDirectory() as tmp:
            script = os.path.join(tmp, "use_ai_core.py")
            with open(script, "w") as f:
                f.write(self.SCRIPT)
            outputs = []
            for _ in range(2):  # two separate processes = a restart
                result = subprocess.run([sys.executable, script], cwd=tmp, env=env,
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                outputs.append(result.stdout.strip())
            self.assertEqual(outputs, ["1", "2"])


if __name__ == "__main__":
    unittest.main()