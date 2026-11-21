import random
import unittest

from ai_core.optimizer import StudyOptimizer
from ai_core.planner import (PlannerConfig, StudyPlanProblem, Task, anneal,
                             effectiveness_from_optimizer, format_timetable,
                             random_baseline)

SLOT_QUALITY = {"morning": 0.7, "afternoon": 0.55, "evening": 0.65, "night": 0.3}


def week_problem():
    """A 15-block week in which a clash-free, on-time plan with a rest day exists."""
    tasks = [Task("Lab report", 135, 2), Task("Math exam", 180, 3),
             Task("AI project", 270, 5), Task("Reading", 90, 6)]
    return StudyPlanProblem(tasks, PlannerConfig(slot_effectiveness=SLOT_QUALITY))


def two_day_problem(task):
    config = PlannerConfig(n_days=2, block_minutes=60,
                           slot_effectiveness={"morning": 0.8, "afternoon": 0.5,
                                               "evening": 0.5, "night": 0.5})
    return StudyPlanProblem([task], config)


class TestProblemSetup(unittest.TestCase):
    def test_tasks_expand_into_blocks_rounding_up(self):
        problem = StudyPlanProblem([Task("A", 100, 3), Task("B", 45, 3)])
        self.assertEqual([t.name for t in problem.blocks], ["A", "A", "A", "B"])

    def test_earlier_deadline_and_higher_importance_mean_more_urgency(self):
        problem = StudyPlanProblem([Task("soon", 45, 0), Task("later", 45, 6),
                                    Task("vital", 45, 6, importance=2.0)])
        self.assertGreater(problem.urgency["soon"], problem.urgency["later"])
        self.assertAlmostEqual(problem.urgency["later"], 1.0)
        self.assertAlmostEqual(problem.urgency["vital"], 2.0)

    def test_invalid_tasks_rejected(self):
        for bad in (Task("", 60, 1), Task("A", 0, 1), Task("A", 60, 1, importance=0),
                    Task("A", 60, -1), Task("A", 60, 7)):
            with self.assertRaises(ValueError, msg=repr(bad)):
                StudyPlanProblem([bad])

    def test_empty_and_duplicate_tasks_rejected(self):
        with self.assertRaises(ValueError):
            StudyPlanProblem([])
        with self.assertRaises(ValueError):
            StudyPlanProblem([Task("A", 60, 1), Task("A", 60, 2)])

    def test_more_blocks_than_slots_rejected(self):
        with self.assertRaises(ValueError):
            StudyPlanProblem([Task("A", 5 * 45, 0)], PlannerConfig(n_days=1))


class TestObjective(unittest.TestCase):
    def test_on_time_block_earns_urgency_weighted_effective_minutes(self):
        problem = two_day_problem(Task("A", 60, 1))  # urgency 1.0
        result = problem.score([(0, 0)])             # morning, effectiveness 0.8
        self.assertAlmostEqual(result["effective_minutes"], 48.0)
        self.assertAlmostEqual(result["total"], 48.0)

    def test_late_block_earns_nothing_and_is_penalised(self):
        problem = two_day_problem(Task("A", 60, 0))
        result = problem.score([(1, 0)])
        self.assertAlmostEqual(result["effective_minutes"], 0.0)
        self.assertAlmostEqual(result["late_penalty"], 80.0)
        self.assertAlmostEqual(result["total"], -80.0)

    def test_clash_is_penalised(self):
        problem = two_day_problem(Task("A", 120, 1))
        result = problem.score([(0, 0), (0, 0)])
        self.assertEqual(problem.violations([(0, 0), (0, 0)])["clashes"], 1)
        self.assertAlmostEqual(result["total"], 96.0 - 100.0)

    def test_overloaded_day_is_penalised_per_minute(self):
        problem = StudyPlanProblem([Task("A", 180, 6)])  # 4 blocks of 45, cap 135
        timetable = [(0, 0), (0, 1), (0, 2), (0, 3)]
        self.assertEqual(problem.violations(timetable)["overload_minutes"], 45)
        self.assertAlmostEqual(problem.score(timetable)["total"], 4 * 22.5 - 45)

    def test_missing_rest_day_is_penalised(self):
        problem = two_day_problem(Task("A", 120, 1))
        timetable = [(0, 0), (1, 0)]  # studying on both days
        self.assertEqual(problem.violations(timetable)["missing_rest_days"], 1)
        self.assertAlmostEqual(problem.score(timetable)["total"], 96.0 - 60.0)

    def test_better_slot_scores_higher(self):
        problem = two_day_problem(Task("A", 60, 1))
        self.assertGreater(problem.score([(0, 0)])["total"], problem.score([(0, 3)])["total"])

    def test_is_valid(self):
        problem = two_day_problem(Task("A", 120, 1))
        self.assertTrue(problem.is_valid([(0, 0), (0, 1)]))
        self.assertFalse(problem.is_valid([(0, 0), (0, 0)]))   # clash
        self.assertFalse(problem.is_valid([(0, 0)]))           # wrong length
        self.assertFalse(problem.is_valid([(0, 0), (2, 0)]))   # day out of range
        self.assertFalse(problem.is_valid([(0, 0), (0, 4)]))   # slot out of range


class TestAnnealing(unittest.TestCase):
    def setUp(self):
        self.problem = week_problem()

    def test_result_is_valid_on_time_and_penalty_free(self):
        result = anneal(self.problem, seed=5)
        self.assertTrue(self.problem.is_valid(result.timetable))
        self.assertEqual(self.problem.violations(result.timetable),
                         {"clashes": 0, "late_blocks": 0,
                          "overload_minutes": 0, "missing_rest_days": 0})

    def test_beats_random_timetables(self):
        baseline = random_baseline(self.problem, n=300, seed=0)
        result = anneal(self.problem, seed=1)
        self.assertGreater(result.score, baseline["best"])
        self.assertGreater(result.score, baseline["mean"])
        self.assertLess(baseline["valid_share"], 0.5)  # random plans mostly clash

    def test_reported_score_matches_the_objective(self):
        result = anneal(self.problem, seed=2)
        self.assertAlmostEqual(result.score, self.problem.score(result.timetable)["total"])
        self.assertAlmostEqual(result.score, result.breakdown["total"])

    def test_same_seed_is_reproducible(self):
        a, b = anneal(self.problem, seed=3), anneal(self.problem, seed=3)
        self.assertEqual(a.timetable, b.timetable)
        self.assertEqual(a.score, b.score)

    def test_history_never_decreases_and_has_the_expected_length(self):
        result = anneal(self.problem, iterations=1000, record_every=100, seed=4)
        self.assertEqual(len(result.history), 10)
        self.assertTrue(all(b >= a for a, b in zip(result.history, result.history[1:])))
        self.assertLessEqual(result.history[-1], result.score)

    def test_accepts_some_but_not_all_moves(self):
        result = anneal(self.problem, iterations=3000, seed=6)
        self.assertEqual(result.iterations, 3000)
        self.assertGreater(result.accepted, 0)
        self.assertLess(result.accepted, 3000)

    def test_never_ends_worse_than_the_starting_timetable(self):
        start = self.problem.random_timetable(random.Random(9))
        result = anneal(self.problem, initial=start, seed=9)
        self.assertGreaterEqual(result.score, self.problem.score(start)["total"])

    def test_single_block_problem_works(self):
        problem = StudyPlanProblem([Task("Only", 30, 2)])
        result = anneal(problem, iterations=200, seed=0)
        self.assertTrue(problem.is_valid(result.timetable))
        self.assertLessEqual(result.timetable[0][0], 2)

    def test_invalid_parameters_rejected(self):
        with self.assertRaises(ValueError):
            anneal(self.problem, iterations=0)
        with self.assertRaises(ValueError):
            anneal(self.problem, start_temp=1.0, end_temp=2.0)
        with self.assertRaises(ValueError):
            anneal(self.problem, end_temp=0)
        with self.assertRaises(ValueError):
            anneal(self.problem, initial=[(0, 0)])


class TestHelpers(unittest.TestCase):
    def test_random_baseline_is_reproducible(self):
        problem = week_problem()
        self.assertEqual(random_baseline(problem, 50, seed=1), random_baseline(problem, 50, seed=1))

    def test_effectiveness_comes_from_the_learned_slot_bandit(self):
        opt = StudyOptimizer(["quiz", "flashcard"])
        for _ in range(5):
            opt.process_checkin("quiz", 10, 10, 5, 0, 5, time_slot="evening")   # score 1.0
            opt.process_checkin("quiz", 0, 10, 1, 5, 1, time_slot="night")      # score 0.0
        eff = effectiveness_from_optimizer(opt)
        self.assertEqual(set(eff), {"morning", "afternoon", "evening", "night"})
        self.assertTrue(all(0.0 <= v <= 1.0 for v in eff.values()))
        self.assertGreater(eff["evening"], eff["night"])
        problem = StudyPlanProblem([Task("A", 45, 3)], PlannerConfig(slot_effectiveness=eff))
        self.assertAlmostEqual(problem.effectiveness["evening"], eff["evening"])

    def test_format_timetable_shows_tasks_labels_and_clashes(self):
        problem = StudyPlanProblem([Task("Math", 90, 2)])
        text = format_timetable(problem, [(0, 0), (0, 0)], day_names=list("MTWTFSS"))
        self.assertIn("Math (clash)", text)
        self.assertIn("morning", text)
        self.assertEqual(len(text.split("\n")), 8)  # header + 7 days
        self.assertIn("Day 0", format_timetable(problem, [(0, 0), (1, 1)]))

    def test_format_timetable_needs_one_label_per_day(self):
        problem = StudyPlanProblem([Task("Math", 45, 2)])
        with self.assertRaises(ValueError):
            format_timetable(problem, [(0, 0)], day_names=["Mon"])


if __name__ == "__main__":
    unittest.main()