import random
import unittest

from ai_core.optimizer import StudyOptimizer
from ai_core.planner import (PlannerConfig, StudyPlanProblem, Task, anneal,
                             effectiveness_from_optimizer, equal_split_timetable,
                             explain_plan, format_timetable, plan_metrics, plan_week,
                             random_baseline, replan)
from ai_core.planner_eval import (compare_planners, deadlines_feasible, format_comparison,
                                  random_week, summarize_comparison)

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
        self.assertAlmostEqual(result["late_penalty"], problem.config.late_penalty)
        self.assertAlmostEqual(result["total"], -problem.config.late_penalty)

    def test_clash_is_penalised(self):
        problem = two_day_problem(Task("A", 120, 1))
        result = problem.score([(0, 0), (0, 0)])
        self.assertEqual(problem.violations([(0, 0), (0, 0)])["clashes"], 1)
        self.assertAlmostEqual(result["total"], 96.0 - problem.config.clash_penalty)

    def test_overloaded_day_is_penalised_per_minute(self):
        problem = StudyPlanProblem([Task("A", 180, 6)])  # 4 blocks of 45, cap 135
        timetable = [(0, 0), (0, 1), (0, 2), (0, 3)]
        self.assertEqual(problem.violations(timetable)["overload_minutes"], 45)
        self.assertAlmostEqual(problem.score(timetable)["total"], 4 * 22.5 - 45)

    def test_missing_rest_day_is_penalised(self):
        problem = two_day_problem(Task("A", 120, 1))
        timetable = [(0, 0), (1, 0)]  # studying on both days
        self.assertEqual(problem.violations(timetable)["missing_rest_days"], 1)
        self.assertAlmostEqual(problem.score(timetable)["total"], 96.0 - problem.config.rest_penalty)

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


class TestMetricsAndBaseline(unittest.TestCase):
    def test_plan_metrics_on_a_hand_checked_plan(self):
        problem = two_day_problem(Task("A", 120, 1))
        metrics = plan_metrics(problem, [(0, 0), (0, 0)])
        self.assertEqual((metrics["clashes"], metrics["late_blocks"], metrics["tasks_missed"],
                          metrics["overload_minutes"], metrics["rest_days"]), (1, 0, 0, 0, 1))

    def test_tasks_missed_counts_tasks_not_blocks(self):
        problem = two_day_problem(Task("A", 120, 0))
        metrics = plan_metrics(problem, [(1, 0), (1, 1)])
        self.assertEqual(metrics["late_blocks"], 2)
        self.assertEqual(metrics["tasks_missed"], 1)

    def test_equal_split_spreads_a_task_evenly_and_ignores_slot_quality(self):
        problem = StudyPlanProblem([Task("A", 4 * 45, 3)],
                                   PlannerConfig(slot_effectiveness=SLOT_QUALITY))
        timetable = equal_split_timetable(problem)
        self.assertEqual(sorted(day for day, _ in timetable), [0, 1, 2, 3])
        self.assertTrue(all(slot == 0 for _, slot in timetable))  # always the first slot

    def test_equal_split_is_clash_free_and_on_time_when_there_is_room(self):
        problem = week_problem()
        timetable = equal_split_timetable(problem)
        self.assertTrue(problem.is_valid(timetable))
        self.assertEqual(plan_metrics(problem, timetable)["late_blocks"], 0)

    def test_equal_split_overloads_a_day_that_the_planner_avoids(self):
        problem = week_problem()
        naive = plan_metrics(problem, equal_split_timetable(problem))
        planned = plan_metrics(problem, plan_week(problem, seed=1).timetable)
        self.assertGreater(naive["overload_minutes"], 0)
        self.assertEqual(planned["overload_minutes"], 0)
        self.assertEqual(planned["clashes"], 0)
        self.assertEqual(planned["late_blocks"], 0)

    def test_plan_week_is_never_worse_than_equal_split_and_is_reproducible(self):
        problem = week_problem()
        naive_score = problem.score(equal_split_timetable(problem))["total"]
        a, b = plan_week(problem, seed=7), plan_week(problem, seed=7)
        self.assertGreaterEqual(a.score, naive_score)
        self.assertEqual(a.timetable, b.timetable)


class TestExplain(unittest.TestCase):
    def test_every_block_gets_a_sorted_reason(self):
        problem = week_problem()
        result = plan_week(problem, seed=1)
        rows = explain_plan(problem, result.timetable, ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        self.assertEqual(len(rows), len(problem.blocks))
        order = [(r["day"], problem.config.slots.index(r["slot"])) for r in rows]
        self.assertEqual(order, sorted(order))
        for row in rows:
            self.assertTrue(row["reason"].endswith("."))
            self.assertIn("deadline", row["reason"])
        self.assertTrue(any("best slot" in r["reason"] for r in rows))

    def test_mentions_booked_slots_only_when_the_better_slot_really_is_taken(self):
        problem = two_day_problem(Task("A", 120, 1))  # morning is the best slot
        taken = explain_plan(problem, [(0, 0), (0, 1)])
        self.assertIn("better slots that day are already booked", taken[1]["reason"])
        free = explain_plan(two_day_problem(Task("A", 60, 1)), [(0, 1)])
        self.assertNotIn("already booked", free[0]["reason"])

    def test_flags_late_double_booked_and_overloaded_blocks(self):
        late = explain_plan(two_day_problem(Task("A", 60, 0)), [(1, 0)])
        self.assertIn("after the A deadline", late[0]["reason"])
        self.assertIn("no credit", late[0]["reason"])
        clash = explain_plan(two_day_problem(Task("A", 120, 1)), [(0, 0), (0, 0)])
        self.assertIn("double-booked", clash[0]["reason"])
        heavy = StudyPlanProblem([Task("A", 180, 6)])
        rows = explain_plan(heavy, [(0, 0), (0, 1), (0, 2), (0, 3)])
        self.assertIn("over the daily study limit", rows[0]["reason"])

    def test_deadline_wording_for_the_deadline_day_and_labels(self):
        problem = two_day_problem(Task("A", 60, 1))
        rows = explain_plan(problem, [(1, 0)], ["Mon", "Tue"])
        self.assertIn("on the A deadline day (Tue)", rows[0]["reason"])
        with self.assertRaises(ValueError):
            explain_plan(problem, [(1, 0)], ["Mon"])


class TestReplan(unittest.TestCase):
    def setUp(self):
        self.problem = week_problem()

    def test_replanning_covers_only_the_remaining_work_and_days(self):
        rp = replan(self.problem, {"Lab report": 3, "Math exam": 1}, from_day=2, seed=1)
        self.assertEqual(len(rp.problem.blocks), 11)
        self.assertEqual(rp.problem.config.n_days, 5)
        deadlines = {t.name: t.deadline_day for t in rp.problem.tasks}
        self.assertEqual(deadlines, {"Math exam": 1, "AI project": 3, "Reading": 4})
        self.assertEqual(rp.overdue_tasks, [])
        self.assertTrue(rp.problem.is_valid(rp.result.timetable))
        self.assertEqual(rp.from_day, 2)

    def test_overdue_tasks_become_most_urgent(self):
        rp = replan(self.problem, {}, from_day=3, seed=1)
        self.assertEqual(rp.overdue_tasks, ["Lab report"])
        deadlines = {t.name: t.deadline_day for t in rp.problem.tasks}
        self.assertEqual(deadlines["Lab report"], 0)
        self.assertEqual(deadlines["Math exam"], 0)  # due today

    def test_importance_is_kept_and_the_rest_rule_adapts(self):
        problem = StudyPlanProblem([Task("A", 45, 6, importance=2.0)])
        rp = replan(problem, {}, from_day=6, seed=0)
        self.assertEqual(rp.problem.tasks[0].importance, 2.0)
        self.assertEqual(rp.problem.config.min_rest_days, 0)

    def test_replanning_is_reproducible(self):
        a = replan(self.problem, {"Lab report": 1}, from_day=1, seed=4)
        b = replan(self.problem, {"Lab report": 1}, from_day=1, seed=4)
        self.assertEqual(a.result.timetable, b.result.timetable)

    def test_bad_input_rejected(self):
        everything = {"Lab report": 3, "Math exam": 4, "AI project": 6, "Reading": 2}
        with self.assertRaises(ValueError):
            replan(self.problem, everything, from_day=1)            # nothing left
        with self.assertRaises(ValueError):
            replan(self.problem, {}, from_day=7)                    # day out of range
        with self.assertRaises(ValueError):
            replan(self.problem, {"Sleep": 1}, from_day=1)          # unknown task
        with self.assertRaises(ValueError):
            replan(self.problem, {"Reading": 3}, from_day=1)        # more than exists
        with self.assertRaises(ValueError):
            replan(self.problem, {}, from_day=6)                    # 15 blocks, 4 slots left


class TestPlannerComparison(unittest.TestCase):
    def test_random_week_is_reproducible_and_in_range(self):
        a, b = random_week(random.Random(3)), random_week(random.Random(3))
        self.assertEqual([(t.name, t.minutes, t.deadline_day) for t in a.tasks],
                         [(t.name, t.minutes, t.deadline_day) for t in b.tasks])
        rng = random.Random(0)
        for _ in range(50):
            week = random_week(rng)
            self.assertGreaterEqual(len(week.tasks), 2)
            self.assertTrue(12 <= len(week.blocks) <= 20)
            self.assertTrue(all(1 <= t.deadline_day <= 6 for t in week.tasks))

    def test_deadlines_feasible(self):
        self.assertFalse(deadlines_feasible(StudyPlanProblem([Task("A", 5 * 45, 0)])))
        self.assertTrue(deadlines_feasible(StudyPlanProblem([Task("A", 4 * 45, 0)])))

    def test_planner_cuts_overload_and_never_double_books(self):
        results = compare_planners(n_weeks=12, seed=0, iterations=800)
        rows = {r["method"]: r for r in summarize_comparison(results)}
        self.assertLess(rows["Simulated annealing"]["overload_minutes"],
                        rows["Equal split"]["overload_minutes"])
        self.assertEqual(rows["Simulated annealing"]["plans_clash_free"], 1.0)
        self.assertEqual(rows["Simulated annealing"]["on_time_when_possible"], 1.0)

    def test_comparison_is_reproducible_and_formats(self):
        a = compare_planners(n_weeks=5, seed=2, iterations=300)
        b = compare_planners(n_weeks=5, seed=2, iterations=300)
        self.assertEqual(a, b)
        table = format_comparison(summarize_comparison(a))
        self.assertEqual(len(table.split("\n")), 4)
        self.assertIn("Equal split", table)

    def test_comparison_rejects_zero_weeks(self):
        with self.assertRaises(ValueError):
            compare_planners(n_weeks=0)


if __name__ == "__main__":
    unittest.main()