"""The repeat check: re-verifying is refused, and a real follow-up is allowed. No model calls."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app  # noqa: E402

FINDINGS = "- The Cubs averaged 37,259 fans a game in 2025 (table: team_season_model)."


def run_after(role: str, task: str, findings: str = FINDINGS, failed: bool = False) -> app.Research:
    run = app.Research(hooks=None, question="test")
    run.tasks.append({"role": role, "words": app.task_words(task), "scope": app.task_scope(task),
                      "failed": failed, "findings": findings})
    return run


def test_same_task_again_is_refused():
    run = run_after("viewership", "How many viewers watched the Yankees vs Red Sox game on Fox on June 7, 2025?")
    assert app.repeated_task(run, "viewership", "Find the viewers for the Yankees vs Red Sox Fox game on June 7, 2025.")


def test_new_breakdown_is_allowed():
    run = run_after("attendance", "Find the Chicago Cubs 2025 average attendance.")
    assert not app.repeated_task(run, "attendance", "Find the Chicago Cubs 2025 average attendance by month.")


def test_new_season_is_allowed():
    run = run_after("attendance", "Find the Chicago Cubs 2025 average attendance.")
    assert not app.repeated_task(run, "attendance", "Find the Chicago Cubs 2024 average attendance.")


def test_new_team_is_allowed():
    run = run_after("attendance", "Find the Chicago Cubs 2025 average attendance.")
    assert not app.repeated_task(run, "attendance", "Find the Chicago White Sox 2025 average attendance.")


def test_gap_in_findings_allows_a_retry():
    run = run_after("attendance", "Find the Chicago Cubs 2025 average attendance.",
                    findings="- The query found no rows for the Cubs in 2025.")
    assert not app.repeated_task(run, "attendance", "Find the Chicago Cubs 2025 average attendance.")


def test_failed_researcher_allows_a_retry():
    run = run_after("attendance", "Find the Chicago Cubs 2025 average attendance.", failed=True)
    assert not app.repeated_task(run, "attendance", "Find the Chicago Cubs 2025 average attendance.")


def test_other_role_is_allowed():
    run = run_after("attendance", "Find the Chicago Cubs 2025 average attendance.")
    assert not app.repeated_task(run, "viewership", "Find the Chicago Cubs 2025 average attendance.")


if __name__ == "__main__":
    tests = [f for name, f in dict(globals()).items() if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} of {len(tests)} passed")
