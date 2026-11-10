"""Stage 3 checks: they detect and never change text; a sentence still failing after the retry is
dropped unless it holds a required fact, and an answer is never empty. No model calls."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app  # noqa: E402
import data_tools  # noqa: E402
import facts  # noqa: E402
from tests.runner import run_tests  # noqa: E402
from tests.test_cleanup_keeps_good_text import GOOD  # noqa: E402


def run_for(report: str, question: str = "How did the Cubs do in 2025?", tool_calls: list | None = None) -> app.Research:
    return app.Research(hooks=None, question=question, report=report, tool_calls=tool_calls or [])


def fired(report: str, question: str = "How did the Cubs do in 2025?", tool_calls: list | None = None, final: bool = False) -> set[str]:
    return {p["check"] for p in app.check_report(run_for(report, question, tool_calls), question, final=final)}


def test_good_texts_pass_every_check_but_sourcing():
    for name, (q, r) in GOOD.items():
        got = fired(r, q, final=True) - {"untraced"}  # no tool calls here, so no number has a source
        assert not got - ({"both_sides"} if len(app.teams_named(q)) >= 2 else set()), (name, got)


def test_each_sentence_check_fires():
    cases = {
        "first_person": "We found the Cubs drew 89.9% of capacity.",
        "stated_cause": "Cubs attendance fell because the team lost more games.",
        "units": "The Cubs' fill rate was 44.9% higher than the White Sox's fill rate.",
        "derived_seats": "Wrigley Field had 3,000,000 seats across the season.",
        "untraced": "The Cubs drew 12,345 fans.",
    }
    for check, text in cases.items():
        assert check in fired(text), (check, fired(text))
    assert "stated_cause" in fired("Team popularity does not explain the link between crowds and audiences.")
    assert "repeated_numbers" in fired("The Cubs drew 39,316 fans. That 39,316 led the division.")
    assert "repeated_point" in fired("Local RSN viewership is not available in the data. "
                                     "Local RSN viewership is also not available for any team.")
    assert "stated_cause" not in fired("The game has no audience figure in the data because the source gives one number for the window.")
    assert "repeated_numbers" not in fired("The Cubs drew 39,316 fans in 2025. Mix: 7 on TBS in 2025, and 8 in 2026.")
    assert "double_in_data" in fired("In the data, the Cubs had 15 broadcasts in the 2025 data.")


def test_fixed_phrases_are_never_causes():
    fact = data_tools.team_crowd_tv_fact()
    text = (facts.team_crowd_tv_text(fact) + " " + facts.PHRASES["rsn"][0] + ". The Cubs and Mets "
            "averaged more viewers in 2026 than in 2025" + app.NIELSEN_CLAUSE + ".")
    capped = "Demand may have been higher, because the park was full on most nights"
    calls = [{"name": "run_sql", "args": {"query": "select capped_text from x"}, "agent": "A",
              "result": json.dumps({"columns": ["capped_text"], "rows": [[capped + "."]]})}]
    got = app.check_report(run_for(text + " " + capped + ".", tool_calls=calls), "q")
    assert not [p for p in got if p["check"] == "stated_cause"], got


def test_official_total_check():
    swaps = [f for f in data_tools.season_figures() if f["clean_total"] and round(f["clean_total"]) != round(f["official_total"])]
    clean = f"{round(swaps[0]['clean_total']):,}"
    assert "official_total" in fired(f"The team drew {clean} fans at home that season.", final=True)


def test_both_sides_and_length():
    q = "Compare the Cubs and White Sox on attendance in 2025."
    assert "both_sides" in fired("The Cubs drew 89.9% of capacity (table: games_clean).", q)
    assert "both_sides" not in fired("The Cubs drew 89.9% of capacity, while the White Sox drew 45.0%.", q)
    assert "length" in fired("The Cubs played well. " * 120)


def test_drop_keeps_required_facts_and_logs():
    calls = [{"name": "run_sql", "args": {"query": "select avg_attendance from team_season_model"}, "agent": "A",
              "result": json.dumps({"columns": ["avg_attendance"], "rows": [[37259]]})}]
    run = run_for("The Cubs averaged 37,259 fans. We think 37,259 was great. We loved it.", tool_calls=calls)
    run.required = {"37,259": "average"}
    app.drop_failing(run, app.check_report(run, run.question, final=True))
    assert run.report.startswith("The Cubs averaged 37,259 fans."), run.report
    assert "We loved it." not in run.report and "check_drop" in run.repairs, run.report
    assert any(not f["kept"] for f in run.check_failures)


def test_required_fact_only_copy_is_kept():
    run = run_for("We drew 37,259 fans.")
    run.required = {"37,259": "average"}
    app.drop_failing(run, app.check_report(run, run.question, final=True))
    assert run.report == "We drew 37,259 fans." and run.check_failures[0]["kept"], run.check_failures


def test_never_empty():
    run = run_for("We loved it. Our team won.")
    app.drop_failing(run, app.check_report(run, run.question, final=True))
    assert run.report == "We loved it. Our team won." and run.check_failures[-1]["check"] == "never_empty"


def test_checks_do_not_change_text():
    text = "We found the Cubs drew 12,345 fans because they won."
    run = run_for(text)
    app.check_report(run, run.question, final=True)
    assert run.report == text and not run.repairs


def test_nielsen_clause_needs_a_viewer_number():
    calls = [{"name": "run_sql", "args": {"query": "select * from team_tv_summary"}, "agent": "A",
              "result": json.dumps({"columns": ["x"], "rows": [[1]]})}]
    counts = "The Athletics had 3 national TV broadcasts in 2025 and 2 in 2026, fewer viewers overall (table: team_tv_summary)."
    assert app.nielsen_inline(counts, calls) == counts
    numbers = "The Mets averaged 1,500,000 viewers in 2025 and 1,700,000 viewers in 2026 (table: team_tv_summary)."
    assert "Nielsen" in app.nielsen_inline(numbers, calls)


def test_only_mention_of_a_team_is_kept():
    calls = [{"name": "run_sql", "args": {"query": "select home_team, avg_attendance from team_season_model"}, "agent": "A",
              "result": json.dumps({"columns": ["home_team", "avg_attendance"],
                                    "rows": [["New York Yankees", 40803], ["Tampa Bay Rays", 9713], ["Boston Red Sox", 32518]]})}]
    run = run_for("In the AL East, the Yankees averaged 40,803 fans and the Red Sox averaged 32,518 fans. "
                  "The Rays averaged 9,713 fans as a result of their temporary park. "
                  "The Yankees led as a result of their market.",
                  question="How did AL East attendance compare in 2025?", tool_calls=calls)
    run.findings = ["- Yankees 40,803, Red Sox 32,518, Rays 9,713 (table: team_season_model)."]
    app.drop_failing(run, app.check_report(run, run.question, final=True))
    assert "The Rays averaged 9,713 fans" in run.report, run.report
    assert "led as a result of their market" not in run.report, run.report  # the Yankees are named elsewhere
    assert any(f.get("reason") == "only mention of Rays" for f in run.check_failures), run.check_failures


def test_quota_errors_are_retried_then_raised():
    import asyncio
    calls = []

    class Fake(app.QuotaRetryModel):
        pass

    async def flaky(self, *a, **k):
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        return "ok"

    waits, app.QUOTA_WAITS = app.QUOTA_WAITS, (0, 0)
    base = app.LitellmModel.get_response
    try:
        app.LitellmModel.get_response = flaky
        model = Fake(model="vertex_ai/x", api_key="unused")
        assert asyncio.run(model.get_response()) == "ok" and len(calls) == 3
        calls.clear()

        async def always(self, *a, **k):
            calls.append(1)
            raise RuntimeError("429 quota")
        app.LitellmModel.get_response = always
        try:
            asyncio.run(model.get_response())
            raise AssertionError("expected the quota error")
        except RuntimeError as e:
            assert "429" in str(e) and len(calls) == 3, calls
            assert app.friendly_error(e).startswith("The model has hit its usage limit")
    finally:
        app.LitellmModel.get_response, app.QUOTA_WAITS = base, waits


def test_both_sides_needs_each_teams_own_value():
    q = "Compare the Cubs and White Sox on attendance in 2025."
    calls = [{"name": "run_sql", "args": {"query": "select home_team, avg_attendance, attendance_diff from team_season_model"}, "agent": "A",
              "result": json.dumps({"columns": ["home_team", "avg_attendance", "attendance_diff"],
                                    "rows": [["Chicago Cubs", 37259, 19187], ["Chicago White Sox", 18072, None]]})}]
    gap_only = "The Cubs averaged 37,259 fans, 19,187 more per game than the White Sox (table: team_season_model)."
    problems = [p for p in app.check_report(run_for(gap_only, q, calls), q) if p["check"] == "both_sides"]
    assert len(problems) == 1 and "18,072" in problems[0]["message"], problems
    both = "The Cubs averaged 37,259 fans and the White Sox 18,072 (table: team_season_model)."
    assert "both_sides" not in fired(both, q, calls)


def test_paired_chart_wins_over_a_later_bar():
    import charts
    teams = ["New York Yankees", "Boston Red Sox", "Toronto Blue Jays", "Baltimore Orioles"]
    paired = {"name": "run_sql", "args": {"query": "select home_team, avg_attendance, expected_avg_attendance from team_season_model"}, "agent": "A",
              "result": json.dumps({"columns": ["home_team", "avg_attendance", "expected_avg_attendance"],
                                    "rows": [[t, 40000 - 5000 * i, 38000 - 5000 * i] for i, t in enumerate(teams)]})}
    fill = {"name": "run_sql", "args": {"query": "select home_team, avg(fill_pct) as avg_fill_pct from games_clean"}, "agent": "A",
            "result": json.dumps({"columns": ["home_team", "avg_fill_pct"], "rows": [[t, 90.0 - 10 * i] for i, t in enumerate(teams)]})}
    report = "The " + ", ".join(teams) + " all drew crowds."
    q = "Compare actual and expected average attendance for the AL East teams in 2025."
    assert charts.pick_chart(report, q, [paired, fill], ["team_season_model", "games_clean"])["type"] == "paired"
    assert charts.pick_chart(report, "Compare AL East fill rates in 2025.", [paired, fill], ["games_clean"])["type"] == "bar"


if __name__ == "__main__":
    run_tests(globals())
