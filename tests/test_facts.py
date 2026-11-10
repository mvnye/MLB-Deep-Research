"""Facts with IDs: formatting, filling, gaps, phrases, tagging and exact footnotes. No model calls."""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app  # noqa: E402
import citations  # noqa: E402
import facts  # noqa: E402
from tests.runner import run_tests  # noqa: E402

COLUMNS = ["home_team", "season", "avg_attendance", "yoy_change_pct", "residual_pct", "record", "win_pct",
           "fill_pct", "expected_avg_attendance", "official_date", "viewers_total", "coverage_pct", "corr_team_crowd_tv"]
ROWS = [["Tampa Bay Rays", 2025, 9713, -0.4119, -0.05, "77-85", 0.475, 89.94, 10225.4, "2025-05-01", 3094000, 96, 0.512],
        ["Tampa Bay Rays", 2024, 16515, 0.01, 0.02, "80-82", 0.49, 45.0, 16000, "2024-05-01", 1962000, 4, 0.3]]


def make_run(question: str = "How did the Rays do in 2025?") -> app.Research:
    run = app.Research(hooks=None, question=question)
    result = facts.add_query_id(run, json.dumps({"columns": COLUMNS, "rows": ROWS}))
    run.tool_calls.append({"name": "run_sql", "args": {"query": "select * from team_season_model where team_id = 139"},
                           "result": result, "agent": "Attendance Researcher 1"})
    return run


def test_query_ids_count_up():
    run = make_run()
    second = facts.add_query_id(run, json.dumps({"columns": ["x"], "rows": [[1]]}))
    assert json.loads(second)["query_id"] == "Q2" and run.query_ids == ["Q1", "Q2"]
    assert facts.add_query_id(run, json.dumps({"error": "bad"})) == json.dumps({"error": "bad"})


def test_formats_by_column():
    assert facts.format_value("avg_attendance", 39316.4) == "39,316 fans"
    assert facts.format_value("viewers_total", 3094000) == "3,094,000 viewers"
    assert facts.format_value("yoy_change_pct", -0.4119) == "41.2%"
    assert facts.format_value("fill_pct", 89.94) == "89.9% of capacity"
    assert facts.format_value("avg_fill_pct", 102.7) == "102.7% of listed capacity"
    assert facts.format_value("coverage_pct", 96) == "96%"
    assert facts.format_value("corr_team_crowd_tv", 0.512) == "0.51"
    assert facts.format_value("win_pct", 0.58, {"record": "94-68"}) == "94-68"
    assert facts.format_value("official_date", "2025-05-01") == "May 1, 2025"
    assert facts.format_value("appearances", 15) == "15"
    assert facts.format_value("avg_appearances", 5.333) == "5.3"
    assert facts.format_value("viewers_total", None) == "unknown"


def test_fill_writes_values_and_exact_citation():
    run = make_run()
    out = app.fill_report(run, "The Rays averaged {Q1.r1.avg_attendance} a game in 2025, down "
                               "{Q1.r1.yoy_change_pct} from {Q1.r2.avg_attendance} (table: team_season_model).")
    assert out == ("The Rays averaged 9,713 fans a game in 2025, down 41.2% from 16,515 fans "
                   "(table: team_season_model; query: Q1)."), out
    assert run.fact_stats["filled"] == 3 and run.fact_stats["raw_numbers"] == 0


def test_fill_adds_a_citation_when_missing():
    out = app.fill_report(make_run(), "The Rays went {Q1.r1.win_pct} in 2025.")
    assert out == "The Rays went 77-85 in 2025 (table: team_season_model; query: Q1).", out


def test_typed_number_and_unit_are_not_doubled():
    out = app.fill_report(make_run(), "The Rays averaged 9,713 fans {Q1.r1.avg_attendance} a game, and {Q1.r2.avg_attendance} fans came in 2024.")
    assert "9,713 fans a game" in out and "16,515 fans came" in out and "fans fans" not in out, out


def test_gaps_come_from_shown_values():
    out = app.fill_report(make_run(), "Fill was {diff:Q1.r1.fill_pct,Q1.r2.fill_pct} higher, and the crowd gap was "
                                      "{diff:Q1.r1.avg_attendance,Q1.r2.avg_attendance}, a {change:Q1.r2.avg_attendance,Q1.r1.avg_attendance} drop.")
    assert "44.9 percentage points" in out and "6,802 fans" in out and "41.2% drop" in out, out


def test_phrases_fill():
    run = make_run()
    out = app.fill_report(run, "The White Sox had {no_tv:2025}. {no_viewership:2024}.")
    assert out == "The White Sox had no national TV broadcasts in the 2025 data. There is no 2024 viewership in the data.", out
    assert run.fact_stats["phrases"] == 2


def test_team_crowd_tv_phrase_has_no_causal_verb():
    run = make_run("Did smaller crowds go with smaller TV audiences?")
    out = app.fill_report(run, "{team_crowd_tv}")
    assert "0.51" in out and "n = 56" in out and "0.29 to 0.68" in out, out
    assert not re.search(r"\bdriv|\bcaused?\b|\bexplain", out.replace("causes the other", "")), out
    assert out.endswith("the data does not show that one causes the other."), out
    assert "team_crowd_tv" in app.tables_queried(run.tool_calls)
    assert not app.RULED_OUT.search(out)
    # Cleanup keeps it as it is.
    cleaned = app.Research(hooks=None, question=run.question, report=out, tool_calls=run.tool_calls)
    app.clean_report(cleaned, run.question)
    assert out.replace(" (table: team_crowd_tv)", "") in cleaned.report.replace(" (table: team_crowd_tv)", ""), cleaned.report


def test_unknown_ids_are_reported_not_filled():
    run = make_run()
    out = app.fill_report(run, "The Rays averaged {Q7.r1.avg_attendance} and {Q1.r9.avg_attendance} in {nonsense}.")
    assert out.count("{") == 3 and len(run.fact_stats["unresolved"]) == 3
    assert len(app.unfilled_problems(out)) == 3
    assert app.drop_unsourced(out + " The Rays play in Tampa.", run.tool_calls) == "The Rays play in Tampa."


def test_tagging_findings():
    run = make_run()
    findings = ("- The Rays averaged 9,713 fans a game in 2025, down -41.2% from 16,515 (table: team_season_model).\n"
                "- They went 77-85, and 9,713 {Q1.r1.avg_attendance} is already tagged.")
    tagged = facts.tag_findings(findings, run.tool_calls, "Attendance Researcher 1")
    assert "9,713 fans {Q1.r1.avg_attendance} a game" in tagged, tagged
    assert "-41.2% {Q1.r1.yoy_change_pct}" in tagged and "16,515 {Q1.r2.avg_attendance}" in tagged, tagged
    assert tagged.count("{Q1.r1.avg_attendance}") == 2 and "2025 {" not in tagged, tagged
    assert facts.tag_findings(findings, run.tool_calls, "Someone Else") == findings


def test_footnote_uses_the_named_query():
    run = make_run()
    other = facts.add_query_id(run, json.dumps({"columns": ["avg_attendance"], "rows": [[9713]]}))
    run.tool_calls.append({"name": "run_sql", "args": {"query": "select avg_attendance from team_season_model"},
                           "result": other, "agent": "Attendance Researcher 2"})
    text, sources, _ = citations.apply("The Rays averaged 9,713 fans (table: team_season_model; query: Q1).",
                                       run.tool_calls, set(), list(app.data_tools.TABLE_DOCS), app.sentences)
    assert text == "The Rays averaged 9,713 fans[1].", text
    assert "team_id = 139" in sources[0]["query"], sources


def test_query_ids_are_not_tables_in_checks():
    run = make_run()
    problems = app.citation_problems("The Rays averaged 9,713 fans (table: team_season_model; query: Q1).", run.tool_calls)
    assert not [p for p in problems if "not queried" in p], problems


def test_plain_text_passes_through_fill_unchanged():
    text = "The Cubs drew 89.9% of capacity, while the White Sox drew 45.0% of capacity (table: games_clean)."
    assert app.fill_report(make_run(), text) == text


def test_renamed_columns_keep_their_units():
    run = app.Research(hooks=None, question="Tigers 2025")
    result = facts.add_query_id(run, json.dumps({"columns": ["avg_24", "avg_25", "prior_season_avg"], "rows": [[23824, 29796, 23824.0]]}))
    run.tool_calls.append({"name": "run_sql", "args": {"query": "select t24.avg_attendance as avg_24, t25.avg_attendance as avg_25, "
                           "t25.prior_season_avg from team_season_model t24 join team_season_model t25 on true"},
                           "result": result, "agent": "A"})
    out = app.fill_report(run, "They averaged {Q1.r1.avg_25}, up {change:Q1.r1.prior_season_avg,Q1.r1.avg_25} from `{Q1.r1.avg_24}`.")
    assert out.startswith("They averaged 29,796 fans, up 25.1% from 23,824 fans"), out


def test_made_up_ids_in_findings_are_replaced():
    run = make_run()
    findings = "- Rays averaged 9,713 {attendance.r1.avg_attendance} fans, {Q1.row_count} games, 16,515 {Q1.r2.avg_attendance} (table: x)."
    tagged = facts.tag_findings(findings, run.tool_calls, "Attendance Researcher 1")
    assert tagged == "- Rays averaged 9,713 fans {Q1.r1.avg_attendance}, games, 16,515 {Q1.r2.avg_attendance} (table: x).", tagged


def test_minus_form_and_digit_phrase_keys():
    run = make_run()
    out = app.fill_report(run, "The gap was {Q1.r1.avg_attendance - Q1.r2.avg_attendance}. {late_2025}.")
    assert "6,802 fans" in out and "early September" in out and not run.fact_stats["unresolved"], (out, run.fact_stats)


def test_computed_gaps_count_as_sourced():
    run = make_run()
    out = app.fill_report(run, "The gap was {diff:Q1.r1.avg_attendance,Q1.r2.avg_attendance}.")
    assert not app.unverified_numbers(out, run.tool_calls), app.unverified_numbers(out, run.tool_calls)
    assert app.drop_unsourced(out, run.tool_calls) == out


def test_unit_after_more_is_not_doubled():
    out = app.fill_report(make_run(), "The Rays drew {diff:Q1.r2.avg_attendance,Q1.r1.avg_attendance} more fans in 2024.")
    assert "6,802 more fans" in out and "fans more fans" not in out, out


def test_fill_before_its_own_capacity_is_not_doubled():
    out = app.fill_report(make_run(), "The Rays drew 8,794 fans, representing {Q1.r1.fill_pct} of the listed 11,026 capacity.")
    assert "representing 89.9% of the listed 11,026 capacity" in out, out


if __name__ == "__main__":
    run_tests(globals())
