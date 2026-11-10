"""Correct sentences must come through the cleanup steps unchanged. No model calls.

Each case is an answer a careful Writer could give. A step may add a note at the end, but every
original sentence must survive word for word. A failure here means a step is overcorrecting.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app  # noqa: E402
from tests.runner import run_tests  # noqa: E402

GOOD = {
    "fill share then a gap against expected": (
        "Compare the Cubs and White Sox on attendance in 2025.",
        "The Cubs drew 89.9% of capacity, while the White Sox drew 45.0% of capacity (table: games_clean). "
        "Cubs crowds were 0.9% higher than the 36,939 fans expected (table: team_season_model)."),
    "absence already says in the data": (
        "How did the White Sox do on national TV in 2025?",
        "The White Sox had no national TV broadcasts in the 2025 data (table: team_tv_summary)."),
    "temporary park and a drop": (
        "Why did Rays attendance drop in 2025?",
        "The Rays averaged 9,713 fans per home game in 2025 at George M. Steinbrenner Field, down 41.2% "
        "from 16,515 in 2024 (table: team_season_model)."),
    "official season total": (
        "What was Yankees total attendance in 2024?",
        "The Yankees drew 3,309,838 fans at home in 2024 (table: team_season_attendance)."),
    "fill gap in points": (
        "Compare Cubs and White Sox fill rates in 2025.",
        "The Cubs' fill rate was 44.9 percentage points higher than the White Sox's (table: games_clean)."),
    "above expected": (
        "How did the Mets do on attendance in 2025?",
        "The Mets averaged 39,316 fans per home game in 2025, 21.4% above the 32,383 expected "
        "(table: team_season_model)."),
    "one game against capacity": (
        "How did the Rays crowd on May 1, 2025 compare with capacity?",
        "On May 1, 2025, the Rays drew 8,794 fans at George M. Steinbrenner Field, 79.8% of its 11,026 "
        "capacity (table: games_clean)."),
    "record": (
        "How did the Cubs do in 2025?",
        "The Cubs averaged 37,259 fans per home game in 2025 and went 92-70 (table: team_season_model)."),
}


def plain(text: str) -> list[str]:
    """Sentences without citations or spacing differences, for comparing."""
    text = re.sub(r"\s*\((?:tables?|source): [^)]*\)", "", text)
    return [s.strip() for s in app.sentences(text) if s.strip()]


def cleaned(question: str, report: str) -> app.Research:
    run = app.Research(hooks=None, question=question, report=report)
    app.clean_report(run, question)
    return run


def make_test(name: str, question: str, report: str):
    def test():
        run = cleaned(question, report)
        after = plain(run.report)
        lost = [s for s in plain(report) if s not in after]
        assert not lost, f"changed by {run.repairs}:\n    before: {lost}\n    after:  {after}"
    test.__name__ = "test_keeps_" + re.sub(r"\W+", "_", name)
    return test


for _name, (_q, _r) in GOOD.items():
    globals()[make_test(_name, _q, _r).__name__] = make_test(_name, _q, _r)


def test_opening_before_a_heading_stays_its_own_paragraph():
    team = app.data_tools.team_season_facts()
    t = next(x for x in team if x["home_team"] == "Chicago Cubs" and x["season"] == 2025)
    out = app.lead_with_team_result("### Attendance\n\nThe Cubs drew 89.9% of capacity.", t)
    assert all(not line.strip().startswith("#") or line.strip() == "### Attendance" for line in out.split("\n"))
    assert "\n### Attendance" in out and ". ### " not in out, out


if __name__ == "__main__":
    run_tests(globals())
