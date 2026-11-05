"""Run the agent on evals/questions.json and check each answer against its required facts.

Each entry may set: must (all strings required), must_any (one required), must_not, clarify,
max_researchers, context (whether a Context Researcher should run). Every answer must also pass
the citation and number checks and stay within the word limit.

Usage: uv run python scripts/eval_agent.py [question ids...]
"""

import asyncio
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app  # noqa: E402
from context_panel import AgentContexts, PanelHooks  # noqa: E402

PARALLEL = 3


def uncited_facts(run: app.Research) -> list[str]:
    """Every sentence with a number needs a footnote, and no inline citation may remain."""
    body = run.report.split("\n\n*")[0]
    problems = ["inline citation left in the answer" for _ in [1] if "(table:" in body or "](http" in body]
    for line in body.split("\n"):
        if line.lstrip().startswith("#"):
            continue  # a heading states no fact
        for sentence in app.sentences(line):
            if run.sources and re.search(r"\d{3}", sentence) and not re.search(r"\[\d+\]", sentence):
                problems.append(f"no footnote: {sentence[:70]}")
    return problems


def check(case: dict, run: app.Research) -> list[str]:
    report, failures = run.report.casefold(), []
    if case.get("clarify", False) != run.needs_clarification:
        failures.append(f"clarify should be {case.get('clarify', False)}")
    failures += [f"missing {m!r}" for m in case.get("must", []) if m.casefold() not in report]
    if case.get("must_any") and not any(m.casefold() in report for m in case["must_any"]):
        failures.append(f"needs one of {case['must_any']}")
    failures += [f"should not say {m!r}" for m in case.get("must_not", []) if m.casefold() in report]
    if "max_researchers" in case and run.researchers > case["max_researchers"]:
        failures.append(f"used {run.researchers} researchers")
    context = any(c["name"] == "research" and c["args"]["role"] == "context" for c in run.tool_calls)
    if "context" in case and context != case["context"]:
        failures.append(f"context ran: {context}")
    failures += [f"unresolved: {role}" for role in run.failed_roles]
    made = run.chart["type"] if run.chart else None
    if "chart" in case and case["chart"] != "any" and made != case["chart"]:
        failures.append(f"chart should be {case['chart']}, got {made}")
    if "team_chart" in case and bool(run.team_chart) != case["team_chart"]:
        failures.append(f"team chart should be {case['team_chart']}")
    if "season_lines" in case and len((run.chart or {"series": []})["series"]) != case["season_lines"]:
        failures.append(f"chart should have {case['season_lines']} lines")
    if "tile" in case and not any(t["value"] == case["tile"] for t in run.tiles):
        failures.append(f"no tile for {case['tile']}: {run.tiles}")
    if case.get("no_tiles") and run.tiles:
        failures.append(f"should have no tiles: {run.tiles}")
    if "box_score" in case and bool(run.box_score) != case["box_score"]:
        failures.append(f"box score should be {case['box_score']}")
    failures += app.citation_problems(run.cited, run.tool_calls, case["q"])
    failures += uncited_facts(run)
    if app.word_count(run.report) > app.BROAD_WORD_LIMIT + 60:
        failures.append(f"{app.word_count(run.report)} words")
    return failures


async def run_case(case: dict, gate: asyncio.Semaphore) -> dict:
    async with gate:
        start = time.time()
        run = app.Research(hooks=PanelHooks(AgentContexts(), lambda event: None), question=case["q"])
        try:
            await app.deep_research(run)
            failures = check(case, run)
        except Exception as e:
            failures = [f"crashed: {type(e).__name__}: {str(e)[:200]}"]
        result = {"id": case["id"], "q": case["q"], "passed": not failures, "failures": failures,
                  "seconds": round(time.time() - start), "researchers": run.researchers,
                  "retries": run.retries, "errors": run.errors, "chart": run.chart, "team_chart": run.team_chart,
                  "tiles": run.tiles, "box_score": run.box_score,
                  "report": run.report, "sources": run.sources, "data_note": run.data_note}
        print(f"{'PASS' if not failures else 'FAIL'} {case['id']} ({result['seconds']}s) {failures}", flush=True)
        return result


async def main() -> None:
    cases = json.loads((ROOT / "evals" / "questions.json").read_text())
    if sys.argv[1:]:
        cases = [c for c in cases if c["id"] in sys.argv[1:]]
    gate = asyncio.Semaphore(PARALLEL)
    results = await asyncio.gather(*(run_case(c, gate) for c in cases))
    (ROOT / "evals" / "latest_results.json").write_text(json.dumps(results, indent=1))
    print(f"{sum(r['passed'] for r in results)} of {len(results)} passed, "
          f"{sum(r['retries'] for r in results)} retries, {sum(len(r['errors']) for r in results)} researcher failures")


if __name__ == "__main__":
    asyncio.run(main())
