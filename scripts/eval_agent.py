"""Run the agent on evals/questions.json and check each answer against its required facts.

Each entry may set: must (all strings required), must_any (one required), must_not, clarify,
max_researchers, context (whether a Context Researcher should run). Every answer must also pass
the citation and number checks and stay within the word limit.

Usage: uv run python scripts/eval_agent.py [question ids...]
"""

import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app  # noqa: E402
from context_panel import AgentContexts, PanelHooks  # noqa: E402

PARALLEL = 3


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
    failures += run.errors
    failures += app.citation_problems(run.report.split("\n\n*")[0], run.tool_calls, case["q"])
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
                  "report": run.report}
        print(f"{'PASS' if not failures else 'FAIL'} {case['id']} ({result['seconds']}s) {failures}", flush=True)
        return result


async def main() -> None:
    cases = json.loads((ROOT / "evals" / "questions.json").read_text())
    if sys.argv[1:]:
        cases = [c for c in cases if c["id"] in sys.argv[1:]]
    gate = asyncio.Semaphore(PARALLEL)
    results = await asyncio.gather(*(run_case(c, gate) for c in cases))
    (ROOT / "evals" / "latest_results.json").write_text(json.dumps(results, indent=1))
    print(f"{sum(r['passed'] for r in results)} of {len(results)} passed")


if __name__ == "__main__":
    asyncio.run(main())
