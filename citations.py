"""Turn inline citations into numbered footnotes, using this run's own tool calls.

The checks in app.py run first, on the inline citations. This step replaces '(table: x)' and
'([Title](url))' with [1] markers, one number per distinct query or article, reused across
sentences. A citation to a table or article the run did not use gets no number.
"""

import json
import re

CITATION = re.compile(r"\s*\((?=\s*(?:\[|tables?:))(?:[^()]|\([^()]*\))*\)")
LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
TABLE_LIST = re.compile(r"tables?:\s*([\w, ]+)")  # 'table: a; table: b' gives one match each
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
NO_FOOTNOTE = re.compile(r"^(we cannot|the data cannot|our data|the data has no)", re.I)
MAX_ROWS_SHOWN = 3

ATTENDANCE_TABLES = {"games", "games_clean", "game_model", "team_season_model", "team_season_attendance",
                     "model_coefficients", "model_fit", "teams", "team_aliases"}
VIEWERSHIP_TABLES = {"viewership_games", "game_viewership_clean", "fox_windows", "fox_window_games",
                     "crowd_tv_residual", "crowd_tv_headline", "game_tv_lookup", "team_tv_summary",
                     "team_crowd_tv", "network_audience"}
MARKET_TABLES = {"team_markets"}
MLB_NOTE = "MLB Stats API (statsapi.mlb.com)"
TV_NOTE = ("TV Media Blog MLB viewership archives "
           "([2025](https://tvmediablog.substack.com/p/2025-mlb-season-viewership-database), "
           "[2026](https://tvmediablog.substack.com/p/2026-mlb-viewership-archive))")
MARKET_NOTE = "US Census Bureau and Statistics Canada population estimates"


def query_calls(tool_calls: list[dict]) -> list[dict]:
    return [c for c in tool_calls if c["name"] == "run_sql" and '"error"' not in c["result"][:12]]


def numbers_in(text: str) -> set[str]:
    return {n.replace(",", "") for n in NUMBER.findall(text) if len(n.replace(",", "")) >= 3}


def holds(text: str, number: str) -> bool:
    """Does this result hold the number, allowing for rounding and percent form?"""
    value = float(number)
    step = 0.5 * 10 ** -len(number.split(".")[1]) if "." in number else (1.0 if value >= 1000 else 0.5)
    for found in NUMBER.findall(text):
        raw = float(found.replace(",", ""))
        if abs(raw - value) <= step or abs(raw * 100 - value) <= step:
            return True
    return False


def result_rows(call: dict) -> tuple[list[str], list[list]]:
    result = json.loads(call["result"])
    if "columns" in result:
        return result["columns"], result.get("rows", [])
    return list(result), [list(result.values())]  # a single fact logged by the app


def matching_rows(call: dict, numbers: set[str]) -> tuple[list[str], list[list]]:
    """The rows that hold this sentence's numbers, or the first few."""
    columns, rows = result_rows(call)
    hits = [r for r in rows if any(holds(json.dumps(r, default=str), n) for n in numbers)]
    return columns, (hits or rows)[:MAX_ROWS_SHOWN]


def best_query(calls: list[dict], table: str | None, numbers: set[str]) -> dict | None:
    """The query behind a fact: the one holding most of its numbers, preferring the latest."""
    options = [c for c in calls if table is None or re.search(rf"\b{table}\b", c["args"]["query"], re.I)]
    if not options:
        return None
    score = lambda c: sum(holds(c["result"], n) for n in numbers)
    best = max(enumerate(options), key=lambda p: (score(p[1]), p[0]))[1]
    return best if table is not None or score(best) else None


class Footnotes:
    def __init__(self, tool_calls: list[dict], read_urls: set[str], known_tables: list[str]):
        self.calls = query_calls(tool_calls)
        self.read_urls = read_urls
        self.known = known_tables
        self.index: dict[str, int] = {}
        self.sources: list[dict] = []

    def number(self, key: str, make) -> int:
        if key not in self.index:
            self.index[key] = len(self.sources) + 1
            self.sources.append({"n": self.index[key], **make()})
        return self.index[key]

    def for_query(self, call: dict, numbers: set[str]) -> int:
        def make() -> dict:
            columns, rows = matching_rows(call, numbers)
            tables = [t for t in self.known if re.search(rf"\b(?:from|join)\s+{t}\b", call["args"]["query"], re.I)]
            return {"kind": "query", "tables": tables, "query": " ".join(call["args"]["query"].split()),
                    "columns": columns, "rows": rows}
        return self.number("q:" + call["args"]["query"], make)

    def for_article(self, title: str, url: str) -> int:
        return self.number("a:" + url, lambda: {"kind": "article", "title": title, "url": url})

    def citation_markers(self, citation: str, numbers: set[str]) -> list[int]:
        found = []
        for title, url in LINK.findall(citation):
            if url.split("#")[0] in self.read_urls:
                found.append(self.for_article(title, url))
        for group in TABLE_LIST.findall(citation):
            for table in re.split(r",\s*", group.strip()):
                call = best_query(self.calls, table.strip(), numbers)
                if call:
                    found.append(self.for_query(call, numbers))
        return found


def place(sentence: str, markers: list[int]) -> str:
    tags = "".join(f"[{n}]" for n in dict.fromkeys(markers))
    stripped = sentence.rstrip()
    match = re.search(r"[.!?]+$", stripped)
    return stripped[:match.start()] + tags + match[0] if match else stripped + tags


def apply(report: str, tool_calls: list[dict], read_urls: set[str], known_tables: list[str],
          split_sentences) -> tuple[str, list[dict], str]:
    """Return the report with footnote markers, its sources, and the 'Data sources' note."""
    notes = Footnotes(tool_calls, read_urls, known_tables)
    lines = []
    for line in report.split("\n"):
        if not line.strip() or line.lstrip().startswith("#"):
            lines.append(line)
            continue
        parts: list[tuple[str, list[int]]] = []
        for sentence in split_sentences(line):
            numbers = numbers_in(CITATION.sub("", sentence))
            cited = list(CITATION.finditer(sentence))
            markers = [m for c in cited for m in notes.citation_markers(c[0], numbers)]
            if cited and not markers:
                continue  # its only citations were to sources this run never used
            text = LINK.sub(lambda m: m[1], CITATION.sub("", sentence))
            if not markers and numbers:  # an uncited fact still gets the query that holds its numbers
                call = best_query(notes.calls, None, numbers)
                if call:
                    markers = [notes.for_query(call, numbers)]
            parts.append((text, markers))
        # Numbers were checked against the data earlier, so an unmatched sentence shares its neighbour's.
        for i, (text, markers) in enumerate(parts):
            if not markers and not NO_FOOTNOTE.search(text.strip()):
                nearby = parts[i + 1:] + parts[:i][::-1]
                parts[i] = (text, next((m for _, m in nearby if m), []))
        if parts:
            lines.append(" ".join(place(t, m) if m else t for t, m in parts))
    return "\n".join(lines), notes.sources, data_note(notes.sources)


def data_note(sources: list[dict]) -> str:
    tables = {t for s in sources for t in s.get("tables", [])}
    names = []
    if tables & ATTENDANCE_TABLES:
        names.append(MLB_NOTE)
    if tables & VIEWERSHIP_TABLES:
        names.append(TV_NOTE)
    if tables & MARKET_TABLES:
        names.append(MARKET_NOTE)
    if any(s["kind"] == "article" for s in sources):
        names.append("Wikipedia")
    return f"Data sources: {'; '.join(names)}." if names else ""


def to_markdown(report: str, sources: list[dict], note: str) -> str:
    """The answer with its Sources list at the end, for the /chat JSON."""
    if not sources:
        return f"{report}\n\n{note}".strip()
    lines = ["**Sources**", ""]
    for s in sources:
        if s["kind"] == "article":
            lines.append(f"{s['n']}. [{s['title']}]({s['url']})")
        else:
            rows = "; ".join(", ".join(f"{c}={v}" for c, v in zip(s["columns"], r)) for r in s["rows"])
            lines.append(f"{s['n']}. {', '.join(s['tables'])}: `{s['query']}`" + (f" Row: {rows}" if rows else ""))
    return "\n".join([report, "", *lines, "", note]).strip()
