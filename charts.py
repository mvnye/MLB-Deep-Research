"""Choose at most one chart for an answer, from the rows a researcher's SQL returned.

The chart follows the shape of the result:
  line    a value over seasons or months, 4 or more points
  bar     one value across 4 or more items
  paired  actual and expected across items
  none    single values, 2 to 3 values, or an answer that says the data cannot answer

Charts are built only from query results, never from the Writer's text.
"""

import json
import re

MIN_ITEMS = 4
MIN_PAIRED_ITEMS = 3
MAX_ITEMS = 31
MAX_LABEL_CHARS = 40
MONTHS = {3: "March", 4: "April", 5: "May", 6: "June", 7: "July", 8: "August", 9: "September", 10: "October"}
TIME_COLUMN = re.compile(r"(^|_)(month|season|year)(_name|_num|_number)?$", re.I)
ID_COLUMN = re.compile(r"(_id|_pk|^n|^rank|^row_number|count|games|_games|num_|^number)", re.I)
CHANGE_COLUMN = re.compile(r"diff|delta|prev|prior|lag", re.I)  # the step between rows, not the value
EXPECTED_COLUMN = re.compile(r"expected|predict", re.I)
ACTUAL_COLUMN = re.compile(r"actual|attendance|crowd", re.I)
MEASURE_ORDER = [r"(avg|average|mean).*(attendance|crowd)", r"attendance|crowd", r"viewers|audience", r"fill", r"pct|percent|change", r"avg|average|mean|total|sum"]
EXPECTED_QUESTION = re.compile(r"expected|predict|over-?perform|under-?perform", re.I)
NO_ANSWER = re.compile(r"cannot|can't|not in our data|no number|no audience|does not cover|outside", re.I)
LABELS = {"home_team": "Team", "away_team": "Opponent", "team": "Team", "network": "Network", "weekday": "Day",
          "month": "Month", "month_name": "Month", "season": "Season", "day_night": "Time of day",
          "label": "Factor", "effect_pct": "Change in attendance"}


def words(column: str) -> str:
    column = re.sub(r"^current_", "", column.lower())
    name = re.sub(r"\bavg\b", "average", column.replace("_", " "))
    name = re.sub(r"\bpct\b", "percent", name)
    return LABELS.get(column.lower(), re.sub(r"\s+", " ", name).strip().capitalize())


def is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def tables_in(query: str, known: list[str]) -> list[str]:
    return [t for t in known if re.search(rf"\b(?:from|join)\s+{t}\b", query, re.I)]


def unit_of(column: str, values: list[float]) -> tuple[str, float]:
    """The unit for a column, and what to multiply by to show it."""
    name = column.lower()
    if re.search(r"pct|percent|fill|change|rate|residual", name):
        fraction = max(abs(v) for v in values) <= 1.5 and "fill_pct" not in name
        return "%", 100.0 if fraction else 1.0
    if re.search(r"viewers|audience", name):
        return "viewers", 1.0
    if re.search(r"attendance|crowd", name):
        return "fans", 1.0
    return "", 1.0


def split_columns(columns: list[str], rows: list[list]) -> tuple[list[int], list[int]]:
    """Label columns and measure columns, leaving out IDs, counts and columns with one value."""
    labels, measures = [], []
    for i, name in enumerate(columns):
        values = [r[i] for r in rows]
        varies = len({str(v) for v in values}) > 1
        if TIME_COLUMN.search(name) and varies:
            labels.append(i)
        elif all(is_number(v) or v is None for v in values) and any(is_number(v) for v in values):
            if not ID_COLUMN.search(name) and varies:
                measures.append(i)
        elif all(isinstance(v, str) and len(v) <= MAX_LABEL_CHARS for v in values) and varies:
            labels.append(i)  # longer text is a sentence about the row, not its name
    return labels, measures


def main_measure(columns: list[str], measures: list[int]) -> int:
    for pattern in MEASURE_ORDER:
        for i in measures:
            if re.search(pattern, columns[i], re.I):
                return i
    return measures[0]


def label_text(column: str, value) -> str:
    if re.search("month", column, re.I):
        match = re.fullmatch(r"(?:\d{4}-)?(\d{1,2})(?:\.0)?", str(value))  # 4, 2025-04
        if match:
            return MONTHS.get(int(match[1]), str(value))
    return str(value)


def build(kind: str, columns: list[str], rows: list[list], label: int, measures: list[int], source: list[str]) -> dict:
    first = [r[measures[0]] for r in rows]
    unit, scale = unit_of(columns[measures[0]], first)
    name = words(columns[measures[0]])
    title = "Actual and expected attendance" if kind == "paired" else name
    return {
        "type": kind,
        "title": f"{title} by {words(columns[label]).lower()}",
        "x_label": words(columns[label]),
        "y_label": "Average attendance (fans per game)" if kind == "paired"
                   else name if not unit or unit in name.lower() else f"{name} ({unit})",
        "unit": unit,
        "labels": [label_text(columns[label], r[label]) for r in rows],
        "series": [{"name": "Actual" if kind == "paired" and n == 0 else "Expected" if kind == "paired" else words(columns[m]),
                    "values": [round(r[m] * scale, 1) for r in rows]} for n, m in enumerate(measures)],
        "source": source,
    }


def chart_from_rows(columns: list[str], rows: list[list], source: list[str], wants_expected: bool = True) -> dict | None:
    if not MIN_PAIRED_ITEMS <= len(rows) <= MAX_ITEMS:
        return None
    if any(re.search(r"_\d+$", c) for c in columns):
        return None  # network_1, network_2: a pairwise comparison, not one value per item
    labels, measures = split_columns(columns, rows)
    time_labels = [i for i in labels if TIME_COLUMN.search(columns[i])]
    if len(labels) > 1 and len(time_labels) >= 1:
        labels = time_labels[:1]  # month and month_name are one label said twice
    if len(labels) != 1 or not measures:
        return None  # several label columns means game-level detail, not one comparison
    label = labels[0]
    if len({str(r[label]) for r in rows}) < len(rows):
        return None  # a repeated label means a second split, such as day within season
    expected = [i for i in measures if EXPECTED_COLUMN.search(columns[i])]
    actual = [i for i in measures if ACTUAL_COLUMN.search(columns[i]) and i not in expected]
    if expected and actual and wants_expected:
        known = [r for r in rows if r[expected[0]] is not None and r[actual[0]] is not None]
        if len(known) >= MIN_PAIRED_ITEMS:
            return build("paired", columns, known, label, [actual[0], expected[0]], source)
    if len(rows) < MIN_ITEMS:
        return None
    measures = [i for i in measures if i not in expected and all(r[i] is not None for r in rows)]
    if not measures:
        return None
    measures = [i for i in measures if not CHANGE_COLUMN.search(columns[i])] or measures
    measure = main_measure(columns, measures)
    if TIME_COLUMN.search(columns[label]):
        ordered = sorted(rows, key=lambda r: r[label]) if all(is_number(r[label]) for r in rows) else rows
        return build("line", columns, ordered, label, [measure], source)
    return build("bar", columns, sorted(rows, key=lambda r: -r[measure]), label, [measure], source)


def relevant(chart: dict, report: str) -> bool:
    """A chart has to be about what the answer says: two of its items or values must appear."""
    text = report.lower()
    named = sum(1 for label in chart["labels"] if label.lower() in text)
    values = sum(1 for v in chart["series"][0]["values"] if f"{v:,.0f}" in report or f"{v:.1f}%" in report)
    return max(named, values) >= 2


def pick_chart(report: str, question: str, tool_calls: list[dict], tables: list[str]) -> dict | None:
    """The chart for an answer, from the last query whose result fits, or None."""
    if NO_ANSWER.search(report[:200]):
        return None
    wants_expected = bool(EXPECTED_QUESTION.search(question))
    for call in reversed(tool_calls):
        if call["name"] != "run_sql" or call.get("agent") == "Required facts check":
            continue
        result = json.loads(call["result"])
        if "rows" not in result or result.get("truncated"):
            continue
        chart = chart_from_rows(result["columns"], result["rows"], tables_in(call["args"]["query"], tables), wants_expected)
        if chart and relevant(chart, report):
            return chart
    return None
