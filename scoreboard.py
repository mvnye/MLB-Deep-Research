"""Scoreboard tiles and box scores for an answer. Built from checked text and query rows, never by a model.

Tiles are a few key numbers above the answer: only ones a reader needs no help with, so no correlations. A box score is a tight table for an answer that lists
games. Both stay empty when the answer has nothing that fits, so a label can never misdescribe a number.
"""

import json
import re

MAX_TILES = 4
MAX_BOX_ROWS = 15
MIN_BOX_ROWS = 3
THOUSANDS = re.compile(r"\b\d{1,3}(?:,\d{3})+\b")
FILL = re.compile(r"(\d+(?:\.\d)?%) (?:fill|of (?:listed )?capacity)|fill[^.]*?(\d+(?:\.\d)?%)", re.I)
BOX_COLUMNS = {"official_date": "DATE", "game_date": "DATE", "date": "DATE", "away_team": "AWAY", "home_team": "HOME",
               "opponent": "OPP", "network": "NET", "day_night": "D/N", "attendance": "ATT",
               "expected_attendance": "EXP", "fill_pct": "% CAP", "viewers_total": "VIEWERS"}
NUMBER_COLUMNS = ("attendance", "expected_attendance", "viewers_total")


def team_tiles(fact: dict) -> list[dict]:
    tiles = [{"label": "Avg crowd", "value": f"{fact['avg_attendance']:,.0f}"}]
    if fact["residual_pct"] is not None:
        tiles.append({"label": "Vs expected", "value": f"{fact['residual_pct']:+.1%}"})
    tiles.append({"label": "Vs last year", "value": f"{fact['yoy_change_pct']:+.1%}"})
    return tiles


def sentence_tiles(sentence: str) -> list[dict]:
    """A single audience, crowd and fill rate stated in the answer's first sentence."""
    numbers = [m[0] for m in THOUSANDS.finditer(sentence) if not sentence[m.end():].startswith(" seats")]
    lowered = sentence.lower()
    # Two numbers need the sentence to say which is which, and an average or a high is not one crowd.
    if len(numbers) != 1 or re.search(r"season high|averag|peak", lowered):
        return []
    if "viewers" in lowered:
        label = ("Window total" if "window" in lowered else "Main network only" if "main network" in lowered
                 else "Uncertain total" if "uncertain" in lowered else "TV audience")
    elif re.search(r"attendance|crowd|\bdrew\b", lowered):
        label = "Crowd"
    else:
        return []
    tiles = [{"label": label, "value": numbers[0]}]
    fill = FILL.search(sentence)
    if fill and label == "Crowd":
        share = fill[1] or fill[2]
        tiles.append({"label": "Of listed capacity" if float(share[:-1]) > 100 else "Of capacity", "value": share})
    return tiles


def format_cell(column: str, value) -> str:
    if value is None:
        return "-"
    if column in NUMBER_COLUMNS:
        return f"{value:,.0f}"
    if column == "fill_pct":
        return f"{value:.1f}%"
    return str(value)[:10] if "date" in column else str(value)


def box_score(report: str, tool_calls: list[dict], tables_in) -> dict | None:
    """The latest query that lists games, if the answer quotes at least two of its rows."""
    for call in reversed(tool_calls):
        if call["name"] != "run_sql" or call.get("agent") == "Required facts check":
            continue
        result = json.loads(call["result"])
        rows = result.get("rows", [])
        if not MIN_BOX_ROWS <= len(rows) <= MAX_BOX_ROWS:
            continue
        names = [c.lower() for c in result["columns"]]
        keep = [i for i, c in enumerate(names) if c in BOX_COLUMNS]
        kept = {names[i] for i in keep}
        values = kept & set(NUMBER_COLUMNS)
        if not kept & {"official_date", "game_date", "date"} or not values:
            continue
        measure = names.index(sorted(values)[0])
        quoted = [r for r in rows if r[measure] is not None and f"{r[measure]:,.0f}" in report]
        if len(quoted) >= MIN_BOX_ROWS:  # only the games the answer lists: a query may return more
            return {"columns": [BOX_COLUMNS[names[i]] for i in keep],
                    "rows": [[format_cell(names[i], r[i]) for i in keep] for r in quoted],
                    "source": tables_in(call["args"]["query"])}
    return None
