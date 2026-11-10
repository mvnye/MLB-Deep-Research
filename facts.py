"""Facts with IDs: every run_sql result becomes citable values that code formats.

Each researcher query gets an ID (Q1, Q2, ...). A value is cited as {Q3.r1.avg_attendance}: the
query, the row (from 1) and the column. The Writer puts IDs where numbers go, and code fills in the
value, formatted here: thousands separators, rounding, units, percents, records. Differences and
percent changes between two facts are worked out here from the values as shown, so a gap always
matches the numbers beside it. Absence and limit statements are ready-made phrases, also in braces.

No model calls. Read by app.py and tools.py.
"""

import json
import re
from datetime import date, datetime

QUERY_ID = re.compile(r"^Q\d+$")
REF = r"Q\d+\.r\d+\.[A-Za-z_]\w*"
FACT = re.compile(r"\{(Q\d+)\.r(\d+)\.([A-Za-z_]\w*)\}")
DERIVED = re.compile(rf"\{{(diff|change):\s*({REF})\s*,\s*({REF})\s*\}}")
MINUS = re.compile(rf"\{{\s*({REF})\s*-\s*({REF})\s*\}}")  # {A - B} is read as {diff:A,B}
PHRASE = re.compile(r"\{([a-z][a-z0-9_]*)(?::(\d{4}))?\}")
ANY = re.compile(r"\{[A-Za-z_][^{}\n]{0,80}\}")
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?%?")
# The unit already follows the ID, maybe after one word: "{ID} more fans", "{ID} total fans", "{ID} home fans".
UNIT_AFTER = re.compile(r"^\s+(?:(?!(?:than|and|or|of|to|per|a|an|the|in|for|from|at|on|with)\s)[a-z][a-z-]*\s+)?"
                        r"(fans|viewers|people|of capacity|of listed capacity|percentage points?)\b")
SHOWN_BEFORE = re.compile(r"(\d[\d,]*(?:\.\d+)?%?)(?:\s+(?:fans|viewers))?\s*$")
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]


# --- Ready-made phrases for absence and limit statements ---

def team_crowd_tv_text(fact: dict) -> str:
    """The team-level crowds vs TV finding, in fixed words: it can never gain a causal verb."""
    return (f"Across team-seasons, teams with higher average home attendance also tended to have larger "
            f"average national TV audiences (correlation {fact['corr_team_crowd_tv']:.2f}, n = {fact['n']}, "
            f"range {fact['ci_low']:.2f} to {fact['ci_high']:.2f}) (table: team_crowd_tv). This is consistent "
            "with team-level factors such as popularity and market size, but the data does not show that "
            "one causes the other.")


TEAM_CROWD_TV_END = "the data does not show that one causes the other."

# key -> (text, needs a season). The Writer adds the sentence around them.
PHRASES = {
    "no_tv": ("no national TV broadcasts in the {season} data", True),
    "no_viewership": ("There is no {season} viewership in the data", True),
    "no_season": ("The data covers the 2024 to 2026 regular seasons, so {season} is not in it", True),
    "postseason": ("The data covers the regular season only, so the postseason is not in it", False),
    "rsn": ("Local RSN viewership is not public, so it is not in the data", False),
    "peacock": ("Peacock-only games have no national TV audience in the data", False),
    "late_2025": ("The 2025 viewership tables end in early September, so later 2025 games are not in the data", False),
    "window": ("the data has no audience for this game alone, only for the whole Fox window", False),
    "unknown_audience": ("the audience is unknown", False),
}
PHRASE_HELP = "\n".join(
    f"  {{{k}{':<season>' if needs else ''}}} -> \"{text.replace('{season}', '<season>')}\""
    for k, (text, needs) in PHRASES.items())


# --- Query IDs ---

def add_query_id(run, result: str) -> str:
    """Give a successful result its query ID. Called in the event loop, so IDs never collide."""
    try:
        data = json.loads(result)
    except ValueError:
        return result
    if "columns" not in data or "query_id" in data:
        return result
    ids = getattr(run, "query_ids", None)
    if ids is None:
        return result
    qid = f"Q{len(ids) + 1}"
    ids.append(qid)
    return json.dumps({"query_id": qid, "cite_as": f"{{{qid}.r<row from 1>.<column>}}", **data}, default=str)


class FactBook:
    """The run's query results by ID, for looking up and formatting facts."""

    def __init__(self, tool_calls: list[dict]):
        self.results: dict[str, dict] = {}
        for call in tool_calls:
            if call.get("name") != "run_sql":
                continue
            try:
                data = json.loads(call["result"])
            except (ValueError, TypeError):
                continue
            qid = data.get("query_id")
            if qid and qid not in self.results:
                query = call["args"].get("query", "")
                self.results[qid] = {"columns": data["columns"], "rows": data.get("rows", []),
                                     "query": query, "agent": call.get("agent", ""), "sources": aliases(query)}

    def lookup(self, qid: str, row: int, column: str) -> tuple[object, str, dict] | str:
        """(value, column, row as a dict), or why the ID does not resolve."""
        if qid not in self.results:
            return f"{qid} is not a query in this research"
        result = self.results[qid]
        rows = result["rows"]
        if not 1 <= row <= len(rows):
            return f"{qid} has {len(rows)} rows, so r{row} does not exist"
        names = {c.lower(): i for i, c in enumerate(result["columns"])}
        if column.lower() not in names:
            return f"{qid} has no column {column}; its columns are {', '.join(result['columns'])}"
        if column.lower().endswith("_text"):
            return (f"{column} is a finished sentence, not a value: copy that sentence word for word as its own "
                    "sentence, or use the number columns")
        cells = dict(zip((c.lower() for c in result["columns"]), rows[row - 1]))
        value, name = rows[row - 1][names[column.lower()]], column.lower()
        # 'avg_attendance AS avg_24' formats as attendance: the alias alone says nothing.
        source = result["sources"].get(name)
        if source and kind(name, value) == "number" and kind(source, value) != "number":
            name = source
        return value, name, cells

    def tables(self, qid: str, known: list[str]) -> list[str]:
        query = self.results.get(qid, {}).get("query", "")
        return [t for t in known if re.search(rf"\b(?:from|join)\s+{t}\b", query, re.I)]


def aliases(query: str) -> dict[str, str]:
    """alias -> the column it renames, for 'x.avg_attendance AS avg_24' and 'avg(attendance) AS a'."""
    found = {}
    for m in re.finditer(r"(?:\b(?:\w+\.)?(\w+)\s*\)?|\(\s*(?:\w+\.)?(\w+)\s*\))\s+as\s+(\w+)", query, re.I):
        source = (m[1] or m[2] or "").lower()
        if source and source not in ("select", "end", "distinct"):
            found.setdefault(m[3].lower(), source)
    return found


# --- Formatting ---

DIFFISH = re.compile(r"diff|gap|delta|change|minus|vs_", re.I)


def kind(column: str, value) -> str:
    """What a value is, from its column name and type. Decides its rounding and unit."""
    c = column.lower()
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return "text"
    if isinstance(value, (date, datetime)) or (isinstance(value, str) and re.fullmatch(r"\d{4}-\d\d-\d\d", value)):
        return "date"
    if c == "record" or (isinstance(value, str) and re.fullmatch(r"\d{1,3}-\d{1,3}", value)):
        return "record"
    if not isinstance(value, (int, float)):
        return "text"
    if re.search(r"corr|ci_low|ci_high|^r$|r_squared|p_value|coef", c):
        return "stat"
    if re.search(r"win_pct|win_rate|winning", c):
        return "win"
    if re.search(r"season|year", c) and float(value).is_integer() and 1900 <= value <= 2100:
        return "year"
    if "coverage" in c:
        return "coverage"
    if re.search(r"fill|pct_of_capacity|capacity_pct", c):
        return "fill_points" if DIFFISH.search(c) else "fill"
    if re.search(r"pct|percent|share|_rate|yoy|effect", c):
        return "fraction" if abs(value) <= 1.5 else "percent"
    if re.search(r"viewer|audience|window_total", c):
        return "viewers"
    if re.search(r"attendance|crowd|fans|expected|home_total|home_average|official|season_avg|year_avg|^high$|^low$", c):
        return "fans"
    return "number"


FILL_FRACTION_MAX = 2.0  # no park fills to 2% of capacity, so a fill this small is a fraction


def shown(k: str, value) -> float | None:
    """The value as the reader sees it, rounded, for working out gaps between shown values."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    v = abs(value) if k in ("fraction", "percent", "fill_points") or k in ("fans", "viewers", "number") else value
    if k == "fraction" or (k == "fill" and abs(v) <= FILL_FRACTION_MAX):  # attendance / capacity, not yet x100
        return round(v * 100, 1)
    if k in ("percent", "fill", "fill_points"):
        return round(v, 1)
    if k == "coverage":
        return round(v)
    if k == "stat":
        return round(v, 2)
    if k in ("fans", "viewers") or abs(v) >= 100:
        return round(v)
    return v if float(v).is_integer() else round(v, 1)


def plain_number(v: float) -> str:
    return f"{int(v):,}" if float(v).is_integer() else f"{v:,.1f}"


def format_value(column: str, value, cells: dict | None = None) -> str:
    """One value as text, with its unit: '39,316 fans', '21.4%', '89.9% of capacity', '94-68'."""
    k = kind(column, value)
    cells = cells or {}
    if k == "unknown":
        return "unknown"
    if k == "date":
        d = value if isinstance(value, (date, datetime)) else date.fromisoformat(value)
        return f"{MONTHS[d.month - 1]} {d.day}, {d.year}"
    if k in ("record", "text"):
        return str(value)
    if k == "win":
        if cells.get("record"):
            return str(cells["record"])
        if cells.get("wins") is not None and cells.get("losses") is not None:
            return f"{cells['wins']}-{cells['losses']}"
        return f"{value:.3f}".lstrip("0")
    if k == "year":
        return str(int(value))
    s = shown(k, value)
    if k == "stat":
        return f"{s:.2f}"
    if k in ("fraction", "percent"):
        return f"{s:.1f}%"
    if k == "fill":
        return f"{s:.1f}% of {'listed ' if s > 100 else ''}capacity"
    if k == "fill_points":
        return f"{s:.1f} percentage points"
    if k == "coverage":
        return f"{s}%"
    if k == "fans":
        return f"{int(s):,} fans"
    if k == "viewers":
        return f"{int(s):,} viewers"
    return plain_number(s)


def derived(op: str, a: tuple, b: tuple) -> str:
    """A gap or percent change between two facts, from their values as shown."""
    (va, ca, _), (vb, cb, _) = a, b
    ka, kb = kind(ca, va), kind(cb, vb)
    sa, sb = shown(ka, va), shown(kb, vb)
    if sa is None or sb is None:
        return ""
    group = lambda k: "share" if k in ("fill", "percent", "fraction", "fill_points", "coverage") else k
    if group(ka) != group(kb) and "number" not in (ka, kb):
        return ""
    if ka == "number":
        ka = kb  # a plain count takes the unit of the fact it is compared with
    if op == "change":
        if group(ka) == "share" or sa == 0:
            return ""
        return f"{abs(sb - sa) / abs(sa) * 100:.1f}%"
    gap = abs(sa - sb)
    if group(ka) == "share":
        return f"{gap:.1f} percentage points"
    if ka == "stat":
        return f"{gap:.2f}"
    unit = {"fans": " fans", "viewers": " viewers"}.get(ka, "")
    return f"{round(gap):,}{unit}" if gap >= 100 or ka in ("fans", "viewers") else plain_number(round(gap, 1))


# --- Filling the Writer's text ---

def _same_number(text: str, value: str) -> bool:
    a = [n.replace(",", "") for n in NUMBER.findall(text)]
    b = [n.replace(",", "") for n in NUMBER.findall(value)]
    return bool(a and b) and a[0] == b[0]


def _fit(filled: str, before: str, after: str) -> tuple[str, str]:
    """Fit a filled value to its sentence. Returns (text before, value) with a number the Writer
    also typed just before the ID removed, and a unit dropped if the sentence already has it."""
    typed = SHOWN_BEFORE.search(before)
    if typed and _same_number(typed[1], filled):
        before = before[:typed.start()]
    unit = UNIT_AFTER.match(after)
    if unit:
        filled = re.sub(r"\s+(fans|viewers|of (?:listed )?capacity|percentage points)$", "", filled)
    elif filled.endswith("capacity") and (re.search(r"fill|capacity", before[-30:], re.I)
                                          or re.match(r"\s+of\b", after)):  # "of the listed 11,026 capacity"
        filled = re.sub(r" of (?:listed )?capacity$", "", filled)
    return before, filled


class Filled:
    def __init__(self):
        self.text = ""
        self.facts = 0  # values filled from fact IDs
        self.derived = 0  # gaps and changes worked out here
        self.phrases = 0  # ready-made phrases used
        self.unresolved: list[str] = []
        self.queries: list[str] = []  # the query IDs used, in order
        self.computed: list[dict] = []  # each gap or change worked out here, for the tool log


SENTENCE_PHRASES = {k for k, (text, _) in PHRASES.items() if text[0].isupper()} | {"team_crowd_tv"}
LEAD_OK = re.compile(r"^[-*>#\s]*$|[:.!?]\s*$")  # nothing before it, a label such as "TV:", or a finished sentence
JOINER = re.compile(r"^[\s,;:]*(?:(?:but|and|so|while|although|though|because|as|yet)\b\s*)?", re.I)
TRAILER = re.compile(r"[\s,;:]*(?:\b(?:but|and|so|while|although|though|because|as|yet)\s*)?$", re.I)
ONLY_END = re.compile(r"^\s*[.!?]?\s*(?:\(tables?:[^)]*\)\s*)?[.!?]?\s*$")


def phrase_alone(sentence: str) -> str:
    """A ready-made phrase that is a whole sentence ('There is no 2024 viewership in the data') stands
    alone: never after other words or before more of the sentence. Words before it are kept as their own
    sentence only if they hold a number ('The Athletics have' is dropped); words after it become the
    next sentence."""
    for m in PHRASE.finditer(sentence):
        if m[1] not in SENTENCE_PHRASES:
            continue
        before, after = sentence[:m.start()], sentence[m.end():]
        lead_ok = bool(LEAD_OK.search(before))
        end_ok = bool(ONLY_END.match(after) or re.match(r"\s*[.!?]\s+\S", after))
        if lead_ok and end_ok:
            continue
        lead = "" if lead_ok else TRAILER.sub("", before).strip()
        lead = (lead.rstrip(".") + ". ") if re.search(r"\d", lead) else ("" if not lead_ok else before)
        if end_ok:
            tail = after
        else:
            rest = JOINER.sub("", after, count=1)
            tail = ". " + rest[:1].upper() + rest[1:]
        return phrase_alone(lead + m[0] + tail)
    return sentence


def fill_sentence(sentence: str, book: FactBook, phrase_text, out: Filled) -> tuple[str, list[str]]:
    """Fill one sentence's IDs and phrases. Returns the sentence and the query IDs it used."""
    sentence = re.sub(r"`(" + ANY.pattern + r")`", r"\1", sentence)  # an ID is not code
    sentence = phrase_alone(sentence)
    used: list[str] = []
    pieces, pos = [], 0
    for m in ANY.finditer(sentence):
        before, after, token = sentence[pos:m.start()], sentence[m.end():], m[0]
        value = None
        if fm := FACT.fullmatch(token):
            found = book.lookup(fm[1], int(fm[2]), fm[3])
            if isinstance(found, str):
                out.unresolved.append(f"{token}: {found}")
            else:
                value = format_value(found[1], found[0], found[2])
                used.append(fm[1])
                out.facts += 1
        elif mm := MINUS.fullmatch(token):
            token = f"{{diff:{mm[1]},{mm[2]}}}"
        if value is not None or FACT.fullmatch(token):
            pass
        elif dm := DERIVED.fullmatch(token):
            parts = [FACT.fullmatch("{" + r + "}") for r in (dm[2], dm[3])]
            looked = [book.lookup(p[1], int(p[2]), p[3]) for p in parts]
            bad = [x for x in looked if isinstance(x, str)]
            value = None if bad else derived(dm[1], looked[0], looked[1]) or None
            if value:
                used += [parts[0][1], parts[1][1]]
                out.derived += 1
                out.computed.append({"op": dm[1], "facts": [dm[2], dm[3]], "value": value})
            else:
                out.unresolved.append(f"{token}: " + (bad[0] if bad else "these two facts cannot be compared that way"))
        elif pm := PHRASE.fullmatch(token):
            value = phrase_text(pm[1], pm[2])
            if value is None:
                out.unresolved.append(f"{token}: not a ready-made phrase")
            else:
                out.phrases += 1
        else:
            out.unresolved.append(f"{token}: not a fact ID or a ready-made phrase")
        if value is None:
            pieces.append(before + token)
        else:
            before, value = _fit(value, before, after)
            pieces.append(before + value)
        pos = m.end()
    pieces.append(sentence[pos:])
    text = "".join(pieces)
    text = re.sub(r"(the other\.)\.", r"\1", text)  # the team-level phrase ends its own sentence
    return text, list(dict.fromkeys(used))


def cite(sentence: str, qids: list[str], book: FactBook, known: list[str]) -> str:
    """Name the exact queries behind a sentence's facts in its citation, so its footnote is exact."""
    if not qids:
        return sentence
    ids = ", ".join(qids)
    cited = list(re.finditer(r"\(tables?: [^()]*\)", sentence))
    if cited:
        last = cited[-1]
        inner = last[0][:-1]
        if "query:" in inner:
            return sentence
        return f"{sentence[:last.start()]}{inner}; query: {ids}){sentence[last.end():]}"
    tables = list(dict.fromkeys(t for q in qids for t in book.tables(q, known)))
    citation = f" (table: {', '.join(tables)}; query: {ids})" if tables else ""
    stripped = sentence.rstrip()
    end = re.search(r"[.!?]+$", stripped)
    return (stripped[:end.start()] + citation + end[0]) if end else stripped + citation


def raw_numbers(text: str) -> int:
    """Numbers the Writer typed itself instead of using an ID: 100 or more, or with a decimal or %."""
    plain = re.sub(r"\((?:tables?|source): [^)]*\)|\(https?://[^)]*\)|" + ANY.pattern, "", text)
    found = re.findall(r"(?<![\w.,])\d+(?:,\d{3})*(?:\.\d+)?%?", plain)
    return sum(1 for n in found if "%" in n or "." in n
               or (float(n.replace(",", "")) >= 100 and not 1900 <= float(n.replace(",", "")) <= 2100))


# --- Tagging researcher findings with IDs ---

def _cell_numbers(value) -> list[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return []
    return [float(value), float(value) * 100]


def tag_findings(findings: str, tool_calls: list[dict], agent: str) -> str:
    """Add a fact ID after each number in a researcher's findings that lacks one, when one column of
    that researcher's own query results matches it. Numbers already tagged stay as they are."""
    book = FactBook([c for c in tool_calls if c.get("agent") == agent])
    if not book.results:
        return findings
    cells = []
    for qid, result in book.results.items():
        for r, row in enumerate(result["rows"], 1):
            for column, value in zip(result["columns"], row):
                for v in _cell_numbers(value):
                    cells.append((abs(v), f"{{{qid}.r{r}.{column}}}"))

    def valid(m: re.Match) -> str:
        fm = FACT.fullmatch(m[0])
        return m[0] if fm and not isinstance(book.lookup(fm[1], int(fm[2]), fm[3]), str) else ""

    # A made-up ID, such as {attendance.r1.x} or {Q6.row_count}, is removed, and the number before
    # it is tagged again below.
    findings = re.sub(r" ?" + ANY.pattern, lambda m: (" " + valid(m)) if valid(m) else "", findings)

    def tag(m: re.Match) -> str:
        if re.match(r"\s*\{", m.string[m.end():]):
            return m[0]  # already tagged
        text = m[1]
        number = text.rstrip("%").replace(",", "")
        decimals = len(number.split(".")[1]) if "." in number else 0
        value = float(number)
        if decimals == 0 and "%" not in text and (value < 100 or 1900 <= value <= 2100):
            return m[0]  # small counts, ranks and years
        step = 0.5 * 10 ** -decimals + 1e-9
        hits = list(dict.fromkeys(ref for v, ref in cells if abs(v - value) <= step))
        if not hits or len({ref.rsplit(".", 1)[1] for ref in hits}) != 1:
            return m[0]  # no match, or different columns match: leave it untagged
        return f"{m[0]} {hits[0]}"

    lines = []
    for line in findings.split("\n"):
        if "READY:" in line:  # a finished sentence is copied word for word: no IDs inside it
            lines.append(line)
            continue
        head, sep, tail = line.partition("(table")
        head = re.sub(r"(?<![\w.,{])(\d{1,3}(?:,\d{3})+(?:\.\d+)?%?|\d+(?:\.\d+)?%?)(?![\w}])(\s+(?:fans|viewers)\b)?",
                      tag, head)
        lines.append(head + sep + tail)
    return "\n".join(lines)
