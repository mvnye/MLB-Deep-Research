"""Choose at most one chart for an answer, from the rows a researcher's SQL returned.

The chart follows the shape of the result:
  line    a value over seasons or months, 4 or more points; one line per season for season by month
  bar     one value across 4 or more items
  paired  actual and expected across items
  diverging  the game model's associations, for a question about what drives attendance
  none    single values, 2 to 3 values, or an answer that says the data cannot answer

Charts are built only from query results, never from the Writer's text.

Colors: a team's own color for actual values and gray for expected. Two teams whose colors are
too close, for normal or colorblind vision, get a different color for the second team.
"""

import json
import re

MIN_ITEMS = 4
MIN_PAIRED_ITEMS = 3
MAX_ITEMS = 31
MAX_LABEL_CHARS = 40
MONTHS = {3: "March", 4: "April", 5: "May", 6: "June", 7: "July", 8: "August", 9: "September", 10: "October"}
TIME_COLUMN = re.compile(r"(^|_)(month|season|year)(_name|_num|_number)?$|^(m|mo|mon)$", re.I)
ID_COLUMN = re.compile(r"(_id|_pk|^n|^rank|^row_number|count|games|_games|num_|^number|corr|ci_low|ci_high)", re.I)
CHANGE_COLUMN = re.compile(r"diff|delta|prev|prior|lag", re.I)  # the step between rows, not the value
EXPECTED_COLUMN = re.compile(r"expected|predict", re.I)
ACTUAL_COLUMN = re.compile(r"actual|attendance|crowd", re.I)
MEASURE_ORDER = [r"(avg|average|mean).*(attendance|crowd)", r"attendance|crowd", r"viewers|audience", r"fill", r"pct|percent|change", r"avg|average|mean|total|sum"]
EXPECTED_QUESTION = re.compile(r"expected|predict|over-?perform|under-?perform", re.I)
NO_ANSWER = re.compile(r"cannot|can't|not in our data|no number|no audience|does not cover|outside|not available|only covers", re.I)
MONTH_NUMBERS = {name.lower(): number for number, name in MONTHS.items()}
SEASON_COLUMN = re.compile(r"^(season|year)$", re.I)
MONTH_COLUMN = re.compile(r"(^|_)month(_name|_num|_number)?$|^(m|mo|mon)$", re.I)
FILL_COLUMN = re.compile(r"fill", re.I)
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
    """Month numbers and dates become month names: 4, 2025-04 and 2025-04-01 are all April."""
    if MONTH_COLUMN.search(column):
        return MONTHS.get(month_number(value), str(value))
    return str(value)


def fill_values(columns: list[str], rows: list[list]) -> list | None:
    """A fill rate column, as percents, to show in the tooltip beside the main value."""
    index = next((i for i, c in enumerate(columns) if FILL_COLUMN.search(c)), None)
    if index is None or not any(is_number(r[index]) for r in rows):
        return None
    scale = 100.0 if max(abs(r[index]) for r in rows if is_number(r[index])) <= 1.5 else 1.0
    return [round(r[index] * scale, 1) if is_number(r[index]) else None for r in rows]


def month_number(value) -> int:
    text = str(value).strip().lower()
    match = re.fullmatch(r"(?:\d{4}-)?(\d{1,2})(?:\.0)?|\d{4}-(\d{2})-\d{2}.*", text)
    if match:
        return int(match[1] or match[2])
    return MONTH_NUMBERS.get(text, 0)


def season_lines(columns: list[str], rows: list[list], labels: list[int], measures: list[int], source: list[str]) -> dict | None:
    """One line per season, by month, from rows of season, month and a value."""
    season = next((i for i in labels if SEASON_COLUMN.search(columns[i])), None)
    month = next((i for i in labels if MONTH_COLUMN.search(columns[i])), None)
    if season is None or month is None or not measures:
        return None
    values = [i for i in measures if not CHANGE_COLUMN.search(columns[i]) and not EXPECTED_COLUMN.search(columns[i])]
    if not values:
        return None
    measure = main_measure(columns, values)
    months = sorted({month_number(r[month]) for r in rows} - {0})
    seasons = sorted({r[season] for r in rows})
    if len(seasons) < 2 or len(months) < MIN_PAIRED_ITEMS:
        return None
    found = {(r[season], month_number(r[month])): r[measure] for r in rows}
    fills = fill_values(columns, rows)
    fill_at = {(r[season], month_number(r[month])): f for r, f in zip(rows, fills)} if fills else {}
    unit, scale = unit_of(columns[measure], [v for v in found.values() if v is not None])
    name = words(columns[measure])
    return {
        "type": "line",
        "title": f"{name} by month, {' vs '.join(str(x) for x in seasons)}",
        "x_label": "Month",
        "y_label": name if not unit or unit in name.lower() else f"{name} ({unit})",
        "unit": unit,
        "labels": [MONTHS.get(m, str(m)) for m in months],
        "series": [{"name": str(s), "values": [None if found.get((s, m)) is None else round(found[(s, m)] * scale, 1)
                                               for m in months],
                    **({"fill": [fill_at.get((s, m)) for m in months]} if fill_at else {})} for s in seasons],
        "source": source,
    }


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
    if any(re.search(r"(^|_)date$", c, re.I) for c in columns):
        return None  # one row per game: a box score shows these, not a chart
    labels, measures = split_columns(columns, rows)
    lines = season_lines(columns, rows, labels, measures, source)
    if lines:
        return lines
    unique = [i for i in labels if len({str(r[i]) for r in rows}) == len(rows)]
    if len(labels) > 1 and unique:
        labels = unique  # a text column that repeats, such as a unit, does not name the rows
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
        chart = build("line", columns, ordered, label, [measure], source)
        fills = fill_values(columns, ordered) if not FILL_COLUMN.search(columns[measure]) else None
        if fills:
            chart["series"][0]["fill"] = fills
        return chart
    return build("bar", columns, sorted(rows, key=lambda r: -r[measure]), label, [measure], source)


def relevant(chart: dict, report: str) -> bool:
    """A chart has to be about what the answer says: two of its items or values must appear."""
    text = report.lower()
    named = sum(1 for label in chart["labels"] if label.lower() in text)
    values = sum(1 for v in chart["series"][0]["values"]
                 if v is not None and (f"{v:,.0f}" in report or f"{v:.1f}%" in report))
    return max(named, values) >= max(2, len(chart["labels"]) / 2)  # mostly about what the answer says


def pick_chart(report: str, question: str, tool_calls: list[dict], tables: list[str]) -> dict | None:
    """The chart for an answer, from the last query whose result fits, or None."""
    if NO_ANSWER.search(report[:200]):
        return None
    wants_expected = bool(EXPECTED_QUESTION.search(question))
    fallback = None
    for call in reversed(tool_calls):
        if call["name"] != "run_sql" or call.get("agent") == "Required facts check":
            continue
        result = json.loads(call["result"])
        if "rows" not in result or result.get("truncated"):
            continue
        chart = chart_from_rows(result["columns"], result["rows"], tables_in(call["args"]["query"], tables), wants_expected)
        if chart and len(chart["series"]) == 1 + (chart["type"] == "paired") and relevant(chart, report):
            # one line per season is only for a question comparing seasons: see monthly_chart
            if chart["type"] == "paired" or not wants_expected:
                return chart
            # A question about expected attendance: results with actual and expected values win over a
            # later plain chart (such as a fill query that ran after them).
            fallback = fallback or chart
    return fallback


def monthly_chart(tool_calls: list[dict], tables: list[str], seasons: list[int], kind: str = "attendance") -> dict | None:
    """The latest season-by-month chart a query supports, for these seasons. kind is attendance or viewers."""
    for call in reversed(tool_calls):
        if call["name"] != "run_sql":
            continue
        result = json.loads(call["result"])
        if "rows" not in result or result.get("truncated"):
            continue
        chart = chart_from_rows(result["columns"], result["rows"], tables_in(call["args"]["query"], tables))
        unit = "viewers" if kind == "viewers" else "fans"
        if (chart and len(chart["series"]) >= 2 and chart["unit"] == unit
                and {str(s) for s in seasons} <= {s["name"] for s in chart["series"]}):
            return chart
    return None


# --- What drives attendance ---

DRIVER_BARS_EACH_WAY = 7
OUTLIER = "Home openers"  # far larger than the rest, so it is stated apart to keep the scale readable
PER_UNIT = re.compile(r"^(Each|A win rate)")
UP_COLOR, DOWN_COLOR = "#12402E", "#A8501F"


def short_factor(label: str) -> str:
    label = re.sub(r"^Games against the (.+), compared with the average opponent$", r"vs \1", label)
    label = re.sub(r"^(.+) games, compared with (?:day games|(.+))$", lambda m: f"{m[1]} vs {m[2] or 'day'}", label)
    return re.sub(r", which partly reflects bad weather$", "", label)


def driver_chart(rows: list[list]) -> dict:
    """A diverging bar chart of the clear game-level associations, largest each way."""
    clear = [(label, effect) for label, effect, is_clear in rows if is_clear and not PER_UNIT.search(label)]
    outlier = [(label, effect) for label, effect in clear if label == OUTLIER]
    rest = sorted((c for c in clear if c[0] != OUTLIER), key=lambda c: -c[1])
    up = [c for c in rest if c[1] > 0][:DRIVER_BARS_EACH_WAY]
    down = [c for c in rest if c[1] < 0][-DRIVER_BARS_EACH_WAY:]
    shown = up + down
    return {
        "type": "diverging",
        "title": "Game factors associated with attendance",
        "x_label": "Factor",
        "y_label": "Attendance, % vs a typical game",
        "unit": "%",
        "labels": [short_factor(label) for label, _ in shown],
        "series": [{"name": "Difference", "values": [round(effect * 100, 1) for _, effect in shown], "style": "solid", "color": UP_COLOR}],
        "item_colors": [UP_COLOR if effect > 0 else DOWN_COLOR for _, effect in shown],
        "callouts": [{"label": label, "value": f"{effect:+.1%}"} for label, effect in outlier],
        "note": "Each bar is compared with its own baseline: Friday for days, August for months, and the "
                "average opponent for opponents. Only the largest clear associations are shown.",
        "source": ["model_coefficients"],
    }


# --- Colors ---

# Primary and secondary colors, from teamcolorcodes.com.
TEAM_COLORS = {
    "Arizona Diamondbacks": ("#A71930", "#E3D4AD"), "Athletics": ("#003831", "#EFB21E"),
    "Atlanta Braves": ("#CE1141", "#13274F"), "Baltimore Orioles": ("#DF4601", "#000000"),
    "Boston Red Sox": ("#BD3039", "#0C2340"), "Chicago Cubs": ("#0E3386", "#CC3433"),
    "Chicago White Sox": ("#27251F", "#C4CED4"), "Cincinnati Reds": ("#C6011F", "#000000"),
    "Cleveland Guardians": ("#00385D", "#E50022"), "Colorado Rockies": ("#333366", "#C4CED4"),
    "Detroit Tigers": ("#0C2340", "#FA4616"), "Houston Astros": ("#002D62", "#EB6E1F"),
    "Kansas City Royals": ("#004687", "#BD9B60"), "Los Angeles Angels": ("#BA0021", "#003263"),
    "Los Angeles Dodgers": ("#005A9C", "#EF3E42"), "Miami Marlins": ("#00A3E0", "#EF3340"),
    "Milwaukee Brewers": ("#12284B", "#FFC52F"), "Minnesota Twins": ("#002B5C", "#D31145"),
    "New York Mets": ("#002D72", "#FF5910"), "New York Yankees": ("#003087", "#E4002C"),
    "Oakland Athletics": ("#003831", "#EFB21E"), "Philadelphia Phillies": ("#E81828", "#002D72"),
    "Pittsburgh Pirates": ("#27251F", "#FDB827"), "San Diego Padres": ("#2F241D", "#FFC425"),
    "San Francisco Giants": ("#FD5A1E", "#27251F"), "Seattle Mariners": ("#0C2C56", "#005C5C"),
    "St. Louis Cardinals": ("#C41E3A", "#0C2340"), "Tampa Bay Rays": ("#092C5C", "#8FBCE6"),
    "Texas Rangers": ("#003278", "#C0111F"), "Toronto Blue Jays": ("#134A8E", "#1D2D5C"),
    "Washington Nationals": ("#AB0003", "#14225A"),
}
DEFAULT_COLOR = "#12402E"  # scoreboard green, for charts not about a team
GRAY = "#8A8F98"  # expected values: also drawn hatched, so identity is never color alone
DARK_GRAY = "#5F6670"
FALLBACK_COLORS = ["#0B2545", "#C8102E", "#00797A", "#6A3FA0", "#000000"]  # all distinct from each other
MAX_TEAM_COLORS = 5  # past this, colors stop telling teams apart: the labels do it
MIN_CONTRAST = 3.0  # against the white chart surface
MIN_DISTANCE = 15.0  # normal vision, OKLab x 100
MIN_CVD_DISTANCE = 8.0  # colorblind vision
CVD_MATRICES = [  # Machado 2009, full severity: protan, deutan, tritan
    ((0.152286, 1.052583, -0.204868), (0.114503, 0.786281, 0.099216), (-0.003882, -0.048116, 1.051998)),
    ((0.367322, 0.860646, -0.227968), (0.280085, 0.672501, 0.047413), (-0.011820, 0.042940, 0.968881)),
    ((1.255528, -0.076749, -0.178779), (-0.078411, 0.930809, 0.147602), (0.004733, 0.691367, 0.303900)),
]


def linear_rgb(color: str) -> tuple[float, ...]:
    channels = [int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    return tuple(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels)


def contrast_on_white(color: str) -> float:
    r, g, b = linear_rgb(color)
    return 1.05 / (0.2126 * r + 0.7152 * g + 0.0722 * b + 0.05)


def readable(color: str) -> str:
    """Darken a color until it stands out on white and from the gray used for expected values."""
    channels = [int(color[i:i + 2], 16) for i in (1, 3, 5)]
    hex_color = lambda: "#%02X%02X%02X" % tuple(channels)
    while contrast_on_white(hex_color()) < MIN_CONTRAST or not distinct(hex_color(), GRAY):
        channels = [int(c * 0.95) for c in channels]
    return hex_color()


def oklab(rgb: tuple[float, ...]) -> tuple[float, ...]:
    r, g, b = (max(c, 0.0) for c in rgb)
    l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return (0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s)


def distance(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return 100 * sum((x - y) ** 2 for x, y in zip(oklab(a), oklab(b))) ** 0.5


def distinct(a: str, b: str) -> bool:
    """Can these two colors be told apart, with normal and with colorblind vision?"""
    x, y = linear_rgb(a), linear_rgb(b)
    if distance(x, y) < MIN_DISTANCE:
        return False
    seen = lambda matrix, c: tuple(sum(k * v for k, v in zip(row, c)) for row in matrix)
    return all(distance(seen(m, x), seen(m, y)) >= MIN_CVD_DISTANCE for m in CVD_MATRICES)


def team_name(label: str) -> str | None:
    """'Yankees' or 'New York Yankees' -> 'New York Yankees'."""
    return next((t for t in TEAM_COLORS if label == t or t.endswith(f" {label}")), None)


def team_color(team: str | None) -> str:
    name = team_name(team) if team else None
    return readable(TEAM_COLORS[name][0]) if name else DEFAULT_COLOR


def assign_colors(teams: list[str]) -> list[str]:
    """One color per team: its primary, else its secondary, else the next fallback that is distinct."""
    chosen: list[str] = []
    for team in teams:
        options = [readable(c) for c in (*TEAM_COLORS[team_name(team)], *FALLBACK_COLORS)]
        fits = lambda c: all(distinct(c, other) for other in chosen)
        chosen.append(next((c for c in options if fits(c)), options[0]))
    clash = any(not distinct(a, b) for i, a in enumerate(chosen) for b in chosen[:i])
    return FALLBACK_COLORS[:len(chosen)] if clash else chosen


def paint(chart: dict, team: str | None = None) -> dict:
    """Give a chart its colors. team is the one team the question is about, if any."""
    if chart["type"] == "diverging":
        return chart  # up and down have their own colors
    series = chart["series"]
    main = team_color(team)
    if chart["type"] == "line" and len(series) > 1:  # the newest season stands out, earlier ones recede
        others = [(DARK_GRAY, "dashed"), (GRAY, "dotted")]
        for s, (color, style) in zip(reversed(series), [(main, "solid"), *others, *others]):
            s["color"], s["style"] = color, style
        return chart
    series[0]["color"], series[0]["style"] = main, "solid"
    if len(series) > 1:
        series[1]["color"], series[1]["style"] = GRAY, "hatch"
    if chart["type"] == "bar" or chart["type"] == "paired":
        names = [team_name(label) for label in chart["labels"]]
        if all(names) and len(set(names)) <= MAX_TEAM_COLORS:
            chart["item_colors"] = assign_colors(names)
    return chart
