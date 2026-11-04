"""Clean the Stats API pull and the viewership CSVs and load them into data/mlb.duckdb.

Tables:
  teams                   one row per team per season
  team_season_attendance  the API's own season totals, used to check the game table
  games                   one row per played regular-season game
  viewership_games        one row per single-game national broadcast, including
                          single-game Fox windows
  fox_windows             one row per Fox window (single-game and multi-game)
  fox_window_games        games in each Fox window, with coverage_pct where given
  capacity_overrides      hand-checked capacities, each with a published source

Viewer counts are whole viewers (1.962M is stored as 1962000). NULL always means unknown.
wildcard_2026.csv is not loaded: the postseason is out of scope.

Usage: uv run python scripts/load_duckdb.py
"""

import csv
import json
import re
import sys
from collections import Counter
from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
API = RAW / "statsapi"
DB_PATH = ROOT / "data" / "mlb.duckdb"
SEASONS = [2024, 2025, 2026]
ET = ZoneInfo("America/New_York")

VIEWERSHIP_FILES = [
    ("Fox", 2025, "viewership_2025/fox_2025.csv"),
    ("FS1", 2025, "viewership_2025/fs1_2025.csv"),
    ("ESPN", 2025, "viewership_2025/espn_2025.csv"),
    ("TBS", 2025, "viewership_2025/tbs_2025.csv"),
    ("ABC", 2026, "viewership_2026/abc_2026.csv"),
    ("Fox", 2026, "viewership_2026/fox_2026.csv"),
    ("NBC", 2026, "viewership_2026/nbc_2026.csv"),
    ("ESPN", 2026, "viewership_2026/espn_2026.csv"),
    ("TBS", 2026, "viewership_2026/tbs_2026.csv"),
    ("FS1", 2026, "viewership_2026/fs1_2026.csv"),
    ("Netflix", 2026, "viewership_2026/netflix_2026.csv"),
]

NON_GAMES = {"Home Run Derby", "All-Star Game"}
TEAM_ALIASES = {"dbacks": "Diamondbacks", "d-backs": "Diamondbacks"}

# Games away from the home team's normal park. Excluded from comparisons by default.
SPECIAL_VENUES = {
    "Gocheok Sky Dome": "Seoul Series",
    "Estadio Alfredo Harp Helu": "Mexico City Series",
    "London Stadium": "London Series",
    "Rickwood Field": "Rickwood Field Game",
    "Journey Bank Ballpark": "Little League Classic",
    "Tokyo Dome": "Tokyo Series",
    "Bristol Motor Speedway": "Speedway Classic",
    "Field of Dreams": "Field of Dreams",
    "Las Vegas Ballpark": "Athletics in Las Vegas",
}
# The A's final homestand before leaving Oakland, 9/20 to 9/26/2024.
FAREWELL_GAMES = {pk: "Athletics farewell to Oakland"
                  for pk in (745606, 745610, 745608, 745605, 745607, 745609)}
TEMPORARY_PARKS = {"George M. Steinbrenner Field", "Sutter Health Park"}
RAYS_ID = 139
CAPACITY_OVERRIDES = ROOT / "data" / "capacity_overrides.csv"

# Nielsen Big Data + Panel became the official currency on this date.
NIELSEN_BIG_DATA_START = date(2025, 9, 1)


# ---------- Stats API ----------

def load_jsonl(season: int) -> list[dict]:
    with (API / f"games_{season}.jsonl").open() as f:
        return [json.loads(line) for line in f]


def special_event(game_pk: int, venue: str, home_id: int) -> str | None:
    if game_pk in FAREWELL_GAMES:
        return FAREWELL_GAMES[game_pk]
    if venue == "Tropicana Field" and home_id != RAYS_ID:
        return "Neutral site at Tropicana Field"
    return SPECIAL_VENUES.get(venue)


def parse_wind(wind: str | None) -> tuple[int | None, str | None]:
    """'3 mph, R To L' -> (3, 'R To L')."""
    m = re.fullmatch(r"(\d+) mph, (.+)", wind or "")
    return (int(m[1]), m[2]) if m else (None, None)


def game_row(raw: dict) -> dict:
    s, gd = raw["schedule"], raw["feed"]["gameData"]
    home, away = s["teams"]["home"], s["teams"]["away"]
    weather, info, venue = gd.get("weather", {}), gd.get("gameInfo", {}), gd["venue"]
    field = venue.get("fieldInfo", {})
    pitchers = gd.get("probablePitchers", {})
    start_utc = datetime.fromisoformat(s["gameDate"].replace("Z", "+00:00"))
    official = date.fromisoformat(s["officialDate"])
    wind_mph, wind_dir = parse_wind(weather.get("wind"))
    temp = int(weather["temp"]) if str(weather.get("temp", "")).isdigit() else None
    raw_att = info.get("attendance")
    return {
        "game_pk": s["gamePk"],
        "season": int(s["season"]),
        "official_date": official,
        "weekday": official.strftime("%A"),
        "start_utc": start_utc.replace(tzinfo=None),
        "start_time_et": start_utc.astimezone(ET).time().replace(tzinfo=None),
        "day_night": s["dayNight"],
        "doubleheader": s["doubleHeader"],
        "game_number": s["gameNumber"],
        "status": s["status"]["detailedState"],
        "home_team_id": home["team"]["id"],
        "home_team": home["team"]["name"],
        "away_team_id": away["team"]["id"],
        "away_team": away["team"]["name"],
        "home_score": home.get("score"),
        "away_score": away.get("score"),
        "home_wins_after": home["leagueRecord"]["wins"],
        "home_losses_after": home["leagueRecord"]["losses"],
        "away_wins_after": away["leagueRecord"]["wins"],
        "away_losses_after": away["leagueRecord"]["losses"],
        "venue_id": s["venue"]["id"],
        "venue_name": s["venue"]["name"],
        "venue_city": venue.get("location", {}).get("city"),
        "venue_state": venue.get("location", {}).get("state"),
        "capacity_api": field.get("capacity"),
        "roof_type": field.get("roofType"),
        "turf_type": field.get("turfType"),
        # The API uses 0 for a game with no separate gate, such as game 1 of a
        # single-admission doubleheader. That is unknown, not an empty park.
        "attendance": raw_att or None,
        "attendance_api_zero": raw_att == 0,
        "weather_condition": weather.get("condition"),
        "temp_f": temp or None,  # 0 appears only as a placeholder
        "wind_mph": wind_mph,
        "wind_direction": wind_dir,
        "duration_minutes": info.get("gameDurationMinutes"),
        "home_probable_pitcher": pitchers.get("home", {}).get("fullName"),
        "away_probable_pitcher": pitchers.get("away", {}).get("fullName"),
        "description": s.get("description"),
        "rescheduled_from": s.get("rescheduledFromDate"),
        "resumed_from": s.get("resumedFromDate"),
        "series_game_number": s.get("seriesGameNumber"),
        "games_in_series": s.get("gamesInSeries"),
        "special_event": special_event(s["gamePk"], s["venue"]["name"], home["team"]["id"]),
        "temporary_park": s["venue"]["name"] in TEMPORARY_PARKS,
    }


def add_pregame_records(games: list[dict]) -> None:
    """Add each team's record entering the game, counted from the games themselves.

    The API's leagueRecord is post-game. Results are checked against it at the end.
    """
    records: dict[tuple[int, int], list[int]] = {}
    mismatches = 0
    for g in sorted(games, key=lambda g: (g["official_date"], g["game_number"], g["start_utc"])):
        home_won = g["home_score"] > g["away_score"]
        for side, won in (("home", home_won), ("away", not home_won)):
            rec = records.setdefault((g["season"], g[f"{side}_team_id"]), [0, 0])
            g[f"{side}_wins_before"], g[f"{side}_losses_before"] = rec
            played = rec[0] + rec[1]
            g[f"{side}_win_pct_before"] = rec[0] / played if played else None
            rec[0 if won else 1] += 1
            if rec != [g[f"{side}_wins_after"], g[f"{side}_losses_after"]]:
                mismatches += 1
    print(f"pre-game records: {mismatches} team-games differ from the API's post-game record")


def add_home_openers(games: list[dict]) -> None:
    """Flag each team's first home game of the season, skipping special events.

    Uses the schedule, not the description text, which misses some openers.
    """
    seen = set()
    for g in sorted(games, key=lambda g: (g["official_date"], g["game_number"])):
        key = (g["season"], g["home_team_id"])
        g["home_opener"] = g["special_event"] is None and key not in seen
        if g["home_opener"]:
            seen.add(key)


def add_shared_gates(games: list[dict]) -> None:
    """Flag the game holding the whole gate of a single-admission doubleheader."""
    zero_days = {(g["home_team_id"], g["official_date"]) for g in games
                 if g["attendance_api_zero"] and g["doubleheader"] == "Y"}
    for g in games:
        g["dh_shared_gate"] = (g["attendance"] is not None
                               and (g["home_team_id"], g["official_date"]) in zero_days)


def add_fill_rates(games: list[dict], overrides: dict[tuple[int, str], int]) -> None:
    """fill_rate_raw uses the API capacity, unclamped. fill_rate uses a checked override."""
    for g in games:
        override = overrides.get((g["season"], g["venue_name"]))
        api, att = g["capacity_api"], g["attendance"]
        capacity = override or api
        g["capacity_used"] = capacity
        g["capacity_source"] = "override" if override else ("api" if api else None)
        g["fill_rate_raw"] = att / api if att and api else None
        g["fill_rate"] = att / capacity if att and capacity else None
        g["over_capacity"] = None if g["fill_rate_raw"] is None else g["fill_rate_raw"] > 1


def read_capacity_overrides() -> tuple[list[dict], dict[tuple[int, str], int]]:
    with CAPACITY_OVERRIDES.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if not r["source_url"].startswith("http"):
            sys.exit(f"capacity override without a published source: {r}")
    return rows, {(int(r["season"]), r["venue_name"]): int(r["capacity"]) for r in rows}


# ---------- Viewership parsing ----------

def parse_number(token: str) -> int:
    """Turn '1.962M', '362K', '.844K' or '1478' (thousands) into whole viewers.

    A leading dot before K is a source typo: '.844K' means 844K.
    """
    t = token.strip()
    m = re.fullmatch(r"(\d*\.?\d+)([MK]?)", t)
    if not m:
        raise ValueError(f"cannot parse viewer number {token!r}")
    value, suffix = float(m[1]), m[2]
    if suffix == "M" or (suffix == "K" and t.startswith(".")):
        return round(value * 1_000_000)
    return round(value * 1_000)


def parse_viewers(raw: str, addon_listed: bool) -> dict:
    """Split a viewership cell into total, main network and add-on networks.

    viewers_total stays NULL when an add-on network's number is missing, so a partial
    count can never be compared as if it were a full total.
    """
    out = {"viewers_total": None, "viewers_main": None, "viewers_addon": None,
           "addon_missing": False}
    s = raw.strip()
    if not s or s.upper() == "N/A":
        return out
    m = re.fullmatch(r"(N/A|[\d.]+[MK]?)\s*(?:\((.*?)\)?)?", s)
    if not m:
        raise ValueError(f"cannot parse viewership cell {raw!r}")
    head, inner = m[1], (m[2] or "").strip()

    if inner.lower() in ("tbs only", "additional trutv n/a"):
        out.update(viewers_main=parse_number(head), addon_missing=True)
    elif head == "N/A":  # 'N/A (32K Fox Deportes)': only the add-on number is known
        out.update(viewers_addon=parse_number(inner.split()[0]))
    elif inner:  # '1.962M (1.931M + 31K)'
        parts = [parse_number(p) for p in inner.split("+")]
        out.update(viewers_total=parse_number(head), viewers_main=parts[0],
                   viewers_addon=sum(parts[1:]))
    else:
        total = parse_number(head)
        out.update(viewers_total=total, viewers_main=None if addon_listed else total)
    return out


def parse_date(raw: str, season: int) -> date:
    """'Tuesday, March 18 (Tokyo, co-exist)' -> date. Checks the weekday against the year.

    For 'Saturday, August 2/Sunday, August 3' the first date is used.
    """
    m = re.match(r"\s*(\w+), (\w+) (\d+)", raw)
    if not m:
        raise ValueError(f"cannot parse date {raw!r}")
    d = datetime.strptime(f"{m[2]} {m[3]} {season}", "%B %d %Y").date()
    if d.strftime("%A") != m[1]:
        raise ValueError(f"{raw!r}: {d} is a {d.strftime('%A')} in {season}")
    return d


def parse_time(raw: str) -> time | None:
    """'6:30 AM', '3:30PM', '8PM', '7:00:00 PM/1:00 PM', 'JIPed at 8:30PM' -> time."""
    m = re.search(r"(\d{1,2})(?::(\d{2}))?(?::\d{2})?\s*([AP])M", raw)
    if not m:
        return None
    hour = int(m[1]) % 12 + (12 if m[3] == "P" else 0)
    return time(hour, int(m[2] or 0))


def split_teams(text: str) -> tuple[list[str], list[str]]:
    """'Guardians*/Yankees' -> (['Guardians', 'Yankees'], ['Guardians'])."""
    text = re.sub(r"^LLC:\s*", "", text.strip())
    names = [n.strip() for n in re.split(r"[/-]", text)]
    blackout = [n.rstrip("*") for n in names if n.endswith("*")]
    return [n.rstrip("*") for n in names], blackout


def parse_matchups(raw: str) -> list[dict]:
    """Split 'A/B (64%) or C/D (36%)' into games. Tolerates a missing ')' and ' OR '."""
    games = []
    for part in re.split(r"\s+or\s+", raw.strip(), flags=re.IGNORECASE):
        pct = re.search(r"\((\d+)%\)?", part)
        teams, blackout = split_teams(re.sub(r"\(\d+%\)?", "", part))
        games.append({"teams": teams, "blackout": blackout,
                      "coverage_pct": int(pct[1]) if pct else None})
    return games


def read_csv(path: Path) -> list[dict]:
    """Return rows with normalized keys: date, time, matchup, additional, viewers."""
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = {}
        for name in reader.fieldnames or []:
            low = name.strip().lower()
            if low.startswith("date"):
                fields["date"] = name
            elif low.startswith("time"):
                fields["time"] = name
            elif low.startswith("matchup"):
                fields["matchup"] = name
            elif low.startswith("additional"):
                fields["additional"] = name
            # NBC also has 'Big Data + Adobe Analytics streaming': Nielsen only here.
            elif low.startswith(("viewership", "viewers", "big data viewership")):
                fields["viewers"] = name
        return [{k: (row.get(fields[k]) or "").strip() if k in fields else ""
                 for k in ("date", "time", "matchup", "additional", "viewers")}
                for row in reader]


# ---------- Matching to games ----------

class Matcher:
    def __init__(self, games: list[dict], teams: list[dict]):
        self.games = games
        self.ids = {}
        for t in teams:
            for key in (t["teamName"], t["clubName"], t["franchiseName"]):
                if key:
                    self.ids[key.lower()] = t["id"]
        for alias, name in TEAM_ALIASES.items():
            self.ids[alias] = self.ids[name.lower()]

    def team_id(self, name: str) -> int:
        tid = self.ids.get(name.lower())
        if tid is None:
            raise ValueError(f"unknown team name {name!r}")
        return tid

    def match(self, names: list[str], day: date, listed: time | None, season: int) -> dict | None:
        """Find the game by team pair and date, using start time to split doubleheaders."""
        pair = {self.team_id(n) for n in names}
        same = [g for g in self.games
                if g["season"] == season and {g["home_team_id"], g["away_team_id"]} == pair]
        hits = [g for g in same if g["official_date"] == day]
        if not hits:  # a makeup game keeps its original TV listing date
            hits = [g for g in same if g["rescheduled_from"] == day.isoformat()]
        if len(hits) > 1 and listed:
            hits = [min(hits, key=lambda g: minutes_apart(g["start_time_et"], listed))]
        return hits[0] if len(hits) == 1 else None


def minutes_apart(a: time, b: time) -> int:
    return abs((a.hour * 60 + a.minute) - (b.hour * 60 + b.minute))


def measurement(day: date) -> str:
    """Nielsen's method on this date. The two are not like-for-like."""
    return "panel" if day < NIELSEN_BIG_DATA_START else "big_data_panel"


def build_viewership(matcher: Matcher) -> tuple[list[dict], list[dict], list[dict], Counter]:
    singles, windows, window_games = [], [], []
    stats = Counter()
    problems = []

    for network, season, rel in VIEWERSHIP_FILES:
        for row in read_csv(RAW / rel):
            if row["matchup"].lower().startswith("average"):
                stats["dropped average rows"] += 1
                continue
            if row["matchup"] in NON_GAMES:
                stats["dropped non-games"] += 1
                continue

            day = parse_date(row["date"], season)
            listed = parse_time(row["time"])
            addon_listed = bool(row["additional"])
            if network == "TBS" and season == 2026:
                addon_listed = True  # TruTV simulcast; a plain number has no split
            v = parse_viewers(row["viewers"], addon_listed)
            if v["viewers_total"] and v["viewers_main"] is not None:
                gap = v["viewers_total"] - v["viewers_main"] - (v["viewers_addon"] or 0)
                if abs(gap) > 1000:
                    problems.append(f"{network} {day}: parts do not sum: {row['viewers']}")

            games = parse_matchups(row["matchup"])
            matched = []
            for g in games:
                hit = matcher.match(g["teams"], day, listed, season)
                if hit is None:
                    problems.append(f"{network} {day} {listed}: no game for {g['teams']}")
                matched.append(hit)

            basis = ("total" if v["viewers_total"] is not None
                     else "main_only" if v["viewers_main"] is not None else None)
            common = {
                "season": season, "network": network, "listed_date": day,
                "viewers_within_network": (v["viewers_total"] if basis == "total"
                                           else v["viewers_main"]),
                "viewers_basis": basis,
                # A plain number with a simulcast listed: it may or may not include truTV.
                "total_uncertain": (v["viewers_total"] is not None
                                    and v["viewers_main"] is None),
                "audience_scope": ("streaming_only" if network == "Netflix"
                                   else "tv_excl_streaming_addons"),
                "listed_time_et": listed, "additional_networks": row["additional"] or None,
                "measurement": measurement(day), "date_raw": row["date"],
                "time_raw": row["time"] or None, "matchup_raw": row["matchup"],
                "viewers_raw": row["viewers"] or None, **v,
            }

            window_id = None
            if network == "Fox":
                stamp = listed.strftime("%H%M") if listed else "0000"
                window_id = f"fox-{day.isoformat()}-{stamp}"
                windows.append({"window_id": window_id, "n_games": len(games), **common})
                for g, hit in zip(games, matched):
                    window_games.append({
                        "window_id": window_id, "game_pk": hit["game_pk"] if hit else None,
                        "matchup": "/".join(g["teams"]), "coverage_pct": g["coverage_pct"],
                    })
                pcts = [g["coverage_pct"] for g in games if g["coverage_pct"] is not None]
                if pcts and sum(pcts) != 100:
                    problems.append(f"Fox {day}: coverage adds to {sum(pcts)}")

            if len(games) == 1:
                hit, g = matched[0], games[0]
                if hit and listed and "/" not in row["time"]:
                    gap = minutes_apart(hit["start_time_et"], listed)
                    if gap > 45:
                        stats["time gaps over 45 min"] += 1
                        print(f"  note: {network} {day} {row['matchup']}: listed {listed}, "
                              f"API {hit['start_time_et']}")
                singles.append({
                    "game_pk": hit["game_pk"] if hit else None, "fox_window_id": window_id,
                    "joined_in_progress": "jip" in row["time"].lower(),
                    "blackout_lifted": bool(g["blackout"]),
                    "blackout_lifted_teams": ", ".join(g["blackout"]) or None, **common,
                })

    if problems:
        print("\n".join(problems))
        sys.exit(f"{len(problems)} viewership rows failed checks")
    return singles, windows, window_games, stats


# ---------- Load ----------

def create_table(con, name: str, rows: list[dict], types: dict[str, str]) -> None:
    """Create a table from dicts. Column types come from `types`, in `types` order."""
    cols = ", ".join(f"{c} {t}" for c, t in types.items())
    con.execute(f"CREATE TABLE {name} ({cols})")
    marks = ", ".join("?" for _ in types)
    if rows:
        con.executemany(f"INSERT INTO {name} VALUES ({marks})",
                        [[r.get(c) for c in types] for r in rows])


GAME_TYPES = {
    "game_pk": "INTEGER PRIMARY KEY", "season": "INTEGER", "official_date": "DATE",
    "weekday": "VARCHAR", "start_utc": "TIMESTAMP", "start_time_et": "TIME",
    "day_night": "VARCHAR", "doubleheader": "VARCHAR", "game_number": "INTEGER",
    "status": "VARCHAR", "home_team_id": "INTEGER", "home_team": "VARCHAR",
    "away_team_id": "INTEGER", "away_team": "VARCHAR", "home_score": "INTEGER",
    "away_score": "INTEGER", "home_wins_after": "INTEGER", "home_losses_after": "INTEGER",
    "away_wins_after": "INTEGER", "away_losses_after": "INTEGER",
    "home_wins_before": "INTEGER", "home_losses_before": "INTEGER",
    "home_win_pct_before": "DOUBLE", "away_wins_before": "INTEGER",
    "away_losses_before": "INTEGER", "away_win_pct_before": "DOUBLE", "venue_id": "INTEGER",
    "venue_name": "VARCHAR", "venue_city": "VARCHAR", "venue_state": "VARCHAR",
    "capacity_api": "INTEGER", "roof_type": "VARCHAR", "turf_type": "VARCHAR",
    "attendance": "INTEGER", "attendance_api_zero": "BOOLEAN", "dh_shared_gate": "BOOLEAN",
    "capacity_used": "INTEGER", "capacity_source": "VARCHAR", "fill_rate_raw": "DOUBLE",
    "fill_rate": "DOUBLE", "over_capacity": "BOOLEAN",
    "weather_condition": "VARCHAR", "temp_f": "INTEGER", "wind_mph": "INTEGER",
    "wind_direction": "VARCHAR", "duration_minutes": "INTEGER",
    "home_probable_pitcher": "VARCHAR", "away_probable_pitcher": "VARCHAR",
    "description": "VARCHAR", "rescheduled_from": "DATE", "resumed_from": "DATE",
    "series_game_number": "INTEGER", "games_in_series": "INTEGER",
    "special_event": "VARCHAR", "temporary_park": "BOOLEAN", "home_opener": "BOOLEAN",
}
BROADCAST_TYPES = {
    "season": "INTEGER", "network": "VARCHAR", "listed_date": "DATE",
    "listed_time_et": "TIME", "additional_networks": "VARCHAR", "viewers_total": "INTEGER",
    "viewers_main": "INTEGER", "viewers_addon": "INTEGER", "addon_missing": "BOOLEAN",
    "viewers_within_network": "INTEGER", "viewers_basis": "VARCHAR", "total_uncertain": "BOOLEAN",
    "audience_scope": "VARCHAR",
    "measurement": "VARCHAR", "date_raw": "VARCHAR", "time_raw": "VARCHAR",
    "matchup_raw": "VARCHAR", "viewers_raw": "VARCHAR",
}
SINGLE_TYPES = {
    "game_pk": "INTEGER REFERENCES games(game_pk)", "fox_window_id": "VARCHAR",
    "joined_in_progress": "BOOLEAN", "blackout_lifted": "BOOLEAN",
    "blackout_lifted_teams": "VARCHAR", **BROADCAST_TYPES,
}
WINDOW_TYPES = {"window_id": "VARCHAR PRIMARY KEY", "n_games": "INTEGER", **BROADCAST_TYPES}
WINDOW_GAME_TYPES = {
    "window_id": "VARCHAR REFERENCES fox_windows(window_id)",
    "game_pk": "INTEGER REFERENCES games(game_pk)", "matchup": "VARCHAR",
    "coverage_pct": "INTEGER",
}
TEAM_TYPES = {
    "team_id": "INTEGER", "season": "INTEGER", "name": "VARCHAR", "club_name": "VARCHAR",
    "abbreviation": "VARCHAR", "league": "VARCHAR", "division": "VARCHAR",
}
OVERRIDE_TYPES = {
    "season": "INTEGER", "venue_name": "VARCHAR", "capacity": "INTEGER",
    "source_url": "VARCHAR", "checked_on": "DATE", "note": "VARCHAR",
}
ATTENDANCE_TYPES = {
    "team_id": "INTEGER", "season": "INTEGER", "home_openings": "INTEGER",
    "home_total": "INTEGER", "home_average": "INTEGER", "high": "INTEGER", "low": "INTEGER",
}


VIEWER_COMMENTS = {
    "viewers_total": "National TV audience, excluding streaming add-ons: main network plus TV "
                     "simulcasts (Deportes, ESPN2, truTV). Peacock/Adobe and HBO Max are not "
                     "counted. Netflix games are streaming-only. NULL means unknown.",
    "viewers_main": "Main network only. NULL if not reported separately.",
    "viewers_within_network": "viewers_total, or viewers_main when the total is unknown. Use "
                              "only for comparisons within one network, and say when a value "
                              "is main_only. Never compare across networks.",
    "viewers_basis": "total or main_only: which number viewers_within_network holds.",
    "total_uncertain": "TRUE when the source gives one number with no split, so it may not "
                       "include the simulcast. Say the total is uncertain whenever using "
                       "this row.",
    "audience_scope": "tv_excl_streaming_addons, or streaming_only for Netflix.",
    "measurement": "panel before 2025-09-01, then big_data_panel. Not like-for-like.",
}
GAME_COMMENTS = {
    "attendance": "NULL means unknown, not zero.",
    "dh_shared_gate": "Single-admission doubleheader: this attendance covers both games.",
    "fill_rate_raw": "attendance / capacity_api, not clamped. Can exceed 1.",
    "fill_rate": "attendance / capacity_used, which uses a checked override if one exists.",
    "over_capacity": "fill_rate_raw above 1. May be standing room or a stale API capacity.",
    "home_wins_before": "Record entering the game. The *_after columns are post-game.",
}
TABLE_COMMENT = ("Publicly reported average audience (P2+) compiled by TV Media Blog from network "
                 "press releases and trade coverage. May differ from official Nielsen data.")


def add_comments(con) -> None:
    for table in ("viewership_games", "fox_windows"):
        con.execute(f"COMMENT ON TABLE {table} IS '{TABLE_COMMENT}'")
        for col, text in VIEWER_COMMENTS.items():
            con.execute(f"COMMENT ON COLUMN {table}.{col} IS '{text}'")
    for col, text in GAME_COMMENTS.items():
        con.execute(f"COMMENT ON COLUMN games.{col} IS '{text}'")


def main() -> None:
    games, teams, totals = [], [], []
    for season in SEASONS:
        games += [game_row(g) for g in load_jsonl(season)]
        teams += [{**t, "season": season}
                  for t in json.loads((API / f"teams_{season}.json").read_text())]
        for r in json.loads((API / f"attendance_{season}.json").read_text()):
            totals.append({
                "team_id": r["team_id"], "season": season,
                "home_openings": r["openingsTotalHome"], "home_total": r["attendanceTotalHome"],
                "home_average": r["attendanceAverageHome"], "high": r.get("attendanceHigh"),
                "low": r.get("attendanceLow"),
            })

    override_rows, overrides = read_capacity_overrides()
    add_pregame_records(games)
    add_home_openers(games)
    add_shared_gates(games)
    add_fill_rates(games, overrides)

    singles, windows, window_games, stats = build_viewership(Matcher(games, teams))

    DB_PATH.unlink(missing_ok=True)
    con = duckdb.connect(str(DB_PATH))
    team_rows = [{"team_id": t["id"], "name": t["name"], "club_name": t["clubName"],
                  "abbreviation": t["abbreviation"], "league": t["league"],
                  "division": t["division"], "season": t["season"]} for t in teams]
    create_table(con, "teams", team_rows, TEAM_TYPES)
    create_table(con, "team_season_attendance", totals, ATTENDANCE_TYPES)
    create_table(con, "games", games, GAME_TYPES)
    create_table(con, "fox_windows", windows, WINDOW_TYPES)
    create_table(con, "fox_window_games", window_games, WINDOW_GAME_TYPES)
    create_table(con, "viewership_games", singles, SINGLE_TYPES)
    create_table(con, "capacity_overrides", override_rows, OVERRIDE_TYPES)
    add_comments(con)
    con.close()

    print(f"games {len(games)}, viewership_games {len(singles)}, fox_windows {len(windows)}, "
          f"fox_window_games {len(window_games)}")
    print(dict(stats))


if __name__ == "__main__":
    main()
