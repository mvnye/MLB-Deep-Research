"""Read-only access to data/mlb.duckdb for the agents.

Every failure returns a specific message with a hint, because the model cannot see an
exception and a generic error gives it nothing to correct.
"""

import json
import math
import re
import threading
from pathlib import Path

import duckdb

DB_PATH = Path(__file__).parent / "data" / "mlb.duckdb"
MAX_ROWS = 200
MAX_CHARS = 20_000
TIMEOUT_SECONDS = 15
FIRST_SEASON, LAST_SEASON = 2024, 2026
COVERAGE = "The data covers the 2024 to 2026 regular seasons. The postseason is not included."
COVERAGE_FACTS = {
    "attendance_seasons": "2024 to 2026, regular season only",
    "viewership_starts": 2025,
    "viewership_note": "National TV only, 2025 and 2026. The 2025 tables end in early September. A "
                       "season before 2025 is not in our data, which does not mean a team was not on TV.",
}

# Kept for audit, not listed for researchers: games_clean holds what answers need.
HIDDEN_TABLES = {"game_model", "game_model_raw"}

TABLE_DOCS = {
    "games_clean": "Default for attendance work. Every played game except special events and "
                   "shared-gate doubleheader games, with the game_model columns included.",
    "game_viewership_clean": "Default for crowd vs TV work. games_clean joined to single-game "
                             "national broadcasts: one row per broadcast.",
    "crowd_tv_headline": "The fixed answer to whether crowds and TV audiences move together. Use "
                         "headline_text for that question, word for word.",
    "crowd_tv_residual": "Crowd vs TV correlations per network and season, with n and ranges.",
    "team_crowd_tv": "Team-level link: each team-season's average crowd against its average national "
                     "TV audience, single-game broadcasts, with n and a 95% range.",
    "game_tv_lookup": "One row per nationally televised regular game, including Fox windows. Special "
                      "events are left out, as in games_clean. Use this first for how many people "
                      "watched a game, and for any average or count of broadcasts.",
    "game_tv_lookup_all": "game_tv_lookup plus the special events (Tokyo Series, Little League Classic, "
                          "Speedway Classic, Field of Dreams). Use only for a question about a special "
                          "event itself.",
    "team_tv_summary": "One row per team and season for 2025 and 2026: appearances, single games, Fox "
                       "window games, special events, audiences and network mix. Use this for a "
                       "team's TV picture.",
    "team_aliases": "Finds team_id from any team name. alias_key is lower case with no punctuation, "
                    "e.g. 'as' for the A's. Filter other tables by team_id, not by name.",
    "data_coverage": "Which seasons each dataset covers.",
    "network_audience": "Average audience per network and season, counting only known audiences. "
                        "network_text is a finished sentence. Use this to compare networks.",
    "games": "One row per played regular-season game, 2024 to 2026. Key: game_pk. NULL attendance "
             "means unknown. Includes special events: use games_clean unless asked about them.",
    "teams": "One row per team per season, with league and division. Key: team_id, season.",
    "team_season_attendance": "The official season totals: MLB Stats API home_total and home_average per "
                              "team and season. Key: team_id, season. Every season total comes from here "
                              "(also in team_season_model as official_home_total), never from a sum of games.",
    "viewership_games": "One row per single-game national broadcast, 2025 and 2026, including "
                        "single-game Fox windows. Join to games on game_pk. Use for game-level "
                        "crowd vs TV comparisons.",
    "fox_windows": "One row per Fox window, 2025 and 2026. Viewers are for the whole window. "
                   "Single-game windows also appear in viewership_games: do not add the two.",
    "fox_window_games": "Games in each Fox window. coverage_pct is the share of the country that "
                        "received the game, not its share of viewers: never turn it into a "
                        "per-game viewer number.",
    "team_season_model": "Expected average attendance per team-season. Key: team_id, season.",
    "team_markets": "Metro population per team and season. Two-team metros are not split.",
    "model_coefficients": "Coefficients of both models. model is game or team.",
    "model_fit": "Fit statistics of both models.",
    "capacity_overrides": "Hand-checked capacities with a published source. Currently empty.",
}

# Sent with any result that touches these tables, so the caveat travels with the numbers.
RESULT_NOTES = {
    "crowd_tv_headline": "Copy headline_text word for word.",
    "team_crowd_tv": "Report corr_team_crowd_tv with n and ci_low to ci_high.",
    "network_audience": "Copy network_text word for word.",
    "game_tv_lookup": "When can_split is false, the data has no audience for this game alone: report "
                      "window_total as the whole window, and coverage_pct as the share of the country.",
    "team_tv_summary": "window games have no audience of their own, so they are not in the averages. "
                       "Special events (such as the Tokyo Series) are counted in special_event_games "
                       "only, and left out of every other column.",
    "team_season_model": "residual_pct is against the prediction, not a year-over-year change.",
    "model_coefficients": "Report label and effect_pct, never coef. Filter to label is not null.",
}


def _connect() -> duckdb.DuckDBPyConnection:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"{DB_PATH} is missing. Run scripts/build_all.py to create it.")
    return duckdb.connect(str(DB_PATH), read_only=True,
                          config={"enable_external_access": "false"})


def _error(message: str, hint: str) -> str:
    return json.dumps({"error": message, "hint": hint})


def _table_names(con) -> list[str]:
    rows = con.execute("""select table_name from duckdb_tables()
                          union select view_name from duckdb_views() where not internal
                          order by 1""").fetchall()
    return [r[0] for r in rows if r[0] not in HIDDEN_TABLES]


def _tables_in(query: str, names: list[str]) -> list[str]:
    return [t for t in names if re.search(rf"\b{t}\b", query, re.IGNORECASE)]


def _columns(con, table: str) -> list[str]:
    rows = con.execute("select column_name from duckdb_columns() where table_name = ?",
                       [table]).fetchall()
    return [r[0] for r in rows]


def _sql_error_hint(con, query: str, error: Exception) -> str:
    text = str(error)
    names = _table_names(con)
    if isinstance(error, duckdb.CatalogException) and "does not exist" in text:
        return f"Available tables: {', '.join(names)}. Call describe_schema for details."
    if isinstance(error, duckdb.BinderException):
        used = _tables_in(query, names) or names
        listing = "; ".join(f"{t}: {', '.join(_columns(con, t))}" for t in used)
        extra = " There is no postseason data." if re.search(
            r"game_?type|post_?season|playoff", text, re.IGNORECASE) else ""
        return f"Columns by table. {listing}.{extra}"
    if isinstance(error, duckdb.ParserException):
        return ("Fix the SQL syntax at the position shown. DuckDB uses PostgreSQL-style SQL. Write an "
                "apostrophe in a string as two: 'A''s'. For teams, use team_aliases and team_id.")
    if isinstance(error, duckdb.PermissionException):
        return "Only the tables in this database can be read. No files or URLs."
    if isinstance(error, duckdb.ConversionException):
        return ("A value does not fit the column type. team_id is a number: find it in team_aliases. "
                "Check other columns with describe_schema.")
    return "Check the query against describe_schema."


def _years_outside_data(query: str) -> list[int]:
    """Years in season filters or date literals that fall outside the data."""
    found = re.findall(r"\bseason\s*(?:(?:>=|<=|<>|!=|=|>|<)\s*\d{4}|in\s*\([\d,\s]+\)"
                       r"|between\s+\d{4}\s+and\s+\d{4})", query, re.IGNORECASE)
    years = [int(y) for chunk in found for y in re.findall(r"\b((?:19|20)\d\d)\b", chunk)]
    years += [int(y) for y in re.findall(r"'((?:19|20)\d\d)-\d\d", query)]
    return sorted({y for y in years if not FIRST_SEASON <= y <= LAST_SEASON})


def _split_and(where: str) -> list[str]:
    """Split a WHERE clause on top-level AND, keeping BETWEEN x AND y together."""
    parts, depth, start, i = [], 0, 0, 0
    in_between = False
    while i < len(where):
        ch = where[i]
        depth += (ch == "(") - (ch == ")")
        if depth == 0 and re.match(r"\bbetween\b", where[i:], re.IGNORECASE):
            in_between = True
        if depth == 0 and re.match(r"\band\b", where[i:], re.IGNORECASE) and (
                i == 0 or not where[i - 1].isalnum()):
            if in_between:
                in_between = False
            else:
                parts.append(where[start:i].strip())
                start = i + 3
        i += 1
    parts.append(where[start:].strip())
    return [p for p in parts if p]


def _empty_result_message(con, query: str) -> str:
    """Explain zero rows by testing each WHERE filter alone."""
    outside = _years_outside_data(query)
    if outside:
        return f"No rows. {outside} is outside the data. {COVERAGE}"

    match = re.search(r"\bfrom\b(.+?)\bwhere\b(.+?)(?:\bgroup\s+by\b|\border\s+by\b|\bhaving\b"
                      r"|\blimit\b|;|$)", query, re.IGNORECASE | re.DOTALL)
    simple = match and len(re.findall(r"\bselect\b", query, re.IGNORECASE)) == 1
    if not simple:
        return "No rows. Run each filter on its own to find which one matches nothing."

    source, where = match[1].strip(), match[2].strip()
    failed = []
    for clause in _split_and(where):
        try:
            count = con.execute(f"select count(*) from {source} where {clause}").fetchone()[0]
        except duckdb.Error:
            continue
        if count == 0:
            failed.append(clause)
    if not failed:
        return ("No rows. Each filter matches rows alone, but no row passes all of them: "
                f"{'; '.join(_split_and(where))}.")

    hints = []
    text = " ".join(failed).lower()
    if "team" in text or "club_name" in text:
        rows = con.execute("select distinct name from teams where season = 2026 order by 1").fetchall()
        hints.append("games.home_team and away_team use full names such as 'New York Yankees'. "
                     f"Teams: {', '.join(r[0] for r in rows)}.")
    if "season" in text and re.search(r"viewership|fox_window|tv_lookup|tv_summary", source, re.IGNORECASE):
        hints.append("Viewership covers 2025 and 2026 only. The 2025 tables end in early September. "
                     "An earlier season is not in our data, which does not mean no TV games.")
    if "network" in text:
        rows = con.execute("select distinct network from viewership_games order by 1").fetchall()
        hints.append(f"Networks: {', '.join(r[0] for r in rows)}.")
    return f"No rows. These filters matched nothing: {'; '.join(failed)}. {' '.join(hints)}".strip()


ALL_EVENT_TABLES = ("games", "game_tv_lookup_all", "viewership_games")  # these keep special events
GAME_TABLES = ("games", "games_clean", "game_tv_lookup", "game_tv_lookup_all", "game_viewership_clean",
               "viewership_games")
CLEAN_SUM = re.compile(r"\bsum\s*\(\s*(?:\w+\.)?attendance\s*\)", re.I)


# A filter or grouping that limits a sum to part of a season: a series, a month, a homestand, an opponent.
PART_OF_SEASON = re.compile(r"official_date|game_date|\bmonth\s*\(|date_trunc|strftime|dayname|\bweek|weekday|"
                            r"day_night|series|away_team|opponent|game_pk|\bbetween\b", re.I)


def season_total_error(query: str) -> str | None:
    """A whole-season sum from games_clean undercounts: it leaves out special events and some doubleheaders.

    A sum over part of a season (a series, month or homestand) is allowed.
    """
    if (CLEAN_SUM.search(query) and _tables_in(query, ["games_clean", "game_viewership_clean"])
            and not PART_OF_SEASON.search(query)):
        return _error("A season total cannot come from a sum of games_clean: it leaves out special events and "
                      "some doubleheader games, so it undercounts.",
                      "Season totals come from the official totals table: select home_total from "
                      "team_season_attendance where team_id = ... and season = ... (the same figure is "
                      "official_home_total in team_season_model). For part of a season (a "
                      "series, month or homestand), filter by official_date and sum attendance.")
    return None


def aggregate_notes(query: str) -> list[str]:
    """Rules for an average or count over games that no precomputed table has applied."""
    q = query.lower()
    if not re.search(r"\b(?:avg|sum|count|median)\s*\(", q) or not _tables_in(query, list(GAME_TABLES)):
        return []
    notes = []
    if _tables_in(query, list(ALL_EVENT_TABLES)) and "special_event" not in q:
        notes.append("This table keeps special events (such as the Tokyo Series or the Little League "
                     "Classic). For an average or count of regular games, use games_clean or game_tv_lookup.")
    if "avg(" in q.replace(" ", "") and "count(" not in q.replace(" ", ""):
        notes.append("Add count(*) so the answer can say how many games the average covers.")
    return notes


def run_sql(query: str) -> str:
    """Run one read-only SELECT and return JSON with columns, rows and any notes."""
    if not query.strip():
        return _error("The query is empty.", "Send one SELECT statement.")
    if error := season_total_error(query):
        return error
    try:
        con = _connect()
    except FileNotFoundError as e:
        return _error(str(e), "The database must be built before the tools can run.")

    timer = threading.Timer(TIMEOUT_SECONDS, con.interrupt)
    try:
        try:
            statements = con.extract_statements(query)
        except duckdb.Error as e:
            return _error(str(e), _sql_error_hint(con, query, e))
        if len(statements) != 1:
            return _error(f"Got {len(statements)} statements.", "Send one statement at a time.")
        if statements[0].type != duckdb.StatementType.SELECT:
            return _error("Only SELECT is allowed.", "This database is read-only.")

        timer.start()
        try:
            cursor = con.execute(query)
            columns = [d[0] for d in cursor.description]
            rows = cursor.fetchmany(MAX_ROWS + 1)
        except duckdb.InterruptException:
            return _error(f"The query ran past {TIMEOUT_SECONDS} seconds and was stopped.",
                          "Filter to fewer rows, or aggregate before joining.")
        except duckdb.Error as e:
            return _error(str(e), _sql_error_hint(con, query, e))
        finally:
            timer.cancel()

        notes = [RESULT_NOTES[t] for t in _tables_in(query, list(RESULT_NOTES))]
        notes = list(dict.fromkeys(notes))
        notes += aggregate_notes(query)
        outside = _years_outside_data(query)
        if rows and outside:
            notes.append(f"{outside} is outside the data. {COVERAGE}")
        # NaN is not valid JSON and crashes the model call: corr() of one row returns it.
        rows = [[None if isinstance(v, float) and not math.isfinite(v) else v for v in row]
                for row in rows]
        result = {"columns": columns, "row_count": min(len(rows), MAX_ROWS), "rows": rows[:MAX_ROWS]}
        if not rows:
            result["message"] = _empty_result_message(con, query)
        if len(rows) > MAX_ROWS:
            result["truncated"] = f"Only the first {MAX_ROWS} rows are shown. Aggregate or filter."
        if notes:
            result["notes"] = notes
        text = json.dumps(result, default=str)
        while len(text) > MAX_CHARS and len(result["rows"]) > 1:
            result["rows"] = result["rows"][: len(result["rows"]) // 2]
            result["row_count"] = len(result["rows"])
            result["truncated"] = (f"Only the first {result['row_count']} rows fit. "
                                   "Select fewer columns or aggregate.")
            text = json.dumps(result, default=str)
        return text
    finally:
        con.close()


def _rows(con, sql: str) -> list[dict]:
    cursor = con.execute(sql)
    columns = [c[0] for c in cursor.description]
    return [dict(zip(columns, r)) for r in cursor.fetchall()]


def team_seasons(team_id: int) -> list[dict]:
    """One team's actual and expected average by season, for its chart."""
    con = _connect()
    try:
        return _rows(con, f"""select home_team, season, avg_attendance, expected_avg_attendance, exclusion_reason
                from team_season_model where team_id = {int(team_id)} order by season""")
    finally:
        con.close()


def team_aliases() -> list[tuple[str, int, str]]:
    """Every name for a team: (alias_key, team_id, team). Read by code, not by the model."""
    con = _connect()
    try:
        return con.execute("select alias_key, team_id, team from team_aliases").fetchall()
    finally:
        con.close()


def _query(query: str) -> tuple[list[str], list[list]]:
    con = _connect()
    try:
        result = con.execute(query)
        return [d[0] for d in result.description], [list(r) for r in result.fetchall()]
    finally:
        con.close()


def season_figures() -> list[dict]:
    """Each team-season's official total and average, beside the cleaned games' sum and average."""
    columns, rows = _query("""select m.team_id, m.season, m.official_home_total as official_total,
            m.avg_attendance as official_avg, sum(g.attendance) as clean_total, avg(g.attendance) as clean_avg
        from team_season_model m join games_clean g on g.home_team_id = m.team_id and g.season = m.season
        group by 1, 2, 3, 4""")
    return [dict(zip(columns, r)) for r in rows]


def window_game(day: str, team_ids: list[int]) -> tuple[str, list[str], list[list]]:
    """The Fox window row for one game: the two teams on that date. day is like 'June 7 2025'."""
    from datetime import datetime
    date = datetime.strptime(day, "%B %d %Y").date()
    a, b = (int(t) for t in team_ids)
    query = (f"select official_date, away_team, home_team, coverage_pct, window_total, window_games "
             f"from game_tv_lookup where official_date = '{date}' and broadcast = 'regional window' "
             f"and ((home_team_id = {a} and away_team_id = {b}) or (home_team_id = {b} and away_team_id = {a}))")
    return (query, *_query(query))


def flagged_audiences() -> tuple[str, list[str], list[list]]:
    """Audiences that are not a plain total, with the reason. Read by code."""
    query = ("select official_date, away_team, home_team, network, audience_status, viewers_total, "
             "viewers_main, audience_note from game_tv_lookup where audience_note is not null")
    return (query, *_query(query))


def network_mix(team_id: int | None, seasons: list[int]) -> tuple[str, list[str], list[list]]:
    """National broadcasts by network and season, for a team or the league. Read by code."""
    team = f"and (home_team_id = {int(team_id)} or away_team_id = {int(team_id)}) " if team_id else ""
    query = (f"select season, network, count(*) as broadcasts from game_tv_lookup where special_event is null "
             f"and season in ({', '.join(str(int(s)) for s in seasons)}) {team}group by 1, 2 order by 1, 3 desc")
    return (query, *_query(query))


def game_drivers() -> tuple[str, list[list]]:
    """The game model's associations, as the query and its rows. Read by code for the driver chart."""
    query = ("select label, effect_pct, clear_effect from model_coefficients "
             "where model = 'game' and label is not null order by effect_pct desc")
    con = _connect()
    try:
        return query, [list(r) for r in con.execute(query).fetchall()]
    finally:
        con.close()


def crowd_tv_headline() -> str:
    """The fixed answer on crowds and TV. Read by code, so no model can reword it."""
    con = _connect()
    try:
        return con.execute("select headline_text from crowd_tv_headline").fetchone()[0]
    finally:
        con.close()


def team_crowd_tv_fact() -> dict:
    """The team-level correlation across both seasons. Read by code, not by the model."""
    con = _connect()
    try:
        return _rows(con, "select * from team_crowd_tv where scope = '2025 and 2026'")[0]
    finally:
        con.close()


def team_season_facts() -> list[dict]:
    """Facts an answer about a team-season must carry. Read by code, not by the model."""
    con = _connect()
    try:
        rows = con.execute("""select home_team, season, avg_attendance, official_home_total, record, win_pct,
                                     prior_season_avg, yoy_change_pct,
                                     expected_avg_attendance, residual_pct, prior_residual_pct,
                                     possible_rebound from team_season_model""")
        columns = [c[0] for c in rows.description]
        return [dict(zip(columns, r)) for r in rows.fetchall()]
    finally:
        con.close()


def describe_schema(table: str = "") -> str:
    """List all tables, or the columns of one table, as JSON."""
    try:
        con = _connect()
    except FileNotFoundError as e:
        return _error(str(e), "The database must be built before the tools can run.")
    try:
        names = _table_names(con)
        if not table.strip():
            return json.dumps({
                "coverage": COVERAGE,
                "coverage_facts": COVERAGE_FACTS,
                "tables": [{"table": t, "rows": con.execute(f"select count(*) from {t}").fetchone()[0],
                            "description": TABLE_DOCS.get(t, "")} for t in names],
            })
        name = table.strip().lower()
        if name not in names:
            return _error(f"There is no table named {table!r}.", f"Available tables: {', '.join(names)}.")
        columns = con.execute("""select column_name, data_type, comment from duckdb_columns()
                                 where table_name = ? order by column_index""", [name]).fetchall()
        comment = con.execute("""select comment from duckdb_tables() where table_name = ?
                                 union all select comment from duckdb_views()
                                 where view_name = ?""", [name, name]).fetchone()[0]
        return json.dumps({
            "table": name, "description": TABLE_DOCS.get(name, ""), "note": comment,
            "columns": [{"name": c, "type": t, **({"description": d} if d else {})}
                        for c, t, d in columns],
        })
    finally:
        con.close()
