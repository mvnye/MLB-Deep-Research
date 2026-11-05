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

# Kept for audit, not listed for researchers: games_clean holds what answers need.
HIDDEN_TABLES = {"game_model", "game_model_raw"}

TABLE_DOCS = {
    "games_clean": "Default for attendance work. Every played game except special events and "
                   "shared-gate doubleheader games, with the game_model columns included.",
    "game_viewership_clean": "Default for crowd vs TV work. games_clean joined to single-game "
                             "national broadcasts: one row per broadcast.",
    "crowd_tv_residual": "Crowd vs TV correlations per network and season. summary_text is a finished "
                         "sentence. Use this first for whether crowds and TV audiences move together.",
    "network_audience": "Average audience per network and season, counting only known audiences. "
                        "network_text is a finished sentence. Use this to compare networks.",
    "game_tv_lookup": "One row per nationally televised game. tv_text is a finished sentence about its "
                      "audience. Use this first for how many people watched a game.",
    "games": "One row per played regular-season game, 2024 to 2026. Key: game_pk. NULL attendance "
             "means unknown. Includes special events: use games_clean unless asked about them.",
    "teams": "One row per team per season, with league and division. Key: team_id, season.",
    "team_season_attendance": "MLB season attendance totals per team. Key: team_id, season.",
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
    "games_clean": "Copy game_result_text word for word for a single game.",
    "game_viewership_clean": "Copy game_result_text word for word for a single game. Compare viewers "
                             "only within one network.",
    "game_tv_lookup": "Copy tv_text word for word.",
    "network_audience": "Copy network_text word for word.",
    "crowd_tv_residual": "Copy summary_text word for word. Use the rows named 'All networks, within "
                         "network' unless the task names a network.",
    "team_season_model": "Copy team_result_text word for word.",
    "model_coefficients": "Copy effect_text word for word. Never report coef.",
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
        return "Fix the SQL syntax at the position shown. DuckDB uses PostgreSQL-style SQL."
    if isinstance(error, duckdb.PermissionException):
        return "Only the tables in this database can be read. No files or URLs."
    if isinstance(error, duckdb.ConversionException):
        return "A value does not fit the column type. Check it with describe_schema."
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
    if "season" in text and re.search(r"viewership|fox_window", source, re.IGNORECASE):
        hints.append("Viewership covers 2025 and 2026 only. The 2025 tables end in early September.")
    if "network" in text:
        rows = con.execute("select distinct network from viewership_games order by 1").fetchall()
        hints.append(f"Networks: {', '.join(r[0] for r in rows)}.")
    return f"No rows. These filters matched nothing: {'; '.join(failed)}. {' '.join(hints)}".strip()


def run_sql(query: str) -> str:
    """Run one read-only SELECT and return JSON with columns, rows and any notes."""
    if not query.strip():
        return _error("The query is empty.", "Send one SELECT statement.")
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
