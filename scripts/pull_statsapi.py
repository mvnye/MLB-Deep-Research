"""Pull MLB Stats API data for the 2024 to 2026 regular seasons into data/raw/statsapi/.

Writes, per season:
  games_{season}.jsonl       one line per played game: {"schedule": ..., "feed": ...}
  teams_{season}.json        team id, names, league, division
  attendance_{season}.json   the API's own season totals, one record per team

For 2022 and 2023 only teams and attendance are pulled: the team model uses them as prior crowds.

Safe to re-run: games already in the jsonl file are skipped.

Usage: uv run python scripts/pull_statsapi.py [--seasons 2024 2025 2026]
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

BASE = "https://statsapi.mlb.com"
OUT = Path(__file__).resolve().parent.parent / "data" / "raw" / "statsapi"
WORKERS = 8
PRIOR_SEASONS = [2022, 2023]

# Status values that mean the game was actually played. Do not use abstractGameState:
# it reads "Final" for postponed and cancelled games too.
PLAYED = {"Final", "Completed Early"}

# feed/live is about 800 KB per game; this field filter cuts it to under 1 KB.
FEED_FIELDS = ",".join([
    "gamePk", "gameData", "game", "pk", "type", "doubleHeader", "gameNumber", "season",
    "datetime", "dateTime", "officialDate", "originalDate", "resumeDate", "resumedFrom",
    "dayNight", "time", "ampm", "status", "detailedState", "statusCode", "reason",
    "venue", "id", "name", "fieldInfo", "capacity", "roofType", "turfType",
    "location", "city", "state", "timeZone", "tz",
    "gameInfo", "attendance", "firstPitch", "gameDurationMinutes",
    "weather", "condition", "temp", "wind",
    "probablePitchers", "home", "away", "fullName",
])


def get_json(client: httpx.Client, path: str, params: dict | None = None) -> dict:
    """GET with retries on rate limits, server errors and network errors."""
    for attempt in range(5):
        try:
            r = client.get(BASE + path, params=params)
            if r.status_code == 429 or r.status_code >= 500:
                raise httpx.HTTPStatusError("retryable", request=r.request, response=r)
            r.raise_for_status()
            return r.json()
        except (httpx.TransportError, httpx.HTTPStatusError):
            if attempt == 4:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def pull_schedule(client: httpx.Client, season: int) -> list[dict]:
    """Return one schedule row per played regular-season game.

    A postponed game appears twice under the same gamePk (original date as Postponed,
    makeup date as Final), so keep only the played row.
    """
    data = get_json(client, "/api/v1/schedule", {
        "sportId": 1, "gameType": "R",
        "startDate": f"{season}-03-01", "endDate": f"{season}-11-15",
    })
    rows = [g for d in data["dates"] for g in d["games"]]
    played: dict[int, dict] = {}
    for g in rows:
        if g["status"]["detailedState"] in PLAYED:
            g.pop("content", None)
            played[g["gamePk"]] = g
    skipped = {g["gamePk"] for g in rows} - set(played)
    print(f"{season}: {len(rows)} schedule rows, {len(played)} played, "
          f"{len(skipped)} never played {sorted(skipped)}")
    return list(played.values())


def pull_feed(client: httpx.Client, game_pk: int) -> dict:
    return get_json(client, f"/api/v1.1/game/{game_pk}/feed/live", {"fields": FEED_FIELDS})


def pull_games(client: httpx.Client, season: int) -> None:
    path = OUT / f"games_{season}.jsonl"
    done = set()
    if path.exists():
        with path.open() as f:
            done = {json.loads(line)["schedule"]["gamePk"] for line in f}

    todo = [g for g in pull_schedule(client, season) if g["gamePk"] not in done]
    print(f"{season}: {len(done)} already pulled, {len(todo)} to fetch")

    failed = []
    with path.open("a") as f, ThreadPoolExecutor(WORKERS) as pool:
        futures = {pool.submit(pull_feed, client, g["gamePk"]): g for g in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            g = futures[fut]
            try:
                f.write(json.dumps({"schedule": g, "feed": fut.result()}) + "\n")
            except httpx.HTTPError as e:
                failed.append((g["gamePk"], str(e)))
            if i % 500 == 0:
                print(f"{season}: {i}/{len(todo)}", flush=True)
    if failed:
        sys.exit(f"{season}: {len(failed)} feeds failed, re-run to retry: {failed[:5]}")


def pull_teams(client: httpx.Client, season: int) -> list[dict]:
    teams = get_json(client, "/api/v1/teams", {"sportId": 1, "season": season})["teams"]
    keep = ["id", "name", "teamName", "abbreviation", "locationName", "franchiseName", "clubName"]
    rows = [{**{k: t.get(k) for k in keep},
             "league": t["league"]["name"], "division": t["division"]["name"]} for t in teams]
    (OUT / f"teams_{season}.json").write_text(json.dumps(rows, indent=1))
    return rows


def pull_attendance(client: httpx.Client, season: int, teams: list[dict]) -> None:
    rows = []
    for t in teams:
        data = get_json(client, "/api/v1/attendance", {"teamId": t["id"], "season": season})
        rec = [r for r in data["records"] if r["gameType"]["id"] == "R" and r["year"] == str(season)]
        if len(rec) != 1:
            sys.exit(f"{season} {t['name']}: expected 1 regular-season record, got {len(rec)}")
        rows.append({"team_id": t["id"], "season": season, **rec[0]})
    (OUT / f"attendance_{season}.json").write_text(json.dumps(rows, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", nargs="+", type=int, default=[2024, 2025, 2026])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    limits = httpx.Limits(max_connections=WORKERS)
    headers = {"User-Agent": "mlb-deep-research-class-project"}
    with httpx.Client(timeout=30, limits=limits, headers=headers) as client:
        for season in PRIOR_SEASONS:
            pull_attendance(client, season, pull_teams(client, season))
        for season in args.seasons:
            teams = pull_teams(client, season)
            pull_attendance(client, season, teams)
            pull_games(client, season)


if __name__ == "__main__":
    main()
