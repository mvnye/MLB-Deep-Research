"""Checks on the built database: known values and the rules the tables must follow. No model calls."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import data_tools  # noqa: E402
from tests.runner import run_tests  # noqa: E402


def one(sql: str):
    con = data_tools._connect()  # the tools' own settings, so both can open the file in one process
    try:
        return con.execute(sql).fetchone()
    finally:
        con.close()


def test_rays_known_game():
    assert one("select attendance, capacity_used from games_clean "
               "where home_team_id = 139 and official_date = '2025-05-01'") == (8794, 11026)


def test_rays_official_total():
    assert one("select official_home_total from team_season_model where team_id = 139 and season = 2025") == (786750,)


def test_season_figures_are_official():
    """Season averages and totals in the team model come from the Stats API season totals."""
    bad = one("""select count(*) from team_season_model m join team_season_attendance a using (team_id, season)
                 where m.avg_attendance != a.home_average or m.official_home_total != a.home_total""")
    assert bad == (0,), f"{bad[0]} team-seasons differ from the official figures"
    assert one("select official_home_total from team_season_model where team_id = 147 and season = 2024") == (3309838,)


def test_special_events_left_out_of_team_tv_summary():
    """Cubs 2025: the two Tokyo Series games are counted as special events and nowhere else."""
    row = one("""select special_event_games, single_games, single_games_with_known_audience,
                        avg_single_game_viewers, network_mix
                 from team_tv_summary where team_id = 112 and season = 2025""")
    assert row[:4] == (2, 10, 9, 1126444), row
    assert "FS1 2" in row[4], row[4]


def test_team_tv_averages_match_regular_games():
    """Every team's average is the plain average of its regular-season single-game audiences."""
    bad = one("""with apps as (
                     select home_team_id as team_id, season, viewers_total, special_event from game_tv_lookup
                     union all
                     select away_team_id, season, viewers_total, special_event from game_tv_lookup),
                 expected as (select team_id, season, round(avg(viewers_total)) as avg_viewers
                              from apps where special_event is null group by 1, 2)
                 select count(*) from team_tv_summary s join expected e using (team_id, season)
                 where s.avg_single_game_viewers is distinct from e.avg_viewers""")
    assert bad == (0,), f"{bad[0]} team-seasons have a different average"


def test_white_sox_had_no_national_broadcasts_2025():
    assert one("select appearances from team_tv_summary where team_id = 145 and season = 2025") == (0,)


def test_unknown_audiences_are_null_not_zero():
    assert one("select count(*) from game_viewership_clean where viewers_total = 0") == (0,)
    assert one("select count(*) from games_clean where attendance = 0") == (0,)


def test_default_tv_table_has_no_special_events():
    assert one("select count(*) from game_tv_lookup where special_event is not null") == (0,)
    assert one("select count(*) from game_tv_lookup_all where special_event is not null") == (6,)
    # ESPN Sundays, 2025: the Little League Classic is not in the default average.
    assert one("""select count(*), round(avg(viewers_total)) from game_tv_lookup
                  where season = 2025 and network = 'ESPN' and dayname(official_date) = 'Sunday'""") == (22, 1786136)


def sql(query: str) -> dict:
    import json
    return json.loads(data_tools.run_sql(query))


def test_whole_season_sum_of_clean_games_is_refused():
    """Yankees 2024: games_clean sums to 3,227,098, the official total is 3,309,838."""
    for query in ("select sum(attendance) from games_clean where home_team_id = 147 and season = 2024",
                  "select home_team, sum(attendance) from games_clean where season = 2024 group by 1"):
        result = sql(query)
        assert "team_season_attendance" in result.get("hint", ""), result
    assert sql("select home_total from team_season_attendance where team_id = 147 and season = 2024")["rows"] == [[3309838]]


def test_part_of_season_sums_are_allowed():
    month = sql("select month(official_date) as m, sum(attendance) from games_clean "
                "where home_team_id = 147 and season = 2024 group by 1 order by 1")
    series = sql("select sum(attendance) from games_clean where home_team_id = 147 and "
                 "away_team = 'Boston Red Sox' and official_date between '2024-07-05' and '2024-07-07'")
    homestand = sql("select sum(attendance) from games_clean where home_team_id = 147 "
                    "and official_date between '2024-06-04' and '2024-06-13'")
    for result in (month, series, homestand):
        assert "error" not in result and result["rows"], result


def test_official_totals_table_is_listed():
    assert "team_season_attendance" in data_tools.describe_schema("")


if __name__ == "__main__":
    run_tests(globals())
