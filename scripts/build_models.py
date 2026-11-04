"""Fit the attendance models and write their results into data/mlb.duckdb.

Run after load_duckdb.py, which rebuilds the database from scratch. build_all.py runs both.

Game level: log(attendance) on a venue_id x season term plus day/night, weekday, month,
opponent, weather and the home team's pre-game record and games back. Residuals say how a
game did against its own park's norm for that season.

Team level: log(average attendance) on win rate, metro population, prior-year attendance
and season. Residuals say how a team-season did against those expectations.

Tables written: game_model, team_season_model, team_markets, model_coefficients, model_fit,
crowd_tv_residual. Views written: games_clean, game_viewership_clean.

Usage: uv run python scripts/build_models.py
"""

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "mlb.duckdb"
API = ROOT / "data" / "raw" / "statsapi"
MARKETS = ROOT / "data" / "raw" / "market" / "team_markets.csv"

CEILING_FILL = 0.95   # near_ceiling: at least this share of listed capacity, or
CEILING_REPEATS = 5   # at the venue-season maximum in at least this many games
SEASON_HIGH_SHARE = 0.90  # ceiling_game: capped, and at least this share of the season high
PRIOR_GAMES = 10      # pre-game win rate is shrunk toward .500 by this many games
FOLDS = 5
MIN_CORR_GAMES = 5    # below this a correlation is not computed
TOO_FEW_GAMES = 10    # below this a correlation should not be read at all
SMALL_N_GAMES = 30    # below this a correlation is noisy: report it with its range

GAME_FORMULA = (
    "log_attendance ~ C(venue_season) + C(day_night) + C(weekday) + C(month_group)"
    " + C(opponent) + temp_open + I(temp_open ** 2) + indoor + precipitation + wind_mph"
    " + home_win_pct_shrunk + home_games_back_before + home_opener_term"
)
TEAM_FORMULA = (
    "log_avg_attendance ~ win_pct + log_population + shared_market"
    " + log_prior_avg_attendance + C(season)"
)

# Prior-year crowd is not comparable for a team in, or just back from, a temporary park.
TEAM_EXCLUSIONS = {
    (139, 2025): "Rays at Steinbrenner Field, a temporary park",
    (139, 2026): "Rays back at Tropicana Field, so the prior year is the temporary park",
    (133, 2025): "Athletics moved to a temporary park",
    (133, 2026): "Athletics in a temporary park",
}

GAME_TABLE_COMMENT = (
    "Expected attendance from a log model with a venue-season baseline. Residuals compare a "
    "game with its own park that season, not with other parks. Promotions and giveaways are "
    "not in the data, so a residual may reflect one: say so. Correlational, not causal."
)
GAME_COLUMN_COMMENTS = {
    "expected_raw": "exp(fitted log attendance), with no ceiling, so it can exceed what the park "
                    "holds. Do not show this to users. NULL when not in the model.",
    "expected_capped": "expected_raw, capped at the venue-season maximum for near_ceiling games and "
                       "parks with 5+ games at that maximum. Use this for comparisons.",
    "is_capped": "TRUE when expected_capped is below expected_raw. Say the value is capped.",
    "residual_capped": "attendance minus expected_capped.",
    "residual_pct_capped": "Actual over expected_capped, minus 1. 0.10 means 10% above expected.",
    "residual_pct_raw": "Actual over expected_raw, minus 1. Misleading for capped games.",
    "ceiling_game": "Capped or near_ceiling, and at least 90% of the park season high. Never say "
                    "this game beat or missed expected: quote expected_text instead.",
    "expected_text": "The phrase to quote word for word, without quotation marks. For a "
                     "ceiling_game it replaces any comparison with expected.",
    "exclusion_reason": "Why the game is not in the model. NULL when it is.",
    "home_games_back_before": "Games behind the division leader at the start of the day.",
}
TEAM_COLUMN_COMMENTS = {
    "residual_pct": "Average attendance against expected. Expected leans on the prior-year "
                    "crowd, so a rebound after a weak year can look like over-performance. "
                    "Say relative to the prior year, and give a reason from context research "
                    "such as roster changes.",
    "prior_residual_pct": "This team residual_pct one season earlier. NULL when unavailable. "
                          "A negative value followed by a positive one suggests a rebound.",
}
TEAM_TABLE_COMMENT = (
    "Team-season average attendance against win rate, metro population and prior-year crowd. "
    "Market population is not split in two-team metros. Correlational, not causal."
)
CROWD_TV_COMMENT = (
    "Correlation of log crowd and log national TV audience within each network and season, then "
    "after removing the park-season norm and game factors (corr_residual_tv). Excludes special "
    "events and shared-gate doubleheaders. small_n means n under 30: give the range. too_few means "
    "n under 10: do not read it. Correlational, not causal."
)


def games_back(games: pd.DataFrame, teams: pd.DataFrame) -> pd.Series:
    """Home team's games behind its division leader at the start of each game's date."""
    division = {(r.season, r.team_id): r.division for r in teams.itertuples()}
    records: dict[tuple[int, int], list[int]] = {}
    out = {}
    for (season, day), slate in games.groupby(["season", "official_date"], sort=True):
        leaders: dict[str, list[int]] = {}
        for (s, tid), (w, l) in records.items():
            if s == season:
                best = leaders.setdefault(division[(s, tid)], [w, l])
                if w - l > best[0] - best[1]:
                    leaders[division[(s, tid)]] = [w, l]
        for g in slate.itertuples():
            w, l = records.get((season, g.home_team_id), [0, 0])
            lead = leaders.get(division[(season, g.home_team_id)], [0, 0])
            out[g.game_pk] = max(((lead[0] - w) + (l - lead[1])) / 2, 0.0)
        for g in slate.itertuples():
            home_won = g.home_score > g.away_score
            for tid, won in ((g.home_team_id, home_won), (g.away_team_id, not home_won)):
                rec = records.setdefault((season, tid), [0, 0])
                rec[0 if won else 1] += 1
    return games["game_pk"].map(out)


def exclusion_reason(g) -> str | None:
    if pd.notna(g.special_event):
        return "special event"
    if g.dh_shared_gate or (g.attendance_api_zero and g.doubleheader == "Y"):
        return "single-admission doubleheader"
    if pd.isna(g.attendance):
        return "attendance unknown"
    if pd.isna(g.temp_f):
        return "temperature unknown"
    return None


def prepare_games(games: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    g = games.copy()
    shared_days = set(zip(g.loc[g.dh_shared_gate, "home_team_id"],
                          g.loc[g.dh_shared_gate, "official_date"]))
    g["exclusion_reason"] = [exclusion_reason(r) for r in g.itertuples()]
    assert all((h, d) in shared_days for h, d in zip(
        g.loc[g.attendance_api_zero & (g.doubleheader == "Y"), "home_team_id"],
        g.loc[g.attendance_api_zero & (g.doubleheader == "Y"), "official_date"]))
    g["in_model"] = g["exclusion_reason"].isna()
    g["venue_season"] = g["venue_id"].astype(str) + "_" + g["season"].astype(str)
    # One label per team id: the Athletics changed their name after 2024.
    latest = g.sort_values("season").groupby("away_team_id")["away_team"].last()
    g["opponent"] = g["away_team_id"].map(latest)
    month = pd.to_datetime(g["official_date"]).dt.month
    g["month_group"] = month.where(month > 3, 4).map(  # the few March games join April
        {4: "Mar-Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep"})
    g["indoor"] = g["weather_condition"].isin(["Roof Closed", "Dome"]).astype(int)
    g["precipitation"] = g["weather_condition"].isin(["Rain", "Drizzle", "Snow"]).astype(int)
    g["temp_open"] = (g["temp_f"] - 70) * (1 - g["indoor"])
    g["wind_mph"] = g["wind_mph"].fillna(0)
    played = g["home_wins_before"] + g["home_losses_before"]
    g["home_win_pct_shrunk"] = ((g["home_wins_before"] + PRIOR_GAMES / 2)
                                / (played + PRIOR_GAMES))
    g["home_games_back_before"] = games_back(g, teams)
    g["home_opener_term"] = g["home_opener"].astype(int)
    g["log_attendance"] = np.log(g["attendance"].astype(float))
    return g


def cross_validate(sample: pd.DataFrame) -> tuple[float, float]:
    """Held-out RMSE in log points: the model, and the venue-season average alone."""
    fold = np.random.default_rng(7).integers(0, FOLDS, len(sample))
    model_err, base_err = [], []
    for k in range(FOLDS):
        train, test = sample[fold != k], sample[fold == k]
        fit = smf.ols(GAME_FORMULA, train).fit()
        model_err.append(test["log_attendance"] - fit.predict(test))
        means = train.groupby("venue_season")["log_attendance"].mean()
        base_err.append(test["log_attendance"] - test["venue_season"].map(means))
    rmse = lambda errs: float(np.sqrt((pd.concat(errs) ** 2).mean()))
    return rmse(model_err), rmse(base_err)


def fit_game_model(g: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    sample = g[g["in_model"]].copy()
    fit = smf.ols(GAME_FORMULA, sample).fit(
        cov_type="cluster",
        cov_kwds={"groups": sample["venue_season"].astype("category").cat.codes})

    sample["fitted"] = fit.fittedvalues
    sample["expected_raw"] = np.exp(sample["fitted"])
    sample["residual_raw"] = sample["attendance"] - sample["expected_raw"]
    sample["residual_pct_raw"] = sample["attendance"] / sample["expected_raw"] - 1
    sample["venue_season_max"] = sample.groupby("venue_season")["attendance"].transform("max")
    at_max = sample["attendance"] == sample["venue_season_max"]
    repeats = at_max.groupby(sample["venue_season"]).transform("sum")
    sample["near_ceiling"] = ((sample["fill_rate_raw"] >= CEILING_FILL)
                              | (at_max & (repeats >= CEILING_REPEATS)))

    # The model has no ceiling, so expected can exceed what a capped park can hold.
    capped_park = sample["near_ceiling"] | (repeats >= CEILING_REPEATS)
    ceiling = np.minimum(sample["expected_raw"], sample["venue_season_max"])
    sample["expected_capped"] = sample["expected_raw"].where(~capped_park, ceiling)
    sample["is_capped"] = sample["expected_capped"] < sample["expected_raw"]
    sample["residual_capped"] = sample["attendance"] - sample["expected_capped"]
    sample["residual_pct_capped"] = sample["attendance"] / sample["expected_capped"] - 1

    # A crowd at or near the season high cannot show whether it beat or missed demand.
    sample["ceiling_game"] = ((sample["near_ceiling"] | sample["is_capped"])
                              & (sample["attendance"] >= SEASON_HIGH_SHARE * sample["venue_season_max"]))
    # The exact phrase to quote, so a figure and its label cannot drift apart in a report.
    sample["expected_text"] = [
        f"The crowd was at or near the park's season high of {int(top):,}, and demand may have been higher." if ceiling
        else f"{round(value):,} (capped at the park's season high)" if capped
        else f"{round(value):,}"
        for value, top, ceiling, capped in zip(sample["expected_capped"], sample["venue_season_max"],
                                               sample["ceiling_game"], sample["is_capped"])]

    within_y = sample["log_attendance"] - sample.groupby("venue_season")["log_attendance"].transform("mean")
    within_e = sample["log_attendance"] - sample["fitted"]
    cv_model, cv_base = cross_validate(sample)
    stats = {"n": int(fit.nobs), "r2": float(fit.rsquared),
             "r2_within_venue_season": float(1 - (within_e ** 2).sum() / (within_y ** 2).sum()),
             "cv_rmse_log": cv_model, "cv_rmse_log_baseline": cv_base}

    rows = []
    names = list(fit.params.index)
    for name in names:
        if "venue_season" in name or name == "Intercept" or "opponent" in name:
            continue
        rows.append({"model": "game", "term": name, "coef": fit.params[name],
                     "std_err": fit.bse[name], "p_value": fit.pvalues[name]})
    # Opponents are reported against the average opponent, not an arbitrary reference team.
    opp = [n for n in names if "opponent" in n]
    labels = sorted(sample["opponent"].unique())
    for label in labels:
        contrast = np.zeros(len(names))
        for n in opp:
            contrast[names.index(n)] = -1 / len(labels)
        match = f"C(opponent)[T.{label}]"
        if match in names:
            contrast[names.index(match)] += 1
        t = fit.t_test(contrast)
        rows.append({"model": "game", "term": f"opponent: {label}",
                     "coef": float(np.squeeze(t.effect)), "std_err": float(np.squeeze(t.sd)),
                     "p_value": float(np.squeeze(t.pvalue))})
    return sample, pd.DataFrame(rows), stats


def fit_team_model(con, sample: pd.DataFrame, markets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    team = (sample.groupby(["home_team_id", "home_team", "season"])
            .agg(avg_attendance=("attendance", "mean"), model_games=("attendance", "size"),
                 near_ceiling_games=("near_ceiling", "sum"))
            .reset_index().rename(columns={"home_team_id": "team_id"}))

    final = con.execute("""
        select team_id, season, max(wins) wins, max(losses) losses from (
          select home_team_id team_id, season, home_wins_after wins, home_losses_after losses from games
          union all select away_team_id, season, away_wins_after, away_losses_after from games)
        group by 1, 2""").df()
    final["win_pct"] = final["wins"] / (final["wins"] + final["losses"])

    avg = con.execute("select team_id, season, home_average from team_season_attendance").df()
    prior_2023 = [{"team_id": r["team_id"], "season": 2023, "home_average": r["attendanceAverageHome"]}
                  for r in json.loads((API / "attendance_2023.json").read_text())]
    avg = pd.concat([avg, pd.DataFrame(prior_2023)])
    avg["season"] = avg["season"] + 1
    avg = avg.rename(columns={"home_average": "prior_avg_attendance"})

    team = (team.merge(final[["team_id", "season", "win_pct"]], on=["team_id", "season"])
            .merge(avg, on=["team_id", "season"], how="left")
            .merge(markets, on=["team_id", "season"], how="left"))
    assert team[["prior_avg_attendance", "population"]].notna().all().all()

    team["exclusion_reason"] = [TEAM_EXCLUSIONS.get((t, s)) for t, s in zip(team.team_id, team.season)]
    team["in_model"] = team["exclusion_reason"].isna()
    team["shared_market"] = (team["teams_in_market"] > 1).astype(int)
    team["log_avg_attendance"] = np.log(team["avg_attendance"])
    team["log_population"] = np.log(team["population"].astype(float))
    team["log_prior_avg_attendance"] = np.log(team["prior_avg_attendance"].astype(float))

    fitted = team[team["in_model"]]
    fit = smf.ols(TEAM_FORMULA, fitted).fit(
        cov_type="cluster", cov_kwds={"groups": fitted["team_id"]})
    team.loc[fitted.index, "expected_avg_attendance"] = np.exp(fit.fittedvalues)
    team["residual_pct"] = team["avg_attendance"] / team["expected_avg_attendance"] - 1
    prior = team[["team_id", "season", "residual_pct"]].copy()
    prior["season"] += 1
    team = team.merge(prior.rename(columns={"residual_pct": "prior_residual_pct"}),
                      on=["team_id", "season"], how="left")

    rows = [{"model": "team", "term": n, "coef": fit.params[n], "std_err": fit.bse[n],
             "p_value": fit.pvalues[n]} for n in fit.params.index if n != "Intercept"]
    stats = {"n": int(fit.nobs), "r2": float(fit.rsquared)}
    cols = ["team_id", "home_team", "season", "avg_attendance", "model_games", "near_ceiling_games",
            "win_pct", "prior_avg_attendance", "market", "population", "population_year",
            "teams_in_market", "in_model", "exclusion_reason", "expected_avg_attendance",
            "residual_pct", "prior_residual_pct"]
    return team[cols], pd.DataFrame(rows), stats


def create_clean_views(con) -> None:
    """The researchers' defaults: no special events, no shared-gate doubleheader games."""
    con.execute("""
        CREATE OR REPLACE VIEW games_clean AS
        SELECT g.*, m.* EXCLUDE (game_pk, expected_raw, residual_raw, residual_pct_raw)
        FROM games g JOIN game_model m USING (game_pk)
        WHERE g.special_event IS NULL AND NOT g.dh_shared_gate
          AND NOT (g.attendance_api_zero AND g.doubleheader = 'Y')""")
    con.execute("""
        CREATE OR REPLACE VIEW game_viewership_clean AS
        SELECT g.*, v.* EXCLUDE (game_pk, season)
        FROM games_clean g JOIN viewership_games v USING (game_pk)""")


def correlation_row(network: str, season: int, crowd, tv, residual) -> dict:
    """Correlations with TV audience, with a 95% range for the one that removes the baseline."""
    n = len(crowd)
    row = {"network": network, "season": season, "n": n, "corr_crowd_tv": None,
           "corr_residual_tv": None, "residual_ci_low": None, "residual_ci_high": None,
           "too_few": n < TOO_FEW_GAMES, "small_n": n < SMALL_N_GAMES}
    if n >= MIN_CORR_GAMES:
        row["corr_crowd_tv"] = float(np.corrcoef(crowd, tv)[0, 1])
        r = float(np.corrcoef(residual, tv)[0, 1])
        margin = 1.96 / np.sqrt(n - 3)
        row.update(corr_residual_tv=r, residual_ci_low=float(np.tanh(np.arctanh(r) - margin)),
                   residual_ci_high=float(np.tanh(np.arctanh(r) + margin)))
    return row


def crowd_tv_table(con) -> pd.DataFrame:
    """Do crowds and TV audiences move together, before and after removing each park's norm?

    All values are in logs. Audiences are only compared within a network, because network
    reach swamps everything else.
    """
    d = con.execute("""select network, season, attendance, viewers_total, residual_pct_capped
                       from game_viewership_clean
                       where viewers_total is not null and residual_pct_capped is not null""").df()
    d["crowd"] = np.log(d["attendance"].astype(float))
    d["tv"] = np.log(d["viewers_total"].astype(float))
    d["residual"] = np.log1p(d["residual_pct_capped"])
    rows = [correlation_row(network, season, x["crowd"], x["tv"], x["residual"])
            for (network, season), x in d.groupby(["network", "season"])]
    for season, x in d.groupby("season"):
        within = x[["crowd", "tv", "residual"]] - x.groupby("network")[["crowd", "tv", "residual"]].transform("mean")
        rows.append(correlation_row("All networks, within network", season,
                                    within["crowd"], within["tv"], within["residual"]))
    return pd.DataFrame(rows)


def main() -> None:
    con = duckdb.connect(str(DB_PATH))
    games = con.execute("select * from games").df()
    teams = con.execute("select team_id, season, division from teams").df()
    markets = pd.read_csv(MARKETS)

    g = prepare_games(games, teams)
    sample, game_coefs, game_stats = fit_game_model(g)
    team, team_coefs, team_stats = fit_team_model(con, sample, markets)

    keep = ["game_pk", "expected_raw", "residual_raw", "residual_pct_raw", "expected_capped",
            "is_capped", "residual_capped", "residual_pct_capped", "ceiling_game",
            "expected_text", "venue_season_max", "near_ceiling"]
    game_model = g[["game_pk", "in_model", "exclusion_reason", "home_games_back_before"]].merge(
        sample[keep], on="game_pk", how="left")
    coefs = pd.concat([game_coefs, team_coefs])
    coefs["pct_effect"] = np.exp(coefs["coef"]) - 1
    fit_rows = pd.DataFrame([{"model": "game", **game_stats}, {"model": "team", **team_stats}])

    def write_table(name: str, df: pd.DataFrame) -> None:
        con.register("df_view", df)
        con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM df_view")
        con.unregister("df_view")
        # pandas writes missing floats as NaN. In this database, unknown is always NULL.
        floats = con.execute("""select column_name from duckdb_columns()
                                where table_name = ? and data_type = 'DOUBLE'""", [name]).fetchall()
        for (column,) in floats:
            con.execute(f"UPDATE {name} SET {column} = NULL WHERE isnan({column})")

    for name, df in {"game_model": game_model, "team_season_model": team, "team_markets": markets,
                     "model_coefficients": coefs, "model_fit": fit_rows}.items():
        write_table(name, df)
    create_clean_views(con)
    write_table("crowd_tv_residual", crowd_tv_table(con))

    con.execute(f"COMMENT ON TABLE game_model IS '{GAME_TABLE_COMMENT}'")
    con.execute(f"COMMENT ON TABLE team_season_model IS '{TEAM_TABLE_COMMENT}'")
    con.execute(f"COMMENT ON TABLE crowd_tv_residual IS '{CROWD_TV_COMMENT}'")
    for col, text in GAME_COLUMN_COMMENTS.items():
        con.execute(f"COMMENT ON COLUMN game_model.{col} IS '{text}'")
    for col, text in TEAM_COLUMN_COMMENTS.items():
        con.execute(f"COMMENT ON COLUMN team_season_model.{col} IS '{text}'")
    con.execute("COMMENT ON COLUMN model_coefficients.pct_effect IS 'exp(coef) minus 1, per one "
                "unit of the term. For win rate one unit is 1.000, so scale it down. Not "
                "meaningful for log terms in the team model.'")
    con.close()
    print("game", game_stats)
    print("team", team_stats)


if __name__ == "__main__":
    main()
