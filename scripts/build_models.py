"""Fit the attendance models and write their results into data/mlb.duckdb.

Run after load_duckdb.py, which rebuilds the database from scratch. build_all.py runs both.

Game level: log(attendance) on a venue_id x season term plus day/night, weekday, month,
opponent, weather and the home team's pre-game record and games back. Residuals say how a
game did against its own park's norm for that season.

Team level: log(average attendance) on win rate, metro population, the prior two years'
attendance and season. Residuals say how a team-season did against those expectations.

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
REBOUND_SHARE = 0.05  # a prior-year gap this large counts toward a possible rebound
SEASON_HIGH_SHARE = 0.90  # ceiling_game: capped, and at least this share of the season high
PRIOR_GAMES = 10      # pre-game win rate is shrunk toward .500 by this many games
FOLDS = 5
PRIOR_SEASONS = [2022, 2023]  # attendance totals only, for the team model's prior crowds
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
    " + log_prior_two_year_avg + C(season)"
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
    "expected_capped": "expected_raw, capped at the venue-season maximum for near_ceiling games and "
                       "parks with 5+ games at that maximum. Use this for comparisons.",
    "is_capped": "TRUE when expected_capped is below expected_raw. Say the value is capped.",
    "residual_capped": "attendance minus expected_capped.",
    "residual_pct_capped": "Actual over expected_capped, minus 1. 0.10 means 10% above expected.",
    "ceiling_game": "Capped or near_ceiling, and at least 90% of the park season high. Never say "
                    "this game beat or missed expected: use game_result_text instead.",
    "game_result_text": "A finished sentence about this game. Copy it word for word.",
    "exclusion_reason": "Why the game is not in the model. NULL when it is.",
    "home_games_back_before": "Games behind the division leader at the start of the day.",
}
TEAM_COLUMN_COMMENTS = {
    "residual_pct": "Average attendance against what the prior two years of crowds, win rate and "
                    "market would predict. Not a year-over-year change. A rebound after a weak year "
                    "can still show here, so give a reason from context research if there is one.",
    "yoy_change_pct": "Plain change in average attendance from the season before. Report this first.",
    "team_result_text": "A finished sentence about this team-season. Copy it word for word.",
    "prior_residual_pct": "This team residual_pct one season earlier. NULL when unavailable. "
                          "A negative value followed by a positive one suggests a rebound.",
}
TEAM_TABLE_COMMENT = (
    "Team-season average attendance against win rate, metro population and the prior two years of "
    "crowds. "
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


def game_text(g) -> str:
    """A finished sentence for a game, so its figures and labels cannot drift apart in a report."""
    lead = f"{g.away_team} at {g.home_team} on {g.official_date:%B} {g.official_date.day}, {g.season} drew {int(g.attendance):,}."
    if g.ceiling_game:
        return (f"{lead} The crowd was at or near the park's season high of "
                f"{int(g.venue_season_max):,}, and demand may have been higher.")
    label = " (capped at the park's season high)" if g.is_capped else ""
    side = "above" if g.residual_pct_capped >= 0 else "below"
    return f"{lead} That is {abs(g.residual_pct_capped):.1%} {side} the expected {g.expected_capped:,.0f}{label}."


def team_text(t) -> str:
    """A finished sentence for a team-season, in the wording the reports should use."""
    side = "up" if t.yoy_change_pct >= 0 else "down"
    text = (f"The {t.home_team} averaged {t.avg_attendance:,.0f} a game in {t.season}, {side} "
            f"{abs(t.yoy_change_pct):.1%} from {t.prior_season_avg:,.0f} the season before.")
    if pd.isna(t.residual_pct):
        return f"{text} No prediction is given for this season: {t.exclusion_reason}."
    side = "above" if t.residual_pct >= 0 else "below"
    text += (f" That is {abs(t.residual_pct):.1%} {side} the {t.expected_avg_attendance:,.0f} that "
             "the prior two years of crowds, win rate and market would predict.")
    before = t.prior_residual_pct
    if pd.notna(before) and abs(before) >= REBOUND_SHARE and before * t.residual_pct < 0:
        kind = "rebound" if t.residual_pct > 0 else "pullback"
        was = "below" if before < 0 else "above"
        text += f" Part of this may be a {kind}: the team was {abs(before):.1%} {was} its prediction the season before."
    return text


EFFECT_LABELS = {
    "C(day_night)[T.night]": "Night games, compared with day games",
    "C(month_group)[T.Mar-Apr]": "March and April games, compared with August",
    "C(month_group)[T.May]": "May games, compared with August",
    "C(month_group)[T.Jun]": "June games, compared with August",
    "C(month_group)[T.Jul]": "July games, compared with August",
    "C(month_group)[T.Sep]": "September games, compared with August",
    "indoor": "Games with a closed roof or dome, which partly reflects bad weather",
    "precipitation": "Games with rain, drizzle or snow",
    "home_opener_term": "Home openers",
}
PER_UNIT_LABELS = {  # term: (label, how much of one unit to report)
    "home_win_pct_shrunk": ("A home win rate .100 higher", 0.1),
    "home_games_back_before": ("Each game further behind the division leader", 1),
    "wind_mph": ("Each extra mph of wind", 1),
    "win_pct": ("A season win rate .100 higher", 0.1),
    "log_prior_two_year_avg": ("A prior two-year crowd 10% higher", np.log(1.1)),
    "log_population": ("A metro population 10% larger, after accounting for prior crowds", np.log(1.1)),
}


def effect_text(c) -> str | None:
    """A finished sentence for a coefficient, with no statistics jargon. None if not reportable."""
    term = c.term
    if term in PER_UNIT_LABELS:
        label, unit = PER_UNIT_LABELS[term]
        effect = np.exp(c.coef * unit) - 1
    elif term in EFFECT_LABELS:
        label, effect = EFFECT_LABELS[term], c.pct_effect
    elif term.startswith("C(weekday)"):
        label, effect = f"{term.split('T.')[1].rstrip(']')} games, compared with Friday", c.pct_effect
    elif term.startswith("opponent: "):
        label = f"Games against the {term.removeprefix('opponent: ')}, compared with the average opponent"
        effect = c.pct_effect
    else:
        return None
    clear = "" if c.p_value < 0.05 else " This is not a clear effect."
    return f"{label}: {effect:+.1%} attendance.{clear}"


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
    sample["game_result_text"] = [game_text(g) for g in sample.itertuples()]

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
    earlier = [{"team_id": r["team_id"], "season": year, "home_average": r["attendanceAverageHome"]}
               for year in PRIOR_SEASONS
               for r in json.loads((API / f"attendance_{year}.json").read_text())]
    avg = pd.concat([avg, pd.DataFrame(earlier)])
    one_back = avg.assign(season=avg["season"] + 1).rename(columns={"home_average": "prior_avg_attendance"})
    two_back = avg.assign(season=avg["season"] + 2).rename(columns={"home_average": "two_back_avg"})

    team = (team.merge(final[["team_id", "season", "win_pct"]], on=["team_id", "season"])
            .merge(one_back, on=["team_id", "season"], how="left")
            .merge(two_back, on=["team_id", "season"], how="left")
            .merge(markets, on=["team_id", "season"], how="left"))
    assert team[["prior_avg_attendance", "two_back_avg", "population"]].notna().all().all()
    # Two years, so one unusual season does not set the whole expectation.
    team["prior_two_year_avg"] = team[["prior_avg_attendance", "two_back_avg"]].mean(axis=1)
    last_year = team[["team_id", "season", "avg_attendance"]].assign(season=team["season"] + 1)
    team = team.merge(last_year.rename(columns={"avg_attendance": "last_year_avg"}),
                      on=["team_id", "season"], how="left")
    team["prior_season_avg"] = team["last_year_avg"].fillna(team["prior_avg_attendance"])
    team["yoy_change_pct"] = team["avg_attendance"] / team["prior_season_avg"] - 1

    team["exclusion_reason"] = [TEAM_EXCLUSIONS.get((t, s)) for t, s in zip(team.team_id, team.season)]
    team["in_model"] = team["exclusion_reason"].isna()
    team["shared_market"] = (team["teams_in_market"] > 1).astype(int)
    team["log_avg_attendance"] = np.log(team["avg_attendance"])
    team["log_population"] = np.log(team["population"].astype(float))
    team["log_prior_two_year_avg"] = np.log(team["prior_two_year_avg"].astype(float))

    fitted = team[team["in_model"]]
    fit = smf.ols(TEAM_FORMULA, fitted).fit(
        cov_type="cluster", cov_kwds={"groups": fitted["team_id"]})
    team.loc[fitted.index, "expected_avg_attendance"] = np.exp(fit.fittedvalues)
    team["residual_pct"] = team["avg_attendance"] / team["expected_avg_attendance"] - 1
    prior = team[["team_id", "season", "residual_pct"]].copy()
    prior["season"] += 1
    team = team.merge(prior.rename(columns={"residual_pct": "prior_residual_pct"}),
                      on=["team_id", "season"], how="left")

    team["team_result_text"] = [team_text(t) for t in team.itertuples()]

    rows = [{"model": "team", "term": n, "coef": fit.params[n], "std_err": fit.bse[n],
             "p_value": fit.pvalues[n]} for n in fit.params.index if n != "Intercept"]
    stats = {"n": int(fit.nobs), "r2": float(fit.rsquared)}
    cols = ["team_id", "home_team", "season", "avg_attendance", "model_games", "near_ceiling_games",
            "win_pct", "prior_season_avg", "yoy_change_pct", "prior_two_year_avg", "market", "population", "population_year",
            "teams_in_market", "in_model", "exclusion_reason", "expected_avg_attendance",
            "residual_pct", "prior_residual_pct", "team_result_text"]
    return team[cols], pd.DataFrame(rows), stats


def create_clean_views(con) -> None:
    """The researchers' defaults: no special events, no shared-gate doubleheader games."""
    con.execute("""
        CREATE OR REPLACE VIEW games_clean AS
        SELECT g.*, m.* EXCLUDE (game_pk, expected_capped, residual_capped, residual_pct_capped),
               -- A crowd at the ceiling has no meaningful gap from expected.
               CASE WHEN m.ceiling_game THEN NULL ELSE m.residual_pct_capped END AS residual_pct_capped,
               CASE WHEN m.ceiling_game THEN 'not scored: at or near the park season high'
                    WHEN m.in_model THEN 'scored' ELSE 'not in the model' END AS expected_status
        FROM games g JOIN game_model m USING (game_pk)
        WHERE g.special_event IS NULL AND NOT g.dh_shared_gate
          AND NOT (g.attendance_api_zero AND g.doubleheader = 'Y')""")
    con.execute("""
        CREATE OR REPLACE VIEW game_viewership_clean AS
        SELECT g.*, v.* EXCLUDE (game_pk, season)
        FROM games_clean g JOIN viewership_games v USING (game_pk)""")


def strength(r: float) -> str:
    size = abs(r)
    return "weakly" if size < 0.2 else "moderately" if size < 0.5 else "strongly"


def correlation_text(row: dict) -> str:
    """A finished sentence for one network and season, or for all networks combined."""
    where = "Comparing games on the same network" if row["network"].startswith("All") else f"On {row['network']}"
    n = row["n"]
    if row["too_few"] or row["corr_crowd_tv"] is None:
        return f"{where} in {row['season']}, there are too few games to read (n = {n})."
    raw, low, high = row["corr_crowd_tv"], row["crowd_ci_low"], row["crowd_ci_high"]
    moved = (f"crowds and national TV audiences moved together {strength(raw)}" if low > 0
             else "crowds and national TV audiences did not clearly move together")
    adjusted, a_low, a_high = row["corr_residual_tv"], row["residual_ci_low"], row["residual_ci_high"]
    link = "so there is no clear link" if a_low <= 0 <= a_high else "so a link remains"
    size = " This is a small sample." if row["small_n"] else ""
    return (f"{where} in {row['season']}, {moved} (correlation {raw:.2f}, n = {n}, range {low:.2f} to "
            f"{high:.2f}). After comparing each game with its own park's usual crowd, so popular teams "
            f"do not drive the result, the correlation is {adjusted:.2f} (range {a_low:.2f} to "
            f"{a_high:.2f}), {link}.{size}")


def correlation_row(network: str, season: int, crowd, tv, residual) -> dict:
    """Correlations with TV audience, each with a 95% range."""
    n = len(crowd)
    row = {"network": network, "season": season, "n": n, "corr_crowd_tv": None,
           "crowd_ci_low": None, "crowd_ci_high": None,
           "corr_residual_tv": None, "residual_ci_low": None, "residual_ci_high": None,
           "too_few": n < TOO_FEW_GAMES, "small_n": n < SMALL_N_GAMES}
    if n >= MIN_CORR_GAMES:
        raw = float(np.corrcoef(crowd, tv)[0, 1])
        r = float(np.corrcoef(residual, tv)[0, 1])
        margin = 1.96 / np.sqrt(n - 3)
        row.update(corr_crowd_tv=raw, crowd_ci_low=float(np.tanh(np.arctanh(raw) - margin)),
                   crowd_ci_high=float(np.tanh(np.arctanh(raw) + margin)))
        row.update(corr_residual_tv=r, residual_ci_low=float(np.tanh(np.arctanh(r) - margin)),
                   residual_ci_high=float(np.tanh(np.arctanh(r) + margin)))
    return row


def crowd_tv_table(con) -> pd.DataFrame:
    """Do crowds and TV audiences move together, before and after removing each park's norm?

    All values are in logs. Audiences are only compared within a network, because network
    reach swamps everything else.
    """
    d = con.execute("""select v.network, v.season, v.attendance, v.viewers_total, m.residual_pct_capped
                       from game_viewership_clean v join game_model m using (game_pk)
                       where v.viewers_total is not null and m.residual_pct_capped is not null""").df()
    d["crowd"] = np.log(d["attendance"].astype(float))
    d["tv"] = np.log(d["viewers_total"].astype(float))
    d["residual"] = np.log1p(d["residual_pct_capped"])
    rows = [correlation_row(network, season, x["crowd"], x["tv"], x["residual"])
            for (network, season), x in d.groupby(["network", "season"])]
    for season, x in d.groupby("season"):
        within = x[["crowd", "tv", "residual"]] - x.groupby("network")[["crowd", "tv", "residual"]].transform("mean")
        rows.append(correlation_row("All networks, within network", season,
                                    within["crowd"], within["tv"], within["residual"]))
    for row in rows:
        row["summary_text"] = correlation_text(row)
    return pd.DataFrame(rows)


def network_audience_table(con) -> pd.DataFrame:
    """Average audience per network and season, counting only broadcasts with a known total.

    Fox is counted by window, because a window is what its audience figure measures.
    """
    d = con.execute("""
        select network, season, 'games' as unit, count(*) as broadcasts, count(viewers_total) as rated,
               avg(viewers_total) as average_viewers
        from viewership_games where network <> 'Fox' group by 1, 2
        union all
        select network, season, 'windows', count(*), count(viewers_total), avg(viewers_total)
        from fox_windows group by 1, 2 order by 2, 6 desc""").df()
    d["network_text"] = [
        f"{r.network} averaged {r.average_viewers:,.0f} viewers across {r.rated} {r.unit} with known "
        f"audiences in {r.season}"
        + (f", out of {r.broadcasts} {r.unit}." if r.broadcasts > r.rated else ".")
        for r in d.itertuples()]
    return d


def tv_lookup_table(con) -> pd.DataFrame:
    """One row per nationally televised game, with a finished sentence about its audience."""
    single = con.execute("""
        select g.game_pk, g.official_date, g.season, g.home_team, g.away_team, v.network,
               v.viewers_total, v.viewers_main, v.viewers_basis, v.total_uncertain
        from viewership_games v join games g using (game_pk)""").df()
    window = con.execute("""
        select g.game_pk, g.official_date, g.season, g.home_team, g.away_team, w.coverage_pct,
               f.viewers_total, f.window_id
        from fox_window_games w join fox_windows f using (window_id) join games g using (game_pk)
        where f.n_games > 1""").df()

    rows = []
    for g in single.itertuples():
        name = f"{g.away_team} at {g.home_team}"
        if pd.notna(g.viewers_total):
            text = f"{name} drew {int(g.viewers_total):,} viewers on {g.network}."
            if g.total_uncertain:
                text += " This total is uncertain: it may not include the simulcast."
        elif pd.notna(g.viewers_main):
            text = (f"{name} drew {int(g.viewers_main):,} viewers on {g.network} alone. The simulcast "
                    "number is missing, so the total is unknown.")
        else:
            text = f"No audience figure is available for {name} on {g.network}."
        rows.append({"game_pk": g.game_pk, "official_date": g.official_date, "season": g.season,
                     "home_team": g.home_team, "away_team": g.away_team, "network": g.network,
                     "broadcast": "single game", "viewers_total": g.viewers_total, "tv_text": text})
    for g in window.itertuples():
        name = f"{g.away_team} at {g.home_team}"
        others = window[(window.window_id == g.window_id) & (window.game_pk != g.game_pk)]
        with_games = " and ".join(f"{o.away_team} at {o.home_team}" for o in others.itertuples())
        total = (f"the whole window drew {int(g.viewers_total):,} viewers" if pd.notna(g.viewers_total)
                 else "the window's total audience is not known")
        text = (f"The data cannot give a number for {name} alone. It aired in a regional Fox window "
                f"with {with_games}, and {total}.")
        if pd.notna(g.coverage_pct):
            text += f" {name} reached {int(g.coverage_pct)}% of the country."
        rows.append({"game_pk": g.game_pk, "official_date": g.official_date, "season": g.season,
                     "home_team": g.home_team, "away_team": g.away_team, "network": "Fox",
                     "broadcast": "regional window", "viewers_total": g.viewers_total, "tv_text": text})
    return pd.DataFrame(rows)


def main() -> None:
    con = duckdb.connect(str(DB_PATH))
    games = con.execute("select * from games").df()
    teams = con.execute("select team_id, season, division from teams").df()
    markets = pd.read_csv(MARKETS)

    g = prepare_games(games, teams)
    sample, game_coefs, game_stats = fit_game_model(g)
    team, team_coefs, team_stats = fit_team_model(con, sample, markets)

    keep = ["game_pk", "expected_capped",
            "is_capped", "residual_capped", "residual_pct_capped", "ceiling_game",
            "game_result_text", "venue_season_max", "near_ceiling"]
    game_model = g[["game_pk", "in_model", "exclusion_reason", "home_games_back_before"]].merge(
        sample[keep], on="game_pk", how="left")
    # Raw values can exceed what a park holds, so they sit apart from what researchers read.
    raw = sample[["game_pk", "expected_raw", "residual_raw", "residual_pct_raw"]]
    coefs = pd.concat([game_coefs, team_coefs])
    coefs["pct_effect"] = np.exp(coefs["coef"]) - 1
    coefs["effect_text"] = [effect_text(c) for c in coefs.itertuples()]
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

    for name, df in {"game_model": game_model, "game_model_raw": raw, "team_season_model": team, "team_markets": markets,
                     "model_coefficients": coefs, "model_fit": fit_rows}.items():
        write_table(name, df)
    create_clean_views(con)
    write_table("crowd_tv_residual", crowd_tv_table(con))
    write_table("game_tv_lookup", tv_lookup_table(con))
    write_table("network_audience", network_audience_table(con))

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
