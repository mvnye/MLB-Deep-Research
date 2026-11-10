# MLB-Deep-Research

A multi-agent deep research app on MLB attendance. It explores what drives attendance, why specific teams and games over- or under-perform against a regression baseline, and whether smaller crowds went with smaller national TV audiences in 2025 and 2026. A Lead sends parallel Researchers to query a DuckDB database and read Wikipedia, goes back for gaps, and a Writer produces a footnoted answer, with a chart when the result suits one.

**Target user:** a sports media or sponsorship seller preparing team pitches.

**Live app:** https://mlb-deep-research-git-636637173930.europe-west1.run.app (restricted to columbia.edu accounts).

## Data sources

- [MLB Stats API](https://statsapi.mlb.com): 2024 to 2026 regular season games, attendance, weather, venues, and season totals back to 2022. Postseason is out of scope.
- TV Media Blog viewership archives: [2025](https://tvmediablog.substack.com/p/2025-mlb-season-viewership-database) and [2026](https://tvmediablog.substack.com/p/2026-mlb-viewership-archive), saved as CSVs in `data/raw/`.
- Wikipedia team-season articles, read live through the public MediaWiki API.
- Metro population, described under Market size data below.

## Loading the data and running the app

`data/mlb.duckdb` and the raw files are in the repo, so the app runs without a data step. Vertex AI access comes from gcloud credentials.

```bash
uv sync
uv run uvicorn app:app --port 8080
```

To rebuild from scratch, run `uv run python scripts/pull_statsapi.py`, then `uv run python scripts/pull_market.py`, then `uv run python scripts/build_all.py`.

## Sample questions

The expected answer is what the data says, so the agent can be checked against it. The agent answer is its real output from a saved run on November 5, 2026, unedited. Wording varies between runs, but the facts should hold. The no-tools baseline is the same model with no tools and no system prompt.

To test the agent against 37 questions with required facts, run `uv run python scripts/eval_agent.py`.

### 1. Did games with smaller crowds coincide with smaller national TV audiences in 2025 and 2026?

**Expected answer:** Yes on the raw numbers, but there is no clear link after adjusting. Comparing games on the same network, smaller crowds tended to come with smaller TV audiences, a moderate correlation of 0.34 in 2025 (n = 82, range 0.14 to 0.52) and 0.47 in 2026 (n = 83, range 0.29 to 0.62), where 0 means no link and 1 means a perfect match. Adjusting compares each game with its own park's usual crowd, so popular teams do not drive the result. After that, the correlation falls to 0.16 in 2025 (range -0.05 to 0.37) and 0.14 in 2026 (range -0.08 to 0.34). Both ranges include zero, so a crowd below a park's usual level did not clearly mean a smaller TV audience. At the team level, a team-season's average crowd and its average national TV audience correlate at 0.51 across 2025 and 2026 (n = 56, range 0.29 to 0.68). That is consistent with team-level factors such as popularity and market size driving both, though the data does not prove the cause.

**Source:** `crowd_tv_headline`, built from `crowd_tv_residual`, and `team_crowd_tv`.

**Agent answer:**

> Yes, games with smaller crowds tended to have smaller national TV audiences, but a crowd that was small for its own park did not clearly mean a smaller audience[1]. Comparing games on the same network, the two moved together (correlation 0.34 in 2025, n = 82, range 0.14 to 0.52; 0.47 in 2026, n = 83, range 0.29 to 0.62)[1]. After comparing each game with its own park's usual crowd, the link mostly disappears (0.16 in 2025, range -0.05 to 0.37; 0.14 in 2026, range -0.08 to 0.34; both ranges include zero)[1]. Across team-seasons, teams with bigger average crowds also had bigger average national TV audiences (correlation 0.51, n = 56, range 0.29 to 0.68), which is consistent with team-level factors such as popularity and market size driving both, though the data does not prove the cause[2].
>
> *Viewership figures are publicly reported and may differ from official Nielsen data. They are the national TV audience, excluding streaming add-ons. Nielsen changed its method on September 1, 2025, so 2025 and 2026 figures are not like-for-like.*

**No-tools baseline:**

> I do not have access to real-time data or statistics for 2025 and 2026. Therefore, I cannot provide information on whether smaller crowd sizes coincided with smaller national TV audiences for games during those years.

### 2. Why did the Mets over-perform on attendance in 2025?

**Expected answer:** The Mets averaged 39,316 fans a game in 2025 (MLB's official season average), up 33.3% from 29,485 in 2024. That is 21.4% above the 32,383 that their prior two years of crowds, win rate, and market would predict. They were 18.6% below that prediction in 2024, so part of this may be a rebound. The rise coincided with signing Juan Soto, though the data cannot show the signing caused it.

**Source:** `team_season_model`, and the Wikipedia article [2025 New York Mets season](https://en.wikipedia.org/wiki/2025_New_York_Mets_season).

**Agent answer:**

> The New York Mets averaged 39,316 fans per game in 2025, which was 21.4% higher than what their prior two years of crowds, win rate, and market would predict, representing a 33.3% increase from the season before as a rebound from a down 2024[1]. This 2025 average attendance was higher than the 2024 average of 29,485 fans by 9,831 fans, with 94.9% of capacity in 2025 compared to 69.8% of capacity in 2024, a gap of 25.1 percentage points[1]. This surge in crowds coincided with several major events, beginning with the historic signing of outfielder Juan Soto to a 15-year, $765 million contract in December 2024[2]. On the field, the Mets held the best record in MLB at 45–24 through June 12, and on July 19, the team retired former captain David Wright's number 5 and inducted him into the Mets Hall of Fame during a pre-game ceremony at Citi Field[2].

**No-tools baseline:**

> I do not have access to real-time information or events after January 2025, so I cannot provide information on the Mets' 2025 attendance or the reasons for their performance.

### 3. How many people watched the Yankees vs Red Sox game on Fox on June 7, 2025?

**Expected answer:** The data cannot give a number for that game alone. It aired in a regional Fox window alongside Padres/Brewers, and the whole window drew 3,094,000 viewers. Yankees/Red Sox reached 96% of the country.

**Source:** `fox_windows` and `fox_window_games`.

**Agent answer:**

> The Yankees vs Red Sox game on June 7, 2025, reached 96% of the country, but the data has no viewership number for that game alone[1]. The entire regional Fox window, which also featured the San Diego Padres at the Milwaukee Brewers, drew 3,094,000 viewers[1].
>
> *Viewership figures are publicly reported and may differ from official Nielsen data. They are the national TV audience, excluding streaming add-ons.*

**No-tools baseline:**

> I cannot provide the viewership numbers for the Yankees vs. Red Sox game on June 7, 2025, as this date is in the future and the event has not yet occurred.

## Viewership figures

Viewership figures are publicly reported average audience (P2+), compiled by TV Media Blog from network press releases and trade coverage (Sports Business Journal, Sports Media Watch). They may differ from official Nielsen data. `viewers_total` is the national TV audience, excluding streaming add-ons (Peacock/Adobe, HBO Max): the main network plus TV simulcasts (Deportes, ESPN2, truTV). Netflix games are streaming-only.

## Using the deployed app

Answers can take a few minutes. Conversations are kept in memory, so sessions reset when the service restarts.

## Market size data

Team market size uses metro population, one figure per team and season:

- US teams: [US Census Bureau Vintage 2025 metro population estimates](https://www2.census.gov/programs-surveys/popest/datasets/2020-2025/metro/totals/cbsa-est2025-alldata.csv).
- Blue Jays: [Statistics Canada table 17-10-0148-01](https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1710014801), because the US Census does not cover Toronto.

Estimates end at 2025, so 2026 uses the 2025 figure. Two-team metros are not split between teams. The Athletics use San Francisco-Oakland for 2024 and Sacramento for 2025 and 2026.

To load it, run `uv run python scripts/pull_market.py`, then `uv run python scripts/build_all.py`.
