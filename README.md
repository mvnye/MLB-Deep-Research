# MLB-Deep-Research

A multi-agent deep research app on MLB attendance. It explores what drives attendance, why specific teams and games over- or under-perform against a regression baseline, and whether smaller crowds went with smaller national TV audiences in 2025 and 2026. A Lead sends parallel Researchers to query a DuckDB database and read Wikipedia, goes back for gaps, and a Writer produces a footnoted answer, with a chart when the result suits one. Code fills in every number the Writer cites from the researchers' query results, and checks the answer before it is shown.

**Target user:** a sports media or sponsorship seller preparing team pitches.

**Live app:** https://mlb-deep-research-git-636637173930.europe-west1.run.app (restricted to columbia.edu accounts).

## Data sources

- [MLB Stats API](https://statsapi.mlb.com): 2024 to 2026 regular season games (`/api/v1/schedule?sportId=1&gameType=R`), per-game attendance, weather, venue and capacity (`/api/v1.1/game/{gamePk}/feed/live`), and official season totals (`/api/v1/attendance`), back to 2022 for the model's prior years. Postseason is out of scope.
- TV Media Blog viewership archives: [2025](https://tvmediablog.substack.com/p/2025-mlb-season-viewership-database) and [2026](https://tvmediablog.substack.com/p/2026-mlb-viewership-archive), saved as CSVs in `data/raw/`. The 2025 tables end in early September.
- Wikipedia team-season articles, read live through the public MediaWiki API.
- Metro population, described under Market size data below.

Two defaults to know when querying:

- **Official totals.** Season totals and averages come from `team_season_attendance` (the Stats API's `home_total` and `home_average`), also shown as `official_home_total` in `team_season_model`. Summing `games_clean` for a whole season gives a different number (Yankees 2024: 3,227,098 vs the official 3,309,838), so `run_sql` refuses that query and points to the official table. Sums for part of a season are allowed.
- **Special events are left out.** The Tokyo Series, Speedway Classic, Little League Classic and Field of Dreams games are flagged and excluded from `games_clean`, `game_tv_lookup` and the team TV averages in `team_tv_summary`. `game_tv_lookup_all` keeps them, for questions about a special event itself.

## Loading the data and running the app

`data/mlb.duckdb` and the raw files are in the repo, so the app runs without a data step. Vertex AI access comes from gcloud credentials.

```bash
uv sync
uv run uvicorn app:app --port 8080
```

To rebuild from scratch, run `uv run python scripts/pull_statsapi.py`, then `uv run python scripts/pull_market.py`, then `uv run python scripts/build_all.py`. `build_all.py` alone rebuilds `data/mlb.duckdb` from the raw files.

## Sample questions

These are the four sample questions on the app's front page. The expected answer is what the data says, rechecked against `data/mlb.duckdb`. The agent answer is its real output from `POST /chat`, unedited except that the footnote list is left out (the full responses, with sources and every tool call, are in `samples/sample_1.json` to `samples/sample_4.json`). They were run on November 10, 2026 against the app running locally, because the live site sits behind Google sign-in (IAP), which a script cannot pass: samples 2 and 4 on `main` (ded95f7), samples 1 and 3 and the A's viewership question on commit 757ffc5. Wording varies between runs, but the facts should hold. The no-tools baseline is the same model (`gemini-3.5-flash`) with no tools and no system prompt, from `scripts/baselines.py`, saved in `samples/baseline_1.md` to `samples/baseline_4.md`.

To test the agent against 40 questions with required facts, run `uv run python scripts/eval_agent.py`.

### 1. Did games with smaller crowds coincide with smaller national TV audiences in 2025 and 2026?

**Expected answer:** Yes on the raw numbers, but there is no clear link after adjusting. Comparing games on the same network, smaller crowds tended to come with smaller TV audiences: correlation 0.34 in 2025 (n = 82, range 0.14 to 0.52) and 0.47 in 2026 (n = 83, range 0.29 to 0.62), where 0 means no link and 1 means a perfect match. After comparing each game with its own park's usual crowd, the correlation falls to 0.16 in 2025 (range -0.05 to 0.37) and 0.14 in 2026 (range -0.08 to 0.34). Both ranges include zero. At the team level, teams with higher average home attendance also tended to have larger average national TV audiences (correlation 0.51, n = 56, range 0.29 to 0.68). This is consistent with team-level factors such as popularity and market size, but the data does not show that one causes the other.

**Source:** `crowd_tv_headline` (built from `crowd_tv_residual`) and `team_crowd_tv`.

**Agent answer:**

> Yes, games with smaller crowds tended to have smaller national TV audiences, but a crowd that was small for its own park did not clearly mean a smaller audience[1]. Comparing games on the same network, the two moved together (correlation 0.34 in 2025, n = 82, range 0.14 to 0.52; 0.47 in 2026, n = 83, range 0.29 to 0.62)[1]. After comparing each game with its own park's usual crowd, the link mostly disappears (0.16 in 2025, range -0.05 to 0.37; 0.14 in 2026, range -0.08 to 0.34; both ranges include zero)[1]. Across team-seasons, teams with higher average home attendance also tended to have larger average national TV audiences (correlation 0.51, n = 56, range 0.29 to 0.68)[2]. This is consistent with team-level factors such as popularity and market size, but the data does not show that one causes the other[2].
>
> *Viewership figures are publicly reported and may differ from official Nielsen data. They are the national TV audience, excluding streaming add-ons. Nielsen changed its method on September 1, 2025, so 2025 and 2026 figures are not like-for-like.*

**No-tools baseline:**

> I do not have access to real-time data or statistics for 2025 and 2026. Therefore, I cannot provide information on whether smaller crowd sizes coincided with smaller national TV audiences for games during those years.

### 2. Why did the Mets over-perform on attendance in 2025?

**Expected answer:** The Mets averaged 39,316 fans a game in 2025 (MLB's official season average), up 33.3% from 29,485 in 2024. That is 21.4% above the 32,383 that their prior two years of crowds, win rate, and market would predict. They were 18.6% below that prediction in 2024, so part of this may be a rebound. The rise coincided with signing Juan Soto in December 2024, though the data cannot show the signing caused it.

**Source:** `team_season_model` (`SELECT * FROM team_season_model WHERE team_id = 121 AND season = 2025`), and the Wikipedia article [2025 New York Mets season](https://en.wikipedia.org/wiki/2025_New_York_Mets_season).

**Agent answer:**

> The New York Mets averaged 39,316 fans a game in 2025, 21.4% above the 32,383 fans their prior two years of crowds, win rate, and market would predict, up 33.3% from 29,485 the season before, and part of this may be a rebound from 18.6% below prediction[1]. The Mets drew 94.9% of capacity, with game-by-game crowds averaging 1,872 fans more than expected[2][3]. This surge coincided with the blockbuster signing of outfielder Juan Soto to a record 15-year, $765 million contract in December 2024, and the re-signing of first baseman Pete Alonso in February 2025[4]. On the field, the Mets opened the season with an MLB-best 45–24 record through June 12[4].

**No-tools baseline:**

> I do not have access to real-time information or events after January 2025, so I cannot provide information on the Mets' 2025 attendance or the reasons for their performance.

### 3. Compare the Cubs and White Sox on attendance and national TV audiences in 2025.

**Expected answer:** The Cubs averaged 37,259 fans a game in 2025 (3,017,983 in all, 89.9% of capacity). The White Sox averaged 18,072 (1,445,738, 45.0% of capacity), 19,187 fewer a game. The Cubs had 15 national TV appearances in the 2025 data (10 single games and 5 in Fox regional windows; ESPN 5, Fox 5, TBS 3, FS1 2). Their 9 single games with a known audience averaged 1,126,444 viewers, with a high of 2,041,000. The White Sox had no national TV broadcasts in the 2025 data, which ends in early September.

**Source:** `team_season_model` and `team_season_attendance` (official averages and totals), `games_clean` (fill), `team_tv_summary` (`SELECT * FROM team_tv_summary WHERE season = 2025 AND team_id IN (112, 145)`).

**Agent answer:**

> The Chicago Cubs dominated the Chicago White Sox in both ticket sales and media presence in 2025, drawing 1,572,245 more fans at home and securing the city's only national television coverage[1]. The Cubs drew 3,017,983 total fans, averaging 37,259 fans per game and drew 89.9% of capacity[2][3]. In contrast, the White Sox drew 1,445,738 total fans, averaging 18,072 fans per game and drew 45.0% of capacity[2][3]. This left the Cubs 19,187 fans per game ahead of their crosstown rivals, with a fill-rate gap of 44.9 percentage points[1]. On television, the Cubs had 15 national TV appearances and 10 single-game broadcasts, averaging 1,126,444 viewers per broadcast, while the White Sox had no national TV broadcasts in the 2025 data[4].
>
> *Viewership figures are publicly reported and may differ from official Nielsen data. They are the national TV audience, excluding streaming add-ons.*

**No-tools baseline** (shortened; full text in `samples/baseline_3.md`):

> Because my knowledge cutoff is in January 2025, actual attendance figures and national TV ratings for the 2025 Major League Baseball season are not yet available. [...] In 2024, the Cubs drew **2,907,591 fans** (averaging 35,896 per game) [...] In 2024, they [the White Sox] drew just **1,380,183 fans** (averaging 17,039 per game) [...] the Cubs are expected to vastly outpace the White Sox in both stadium attendance and national television viewership [...]

The baseline gives no 2025 figures, and its 2024 figures are close to but not the official ones (Cubs 2,909,794 and 35,923; White Sox 1,380,733 and 17,932).

### 4. Why did Rays attendance drop in 2025?

**Expected answer:** The Rays averaged 9,713 fans a game in 2025 (786,750 in all), down 41.2% from 16,515 in 2024. Hurricane Milton damaged Tropicana Field, so they played 2025 at George M. Steinbrenner Field, a temporary park with 11,026 seats instead of 25,025. Because of the smaller park, they drew 88.1% of capacity, up from 66.0% in 2024. The model gives no expected attendance for a season in a temporary park, so the data cannot say whether 2025 beat expectations.

**Source:** `team_season_model` (`WHERE team_id = 139 AND season IN (2024, 2025)`), `games_clean` (venue, capacity and fill), and the Wikipedia article [2025 Tampa Bay Rays season](https://en.wikipedia.org/wiki/2025_Tampa_Bay_Rays_season).

**Agent answer:**

> The Tampa Bay Rays averaged 9,713 fans a game in 2025, a decrease of 41.2% from the season before, though expected attendance based on prior crowds, win rate, and market size was unavailable because they played in a temporary ballpark[1]. This drop coincided with major stadium changes: Hurricane Milton heavily damaged Tropicana Field, rendering it unusable for the entire 2025 season[2]. Consequently, the Rays played their 2025 home games at George M. Steinbrenner Field in Tampa[2]. While total attendance fell, the smaller venue meant the Rays drew 88.1% of capacity in 2025, which was 22.1 percentage points above their 2024 mark of 66.0% of capacity[3]. On the field, the team spent most of the season above .500 before finishing with a losing record[2].

**No-tools baseline** (shortened; full text in `samples/baseline_4.md`):

> The drop in the Tampa Bay Rays' attendance for the 2025 season is primarily due to a forced relocation to a much smaller temporary venue following severe stadium damage. [...] Steinbrenner Field (the Spring Training home of the New York Yankees) has a capacity of only about **11,000 fans**. [...] Playing afternoon and evening games in the sweltering Florida summer heat [...] served as a natural deterrent for many casual fans [...]

The baseline knows about the move but gives no attendance figures, and states causes (heat, ticket prices) that the data does not test.

## When the data can't answer

**Vague questions get one clarifying question.** Asked "Who is doing the best?", the agent does no research and replies (`samples/vague.json`, `"clarify": true`, no tool calls):

> Are you asking about MLB attendance or TV viewership, and for which season?

**Questions outside the data get a no-data answer and the closest data.** The A's last season in Oakland was 2024, and national TV viewership covers only 2025 and 2026. Asked "How did the Oakland A's viewership change in their final season before moving?", the agent states the limit and gives the closest data, attendance (`samples/athletics_viewership.json`):

> There is no 2024 viewership in the data[1]. The Oakland A's averaged 11,529 fans a game and drew 922,286 total fans in 2024, filling 21.1%[2][3]. In 2025, their total home attendance fell by 16.7% to 768,464 fans, and their average attendance fell by 17.7% to 9,487 fans a game, though they filled 70.7% in their new home[2][4]. For national TV exposure, the A's had 0 appearances in the data in 2025[5].

Asking about attendance instead, "How did the Oakland A's attendance do in 2024?", gives a full answer (`samples/athletics_attendance.json`):

> The Oakland A's averaged 11,529 fans a game in 2024, which was 6.1% below the 12,272 fans predicted by their prior crowds, win rate, and market size, though still up 12.2% from the previous season[1]. Total home attendance reached 922,286 fans as the team went 69-93 on the year[2][1]. Oakland's home games drew 21.1% of capacity on average[3].

The agent also says so for postseason games, seasons outside 2024 to 2026, local RSN viewership, Peacock-only games, late-September 2025 games, and a game in a multi-game Fox window (it gives the window total and the game's coverage instead).

## The /chat endpoint

`POST /chat` takes `{"message": "...", "session_id": "..."}` (omit `session_id` to start a conversation; pass it back for follow-ups). The response has `response` (the answer in markdown, with footnotes), `session_id`, and `tool_calls`: every call made for that message, by every agent, each with `name`, `args`, `result` and `agent`. Researcher calls are included, labeled by researcher. It also returns `chart`, `team_chart`, `tiles`, `box_score`, `title` and `clarify` for the page. A shortened example, from `samples/sample_4.json` (most tool calls and long text cut, marked `...`):

```json
{
  "response": "The Tampa Bay Rays averaged 9,713 fans a game in 2025, a decrease of 41.2% from the season before, ... [1]. ...\n\n**Sources**\n\n1. team_season_model: `SELECT season, avg_attendance, ... FROM team_season_model WHERE team_id = 139 AND season IN (2024, 2025) ORDER BY season` ...",
  "session_id": "76d5c95c-bb5e-444b-917d-90e276c9d884",
  "tool_calls": [
    {
      "name": "research",
      "args": {"task": "Compare Tampa Bay Rays attendance, expected attendance, and fill rates between the 2024 and 2025 seasons, including the difference.", "role": "attendance"},
      "result": "- In 2024, the Tampa Bay Rays averaged 16,515 fans {Q2.r1.avg_attendance} per home game ... (table: team_season_model). ...",
      "agent": "Lead"
    },
    {
      "name": "run_sql",
      "args": {"query": "SELECT season, avg_attendance, expected_avg_attendance, official_home_total, record, yoy_change_pct, residual_pct, exclusion_reason, market FROM team_season_model WHERE team_id = 139 AND season IN (2024, 2025) ORDER BY season"},
      "result": "{\"query_id\": \"Q2\", \"columns\": [\"season\", \"avg_attendance\", ...], \"row_count\": 2, \"rows\": [[2024, 16515, 18343.18, 1337739, \"80-82\", ...], [2025, 9713, null, 786750, \"77-85\", ...]]}",
      "agent": "Attendance Researcher 1"
    },
    {
      "name": "read_wikipedia",
      "args": {"title": "2025 Tampa Bay Rays season", "offset": 0},
      "result": "{\"title\": \"2025 Tampa Bay Rays season\", \"url\": \"https://en.wikipedia.org/wiki/2025_Tampa_Bay_Rays_season\", ...}",
      "agent": "Context Researcher 2"
    },
    {
      "name": "compute_fact",
      "args": {"op": "change", "facts": ["Q2.r1.avg_attendance", "Q2.r2.avg_attendance"]},
      "result": "{\"value\": \"41.2%\"}",
      "agent": "Fact formatter"
    }
  ],
  "title": "...",
  "clarify": false
}
```

## Viewership figures

Viewership figures are publicly reported average audience (P2+), compiled by TV Media Blog from network press releases and trade coverage (Sports Business Journal, Sports Media Watch). They may differ from official Nielsen data. `viewers_total` is the national TV audience, excluding streaming add-ons (Peacock/Adobe, HBO Max): the main network plus TV simulcasts (Deportes, ESPN2, truTV). Netflix games are streaming-only. Nielsen changed its method in September 2025, so 2025 and 2026 figures are not like-for-like.

## Using the deployed app

Answers can take a few minutes. Conversations are kept in memory, so sessions reset when the service restarts. Raw tool calls are not shown in the answer: "Show the agents' work" opens a panel with each agent's prompts, tool calls and findings.

## Market size data

Team market size uses metro population, one figure per team and season:

- US teams: [US Census Bureau Vintage 2025 metro population estimates](https://www2.census.gov/programs-surveys/popest/datasets/2020-2025/metro/totals/cbsa-est2025-alldata.csv).
- Blue Jays: [Statistics Canada table 17-10-0148-01](https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1710014801), because the US Census does not cover Toronto.

Estimates end at 2025, so 2026 uses the 2025 figure. Two-team metros are not split between teams. The Athletics use San Francisco-Oakland for 2024 and Sacramento for 2025 and 2026.

To load it, run `uv run python scripts/pull_market.py`, then `uv run python scripts/build_all.py`.
