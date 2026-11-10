# MLB-Deep-Research

A multi-agent deep research app for sports media and sponsorship sellers preparing team pitches. It explores what drives MLB attendance, why specific teams and games over- or under-perform, and whether smaller crowds went with smaller national TV audiences in 2025 and 2026. A Lead clarifies vague questions and sends Attendance, Viewership and Context Researchers in parallel, and a Writer turns their findings into a footnoted answer, with every number checked against its source.

**Live app:** https://mlb-deep-research-git-636637173930.europe-west1.run.app 

## Loading the data and running the app

To run the app on your own computer:

```bash
uv sync
uv run uvicorn app:app --port 8080
```

To rebuild `data/mlb.duckdb` from the raw files already in `data/raw/`, run:

```bash
uv run python scripts/build_all.py
```

To download fresh data first, run these two before it:

```bash
uv run python scripts/pull_statsapi.py
uv run python scripts/pull_market.py
```

## Sample questions

### 1. Why did the Mets over-perform on attendance in 2025?

**Expected answer:** 39,316 fans a game in 2025 (official), up 33.3% from 29,485 in 2024 and 21.4% above the 32,383 the model predicts. They were 18.6% below prediction in 2024, so part may be a rebound. The rise coincided with signing Juan Soto in December 2024.

**Source:** `team_season_model` (`SELECT * FROM team_season_model WHERE team_id = 121 AND season = 2025`), and the Wikipedia article [2025 New York Mets season](https://en.wikipedia.org/wiki/2025_New_York_Mets_season).

**Agent answer:**

> The New York Mets averaged 39,316 fans a game in 2025, 21.4% above the 32,383 fans their prior two years of crowds, win rate, and market would predict, up 33.3% from 29,485 the season before, and part of this may be a rebound from 18.6% below prediction[1]. The Mets drew 94.9% of capacity, with game-by-game crowds averaging 1,872 fans more than expected[2][3]. This surge coincided with the blockbuster signing of outfielder Juan Soto to a record 15-year, $765 million contract in December 2024, and the re-signing of first baseman Pete Alonso in February 2025[4]. On the field, the Mets opened the season with an MLB-best 45–24 record through June 12[4].

**No-tools baseline:**

> I do not have access to real-time information or events after January 2025, so I cannot provide information on the Mets' 2025 attendance or the reasons for their performance.

### 2. Compare the Cubs and White Sox on attendance and national TV audiences in 2025.

**Expected answer:** Cubs 37,259 fans a game (3,017,983 total, 89.9% of capacity); White Sox 18,072 (1,445,738, 45.0%), 19,187 fewer a game. The Cubs had 15 national TV appearances in the 2025 data, and their 9 single games with a known audience averaged 1,126,444 viewers. The White Sox had none in the data, which ends in early September.

**Source:** `team_season_model` and `team_season_attendance` (official averages and totals), `games_clean` (fill), `team_tv_summary` (`SELECT * FROM team_tv_summary WHERE season = 2025 AND team_id IN (112, 145)`).

**Agent answer:**

> The Chicago Cubs dominated the Chicago White Sox in both ticket sales and media presence in 2025, drawing 1,572,245 more fans at home and securing the city's only national television coverage[1]. The Cubs drew 3,017,983 total fans, averaging 37,259 fans per game and drew 89.9% of capacity[2][3]. In contrast, the White Sox drew 1,445,738 total fans, averaging 18,072 fans per game and drew 45.0% of capacity[2][3]. This left the Cubs 19,187 fans per game ahead of their crosstown rivals, with a fill-rate gap of 44.9 percentage points[1]. On television, the Cubs had 15 national TV appearances and 10 single-game broadcasts, averaging 1,126,444 viewers per broadcast, while the White Sox had no national TV broadcasts in the 2025 data[4].
>
> *Viewership figures are publicly reported and may differ from official Nielsen data. They are the national TV audience, excluding streaming add-ons.*

The 1,126,444 average covers the 9 single games with a known audience, not all 10.

**No-tools baseline** (shortened; full text in `samples/baseline_3.md`):

> Because my knowledge cutoff is in January 2025, actual attendance figures and national TV ratings for the 2025 Major League Baseball season are not yet available. [...] In 2024, the Cubs drew **2,907,591 fans** (averaging 35,896 per game) [...] In 2024, they [the White Sox] drew just **1,380,183 fans** (averaging 17,039 per game) [...] the Cubs are expected to vastly outpace the White Sox in both stadium attendance and national television viewership [...]

The baseline gives no 2025 figures, and its 2024 figures are close to but not the official ones (Cubs 2,909,794 and 35,923; White Sox 1,380,733 and 17,932).

### 3. Did games with smaller crowds coincide with smaller national TV audiences in 2025 and 2026?

**Expected answer:** Yes, but mostly because of which teams were playing. Comparing games on the same network, the correlation was 0.34 in 2025 (n = 82) and 0.47 in 2026 (n = 83). After comparing each game with its own park's usual crowd, it falls to 0.16 and 0.14, and both ranges include zero. Across team-seasons the correlation is 0.51 (n = 56), consistent with popularity and market size, but the data does not show cause.

**Source:** `crowd_tv_headline` (built from `crowd_tv_residual`) and `team_crowd_tv`.

**Agent answer:**

> Yes, games with smaller crowds tended to have smaller national TV audiences, but a crowd that was small for its own park did not clearly mean a smaller audience[1]. Comparing games on the same network, the two moved together (correlation 0.34 in 2025, n = 82, range 0.14 to 0.52; 0.47 in 2026, n = 83, range 0.29 to 0.62)[1]. After comparing each game with its own park's usual crowd, the link mostly disappears (0.16 in 2025, range -0.05 to 0.37; 0.14 in 2026, range -0.08 to 0.34; both ranges include zero)[1]. Across team-seasons, teams with higher average home attendance also tended to have larger average national TV audiences (correlation 0.51, n = 56, range 0.29 to 0.68)[2]. This is consistent with team-level factors such as popularity and market size, but the data does not show that one causes the other[2].
>
> *Viewership figures are publicly reported and may differ from official Nielsen data. They are the national TV audience, excluding streaming add-ons. Nielsen changed its method on September 1, 2025, so 2025 and 2026 figures are not like-for-like.*

**No-tools baseline:**

> I do not have access to real-time data or statistics for 2025 and 2026. Therefore, I cannot provide information on whether smaller crowd sizes coincided with smaller national TV audiences for games during those years.

## Questions the agent can't answer

**Vague questions** get one clarifying question. "Who is doing the best?" gets: "Are you asking about MLB attendance or TV viewership, and for which season?" (`samples/chat_responses/vague.json`)

**Questions outside the data** get a no-data answer and the closest data. National TV viewership covers only 2025 and 2026, so "How did the Oakland A's viewership change in their final season before moving?" gets "There is no 2024 viewership in the data," followed by the A's 2024 attendance (`samples/chat_responses/athletics_viewership.json`).

The agent also flags postseason games, seasons outside 2024 to 2026, local RSN viewership, Peacock-only games, late-September 2025 games, and single games inside a multi-game Fox window.

## The /chat endpoint

The full `POST /chat` output for each sample question, with `response`, `session_id` and every tool call (`name`, `args`, `result`, `agent`, including the researchers' calls), is in `samples/chat_responses/`.

## Data sources

- [MLB Stats API](https://statsapi.mlb.com): 2024 to 2026 regular season games (`/api/v1/schedule?sportId=1&gameType=R`), per-game attendance, weather, venue and capacity (`/api/v1.1/game/{gamePk}/feed/live`), and official season totals (`/api/v1/attendance`), back to 2022 for the model's prior years. Postseason is out of scope.
- TV Media Blog viewership archives: [2025](https://tvmediablog.substack.com/p/2025-mlb-season-viewership-database) and [2026](https://tvmediablog.substack.com/p/2026-mlb-viewership-archive), saved as CSVs in `data/raw/`. The 2025 tables end in early September.
- Wikipedia team-season articles, read live through the public MediaWiki API.
- Metro population, described under Market size data.


### Viewership figures

Viewership figures are publicly reported average audience (P2+), compiled by TV Media Blog from network press releases and trade coverage (Sports Business Journal, Sports Media Watch). They may differ from official Nielsen data. `viewers_total` is the national TV audience, excluding streaming add-ons (Peacock/Adobe, HBO Max): the main network plus TV simulcasts (Deportes, ESPN2, truTV). Netflix games are streaming-only. Nielsen changed its method in September 2025, so 2025 and 2026 figures are not like-for-like.

### Market size data

Team market size uses metro population, one figure per team and season:

- US teams: [US Census Bureau Vintage 2025 metro population estimates](https://www2.census.gov/programs-surveys/popest/datasets/2020-2025/metro/totals/cbsa-est2025-alldata.csv).
- Blue Jays: [Statistics Canada table 17-10-0148-01](https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1710014801), because the US Census does not cover Toronto.

Estimates end at 2025, so 2026 uses the 2025 figure. Two-team metros are not split between teams. The Athletics use San Francisco-Oakland for 2024 and Sacramento for 2025 and 2026.

To load it, run `uv run python scripts/pull_market.py`, then `uv run python scripts/build_all.py`.
