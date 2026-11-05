# MLB-Deep-Research
## Sample questions

The expected answer is what the data says, so the agent can be checked against it. The agent answer is its real output from a saved run on November 5, 2026, unedited. Wording varies between runs, but the facts should hold. The no-tools baseline is the same model with no tools and no system prompt.

To test the agent against 23 questions with required facts, run `uv run python scripts/eval_agent.py`.

### 1. Did games with smaller crowds coincide with smaller national TV audiences in 2025 and 2026?

**Expected answer:** Yes on the raw numbers, but there is no clear link after adjusting. Comparing games on the same network, smaller crowds tended to come with smaller TV audiences, a moderate correlation of 0.34 in 2025 (n = 82, range 0.14 to 0.52) and 0.47 in 2026 (n = 83, range 0.29 to 0.62), where 0 means no link and 1 means a perfect match. Adjusting compares each game with its own park's usual crowd, so popular teams do not drive the result. After that, the correlation falls to 0.16 in 2025 (range -0.05 to 0.37) and 0.14 in 2026 (range -0.08 to 0.34). Both ranges include zero, so a crowd below a park's usual level did not clearly mean a smaller TV audience.

**Source:** `crowd_tv_residual`, built from `game_viewership_clean`.

**Agent answer:**

> Yes on the raw numbers: comparing games on the same network, smaller crowds went with smaller national TV audiences (correlation 0.34 in 2025, n = 82, range 0.14 to 0.52; 0.47 in 2026, n = 83, range 0.29 to 0.62). Mostly no after comparing each game with its own park's usual crowd, so popular teams do not drive the result: the correlation falls to 0.16 in 2025 (range -0.05 to 0.37) and 0.14 in 2026 (range -0.08 to 0.34), and both ranges include zero (table: crowd_tv_headline).
>
> *Viewership figures are publicly reported and may differ from official Nielsen data. They are the national TV audience, excluding streaming add-ons. Nielsen changed its method on September 1, 2025, so 2025 and 2026 figures are not like-for-like. These are associations, not cause and effect.*

**No-tools baseline:**

> I do not have access to real-time data or statistics for 2025 and 2026. Therefore, I cannot provide information on whether smaller crowd sizes coincided with smaller national TV audiences for games during those years.

### 2. Why did the Mets over-perform on attendance in 2025?

**Expected answer:** The Mets averaged 39,776 a game in 2025, up 35.9% from 29,268 in 2024. That is 23.6% above the 32,182 that their prior two years of crowds, win rate and market would predict. They were 18.8% below that prediction in 2024, so part of this may be a rebound. The rise coincided with signing Juan Soto, though the data cannot show the signing caused it.

**Source:** `team_season_model`, and the Wikipedia article [2025 New York Mets season](https://en.wikipedia.org/wiki/2025_New_York_Mets_season).

**Agent answer:**

> The New York Mets' home attendance surged in 2025, with actual average attendance growing by 35.9% from the season before to reach 39,776 fans per game (table: team_season_model). This turnout was 23.6% higher than what the prior two years of crowds, win rate, and market size would predict (table: team_season_model). Part of this may be a rebound: the team was 18.8% below its prediction the season before (table: team_season_model). In comparison, the 2025 actual average of 39,776 fans per game was higher than the 2024 average of 29,268 fans per game by 10,508 fans per game (table: team_season_model). These strong crowds coincided with the franchise signing outfielder Juan Soto to a 15-year, $765 million contract in December 2024 ([2025 New York Mets season](https://en.wikipedia.org/wiki/2025_New_York_Mets_season)). The team also added starting pitchers Frankie Montas and Griffin Canning during the offseason ([2025 New York Mets season](https://en.wikipedia.org/wiki/2025_New_York_Mets_season)), and opened the season with a stellar 45–24 record through June 12 ([2025 New York Mets season](https://en.wikipedia.org/wiki/2025_New_York_Mets_season)).
>
> *Promotions are not in the data. These are associations, not cause and effect.*

**No-tools baseline:**

> I do not have access to real-time information or events after January 2025, so I cannot provide information on the Mets' 2025 attendance or the reasons for their performance.

### 3. How many people watched the Yankees vs Red Sox game on Fox on June 7, 2025?

**Expected answer:** The data cannot give a number for that game alone. It aired in a regional Fox window alongside Padres/Brewers, and the whole window drew 3,094,000 viewers. Yankees/Red Sox reached 96% of the country.

**Source:** `fox_windows` and `fox_window_games`.

**Agent answer:**

> The data has no separate audience number for the Yankees versus Red Sox game alone because it was broadcast in a regional window that cannot be split (table: game_tv_lookup). The entire regional Fox window drew 3,094,000 viewers (table: fox_windows). This window included the Yankees versus Red Sox game, which reached 96.0% of the country, and the San Diego Padres at Milwaukee Brewers game, which reached 4.0% of the country (table: fox_window_games).
>
> *Viewership figures are publicly reported and may differ from official Nielsen data. They are the national TV audience, excluding streaming add-ons.*

**No-tools baseline:**

> I cannot provide the viewership numbers for the Yankees vs. Red Sox game on June 7, 2025, as this date is in the future and the event has not yet occurred.

## Viewership figures

Viewership figures are publicly reported average audience (P2+), compiled by TV Media Blog from network press releases and trade coverage (Sports Business Journal, Sports Media Watch). They may differ from official Nielsen data. `viewers_total` is the national TV audience, excluding streaming add-ons (Peacock/Adobe, HBO Max): the main network plus TV simulcasts (Deportes, ESPN2, truTV). Netflix games are streaming-only.

## Using the deployed app

Answers take up to about 2 minutes. Conversations are kept in memory, so sessions reset when the service restarts.

## Market size data

Team market size uses metro population, one figure per team and season:

- US teams: [US Census Bureau Vintage 2025 metro population estimates](https://www2.census.gov/programs-surveys/popest/datasets/2020-2025/metro/totals/cbsa-est2025-alldata.csv).
- Blue Jays: [Statistics Canada table 17-10-0148-01](https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1710014801), because the US Census does not cover Toronto.

Estimates end at 2025, so 2026 uses the 2025 figure. Two-team metros are not split between teams. The Athletics use San Francisco-Oakland for 2024 and Sacramento for 2025 and 2026.

To load it, run `uv run python scripts/pull_market.py`, then `uv run python scripts/build_all.py`.
