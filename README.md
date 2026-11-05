# MLB-Deep-Research
## Sample questions

The expected answer is what the data says, so the agent can be checked against it. The agent answer is its real output from a saved run on November 5, 2026, unedited. Wording varies between runs, but the facts should hold. The no-tools baseline is the same model with no tools and no system prompt.

### 1. Did games with smaller crowds coincide with smaller national TV audiences in 2025 and 2026?

**Expected answer:** Yes on the raw numbers, but there is no clear link after adjusting. Comparing games on the same network, smaller crowds tended to come with smaller TV audiences, a moderate correlation of 0.34 in 2025 (n = 82, range 0.14 to 0.52) and 0.47 in 2026 (n = 83, range 0.29 to 0.62), where 0 means no link and 1 means a perfect match. Adjusting compares each game with its own park's usual crowd, so popular teams do not drive the result. After that, the correlation falls to 0.16 in 2025 (range -0.05 to 0.37) and 0.14 in 2026 (range -0.08 to 0.34). Both ranges include zero, so a crowd below a park's usual level did not clearly mean a smaller TV audience.

**Source:** `crowd_tv_residual`, built from `game_viewership_clean`.

**Agent answer:**

> Games with smaller crowds coincided with smaller national TV audiences in 2025 and 2026 when looking at raw numbers, but there is no clear link after adjusting for what each park normally draws. In 2025, raw crowd sizes and national TV audiences moved together within a network with a correlation of 0.34 (n = 82, range 0.14 to 0.52) (table: crowd_tv_residual). After adjusting for what each park normally draws, which compares each game with its own park's usual crowd, so popular teams do not drive the result, the correlation was 0.16 (n = 82, range -0.05 to 0.37), meaning there is no clear link after adjusting (table: crowd_tv_residual). In 2026, raw crowds and national TV audiences moved together within a network with a correlation of 0.47 (n = 83, range 0.29 to 0.62) (table: crowd_tv_residual). After adjusting for what each park normally draws, the correlation was 0.14 (n = 83, range -0.08 to 0.34), which also shows no clear link (table: crowd_tv_residual). These numbers show associations and do not show cause and effect. These viewership figures are publicly reported and may differ from official Nielsen data, and they are the national TV audience, excluding streaming add-ons.

**No-tools baseline:**

> I do not have access to real-time data or information regarding events after January 2025. Therefore, I cannot provide data or analysis on crowd sizes and national TV audiences for games in 2025 and 2026.

### 2. Why did the Mets over-perform on attendance in 2025?

**Expected answer:** The Mets averaged 39,776 a game in 2025, up 35.9% from 29,268 in 2024. That is 23.6% above the 32,182 that their prior two years of crowds, win rate and market would predict. They were 18.8% below that prediction in 2024, so part of this may be a rebound. The rise coincided with signing Juan Soto, though the data cannot show the signing caused it.

**Source:** `team_season_model`, and the Wikipedia article [2025 New York Mets season](https://en.wikipedia.org/wiki/2025_New_York_Mets_season).

**Agent answer:**

> The New York Mets' attendance surge in 2025 coincided with signing outfielder Juan Soto to a 15-year, $765 million contract on December 11, 2024, and opening the season with a league-best 45–24 record through June 12 ([2025 New York Mets season](https://en.wikipedia.org/wiki/2025_New_York_Mets_season)). During the 2025 regular season, the team averaged 39,776 fans per home game, compared to an expected average of 32,182 fans (table: team_season_model). This was a 35.9% increase from the season before (table: team_season_model). The crowd was 23.6% above what the prior two years of crowds, win rate and market would predict (table: team_season_model). Since the team was below that prediction the year before, part of this may be a rebound (table: team_season_model). Promotions and giveaways are not in the data to help explain these crowds (table: games_clean, team_season_model).

**No-tools baseline:**

> I do not have access to real-time information or events after January 2025. Therefore, I cannot provide details on the Mets' 2025 attendance or the reasons for their performance this season.

### 3. How many people watched the Yankees vs Red Sox game on Fox on June 7, 2025?

**Expected answer:** The data cannot give a number for that game alone. It aired in a regional Fox window alongside Padres/Brewers, and the whole window drew 3,094,000 viewers. Yankees/Red Sox reached 96% of the country.

**Source:** `fox_windows` and `fox_window_games`.

**Agent answer:**

> We cannot determine the exact number of people who watched the Yankees vs Red Sox game on June 7, 2025, because the data cannot split the total audience between games in a regional Fox window (table: fox_windows, fox_window_games). The regional Fox window had a total audience of 3,094,000 viewers, which included 3,045,000 on the main Fox network and 49,000 on Fox Deportes (table: fox_windows). This window also featured a Padres vs. Brewers game that reached 4% of the country (table: fox_windows, fox_window_games). There is no specific national television viewership data recorded for the Yankees vs. Red Sox game itself (table: game_viewership_clean, viewership_games).
>
> These viewership figures are publicly reported and may differ from official Nielsen data, and they represent the national TV audience, excluding streaming add-ons.

**No-tools baseline:**

> I cannot provide the viewership numbers for the Yankees vs. Red Sox game on June 7, 2025, as this date is after my knowledge cutoff of January 2025, and I do not have access to real-time or future data.

## Viewership figures

Viewership figures are publicly reported average audience (P2+), compiled by TV Media Blog from network press releases and trade coverage (Sports Business Journal, Sports Media Watch). They may differ from official Nielsen data. `viewers_total` is the national TV audience, excluding streaming add-ons (Peacock/Adobe, HBO Max): the main network plus TV simulcasts (Deportes, ESPN2, truTV). Netflix games are streaming-only.

## Market size data

Team market size uses metro population, one figure per team and season:

- US teams: [US Census Bureau Vintage 2025 metro population estimates](https://www2.census.gov/programs-surveys/popest/datasets/2020-2025/metro/totals/cbsa-est2025-alldata.csv).
- Blue Jays: [Statistics Canada table 17-10-0148-01](https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1710014801), because the US Census does not cover Toronto.

Estimates end at 2025, so 2026 uses the 2025 figure. Two-team metros are not split between teams. The Athletics use San Francisco-Oakland for 2024 and Sacramento for 2025 and 2026.

To load it, run `uv run python scripts/pull_market.py`, then `uv run python scripts/build_all.py`.
