# MLB-Deep-Research
## Sample questions

The expected answer is what the data says, so the agent can be checked against it. The agent answer is its real output from a saved run on November 4, 2026, unedited. Wording varies between runs, but the facts should hold. The no-tools baseline is the same model with no tools and no system prompt.

### 1. Did games with smaller crowds coincide with smaller national TV audiences in 2025 and 2026?

**Expected answer:** Yes, but the link is weak. Comparing games on the same network, those with smaller crowds tended to draw smaller TV audiences, with a correlation of 0.34 in 2025 and 0.47 in 2026 (0 means no link, 1 means a perfect match). After adjusting for what each park normally draws, the correlation falls to 0.16 in 2025 (n = 82, range -0.05 to 0.37) and 0.14 in 2026 (n = 83, range -0.08 to 0.34). Both ranges include zero, so a crowd below a park's usual level did not clearly mean a smaller TV audience.

**Source:** `crowd_tv_residual`, built from `game_viewership_clean`.

**Agent answer:**

> Yes, games with smaller crowds coincided with smaller national TV audiences in both 2025 and 2026 (table: crowd_tv_residual). In 2025, the correlation showing how raw crowds and TV audiences move together within a network was 0.34, and after adjusting for what each park normally draws, the correlation was 0.16 (n = 82, range -0.05 to 0.37), which includes zero (table: crowd_tv_residual). In 2026, the correlation within a network was 0.47, and after adjusting for what each park normally draws, the correlation was 0.14 (n = 83, range -0.08 to 0.34), which also includes zero (table: crowd_tv_residual). These viewership figures are publicly reported and may differ from official Nielsen data, and they represent the national TV audience, excluding streaming add-ons (table: crowd_tv_residual). Crowds and TV audiences are correlated, not cause and effect (table: crowd_tv_residual).

**No-tools baseline:**

> I do not have access to real-time data or information regarding events after January 2025. Therefore, I cannot provide data or analysis on crowd sizes and national TV audiences for games in 2025 and 2026.

### 2. Why did the Mets over-perform on attendance in 2025?

**Expected answer:** The Mets averaged 39,776 a game in 2025, 28.7% more than the 30,896 the model expected. Part of that is a rebound, because they were 15.1% below expected in 2024. The jump coincided with signing Juan Soto, though the data cannot show the signing caused it.

**Source:** `team_season_model`, and the Wikipedia article [2025 New York Mets season](https://en.wikipedia.org/wiki/2025_New_York_Mets_season).

**Agent answer:**

> The New York Mets' attendance over-performed in 2025, coinciding with the signing of outfielder Juan Soto to a 15-year, $765 million contract ([2025 New York Mets season](https://en.wikipedia.org/wiki/2025_New_York_Mets_season)) and a league-best 45–24 start through June 12 ([2025 New York Mets season](https://en.wikipedia.org/wiki/2025_New_York_Mets_season)). The Mets drew an average actual attendance of 39,776 per game across 81 home openings, totaling 3,184,570 fans (table: team_season_model, team_season_attendance, games_clean), which ranked as the 5th-highest in Major League Baseball ([2025 New York Mets season](https://en.wikipedia.org/wiki/2025_New_York_Mets_season)). This actual average was 28.7% above the expected average attendance of 30,896 (table: team_season_model). For 50 out of 80 modeled games, the crowd was at or near the park's season high, and demand may have been higher (table: games_clean, game_model). The Mets' over-performance is relative to the prior year, and part of it is a rebound from the prior year's under-performance of -15.1% (table: team_season_model). Crowds and the model's effects are correlated, not cause and effect, and promotions are not in the data.

**No-tools baseline:**

> I do not have access to real-time information or events after January 2025. Therefore, I cannot provide details on the Mets' 2025 attendance or the reasons for their performance this season.

### 3. How many people watched the Yankees vs Red Sox game on Fox on June 7, 2025?

**Expected answer:** The data cannot give a number for that game alone. It aired in a regional Fox window alongside Padres/Brewers, and the whole window drew 3,094,000 viewers. Yankees/Red Sox reached 96% of the country.

**Source:** `fox_windows` and `fox_window_games`.

**Agent answer:**

> We cannot determine the exact number of people who watched the New York Yankees vs Boston Red Sox game on Fox on June 7, 2025, because the data cannot split the regional Fox window viewership between its games (tables: fox_windows, fox_window_games). This game was part of a regional Fox window that reached 96.0% of the country, while the San Diego Padres vs Milwaukee Brewers game reached 4.0% (table: fox_window_games). The national TV audience for the entire Fox window was 3,094,000 viewers, consisting of 3,045,000 on the main network and 49,000 on Fox Deportes (table: fox_windows). These viewership figures are publicly reported and may differ from official Nielsen data, and they are the national TV audience, excluding streaming add-ons (table: fox_windows).

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
