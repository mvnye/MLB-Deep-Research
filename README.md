# MLB-Deep-Research
## Sample questions

The expected answer is what the data says, so the agent can be checked against it. The agent answer is its real output from a saved run on November 4, 2026, unedited. Wording varies between runs, but the facts should hold. The no-tools baseline is the same model with no tools and no system prompt.

### 1. Did smaller crowds go with smaller national TV audiences in 2025 and 2026?

**Expected answer:** Yes, but most of the link disappears once each park's normal crowd and game factors are removed. Within networks, the correlation of crowds and TV audiences is 0.34 in 2025 and 0.47 in 2026. After removing those factors it falls to 0.16 (n = 82, range -0.05 to 0.37) and 0.14 (n = 83, range -0.08 to 0.34), so both ranges include zero. This is correlation, not cause and effect.

**Source:** `crowd_tv_residual`, built from `game_viewership_clean`.

**Agent answer:**

> Smaller raw crowds coincided with smaller national TV audiences, excluding streaming add-ons, in both 2025 and 2026, though this relationship was weaker when looking at how much of that was left after removing each park's normal crowd (table: crowd_tv_residual). In 2025, the raw crowds and TV audiences moved together with a correlation of 0.34, but the correlation was 0.16 after removing each park's normal crowd (n = 82, range -0.05 to 0.37) (table: crowd_tv_residual). In 2026, the raw correlation was 0.47, which fell to 0.14 after removing each park's normal crowd (n = 83, range -0.08 to 0.34) (table: crowd_tv_residual). These publicly reported figures may differ from official Nielsen data, and the crowds and TV audiences are correlated rather than cause and effect (table: crowd_tv_residual).
>
> ### 2025 Season Viewership
>
> For ESPN, the raw correlation was 0.49, and the correlation after removing each park's normal crowd was 0.27; the sample is small (n = 26, range -0.14 to 0.59) (table: crowd_tv_residual). For FS1, the raw correlation was 0.29, and the correlation after removing each park's normal crowd was 0.17 (n = 34, range -0.18 to 0.48) (table: crowd_tv_residual).
>
> ### 2026 Season Viewership
>
> For ESPN, the raw correlation was 0.48, and the correlation after removing each park's normal crowd was 0.30; the sample is small (n = 23, range -0.12 to 0.64) (table: crowd_tv_residual). For FS1, the raw correlation was 0.52, and the correlation after removing each park's normal crowd was 0.43; the sample is small (n = 21, range -0.01 to 0.72) (table: crowd_tv_residual). Additionally, Nielsen changed its measurement panel method on September 1, 2025, meaning the 2025 and 2026 figures are not like-for-like (table: crowd_tv_residual).

**No-tools baseline:** To be added.

### 2. Why did the Mets over-perform on attendance in 2025?

**Expected answer:** The Mets averaged 39,776 against an expected 30,896, which is 28.7% above expected. That is relative to the prior year and follows 15.1% below expected in 2024, so part of it is a rebound. It coincided with signing Juan Soto. The data cannot show that the signing caused it, and promotions are not in the data.

**Source:** `team_season_model`, and the Wikipedia article [2025 New York Mets season](https://en.wikipedia.org/wiki/2025_New_York_Mets_season).

**Agent answer:**

> The New York Mets over-performed on attendance during the 2025 regular season, drawing an average of 39,775 fans per home game, which ranked fifth-highest in Major League Baseball ([2025 New York Mets season](https://en.wikipedia.org/wiki/2025_New_York_Mets_season)). This attendance coincided with major offseason roster additions, including signing outfielder Juan Soto to a historic contract, and a strong 45–24 start to the season ([2025 New York Mets season](https://en.wikipedia.org/wiki/2025_New_York_Mets_season)). However, the team's over-performance is relative to the prior year, meaning the high attendance partly reflects a rebound from a weak 2024 season (table: team_season_model).
>
> ### Attendance Performance
> During the 2025 regular season, the Mets averaged an actual attendance of 39,775 across 80 home games, totaling 3,182,052 attendees (table: games_clean, team_season_attendance). The team-season model expected an average attendance of 30,896, meaning the Mets performed 28.7% above expected (table: team_season_model). Promotions and giveaways are not in the data, so they cannot be used to explain game-level differences (table: games_clean).
>
> ### Roster Changes and Team Performance
> In the offseason, the Mets signed outfielder Juan Soto to a 15-year, $765 million contract, the largest in professional sports history ([2025 New York Mets season](https://en.wikipedia.org/wiki/2025_New_York_Mets_season)). The team also re-signed pitcher Sean Manaea to a three-year, $75 million contract, re-signed first baseman Pete Alonso to a two-year, $54 million contract, signed pitchers Frankie Montas, Clay Holmes, and Griffin Canning, and traded for outfielder Jose Siri ([2025 New York Mets season](https://en.wikipedia.org/wiki/2025_New_York_Mets_season)). The team ultimately went 38–55 in the second half of the season, missing the postseason with an 83–79 record ([2025 New York Mets season](https://en.wikipedia.org/wiki/2025_New_York_Mets_season)).

**No-tools baseline:** To be added.

### 3. How many people watched the Yankees vs Red Sox game on Fox on June 7, 2025?

**Expected answer:** The data cannot say for that game alone. It aired in a Fox window with Padres/Brewers, and the window drew 3,094,000 (3,045,000 on Fox and 49,000 on Fox Deportes). Yankees/Red Sox reached 96% of the country, but that is coverage, not a share of viewers, so the total cannot be split.

**Source:** `fox_windows` and `fox_window_games`.

**Agent answer:**

> On June 7, 2025, the New York Yankees and Boston Red Sox played in a multi-game Fox window that drew a certain national TV audience, excluding streaming add-ons, of 3,094,000 viewers (table: fox_windows). The data cannot split the viewership to show a specific total for the Yankees vs. Red Sox game because it was a multi-game window (table: fox_windows, fox_window_games). Within this broadcast window, the Yankees and Red Sox game had a 96% coverage percentage, while a Padres and Brewers game had a 4% coverage percentage (table: fox_window_games).
>
> ### Broadcast Viewership
>
> The national TV audience, excluding streaming add-ons, for the entire broadcast window was 3,094,000 viewers, and the main network audience was 3,045,000 viewers (table: fox_windows). The total audience within the network was 3,094,000 viewers, and this total is certain (table: fox_windows). These viewership figures are publicly reported and may differ from official Nielsen data (table: fox_windows).
>
> ### Game Coverage
>
> The broadcast window featured two games, but the data cannot split the viewership between them (table: fox_windows, fox_window_games). The game between the Yankees and Red Sox coincided with a 96% coverage percentage in the window (table: fox_window_games). The other game in the window, between the Padres and Brewers, had a coverage percentage of 4% (table: fox_window_games).

**No-tools baseline:** To be added.

## Viewership figures

Viewership figures are publicly reported average audience (P2+), compiled by TV Media Blog from network press releases and trade coverage (Sports Business Journal, Sports Media Watch). They may differ from official Nielsen data. `viewers_total` is the national TV audience, excluding streaming add-ons (Peacock/Adobe, HBO Max): the main network plus TV simulcasts (Deportes, ESPN2, truTV). Netflix games are streaming-only.

## Market size data

Team market size uses metro population, one figure per team and season:

- US teams: [US Census Bureau Vintage 2025 metro population estimates](https://www2.census.gov/programs-surveys/popest/datasets/2020-2025/metro/totals/cbsa-est2025-alldata.csv).
- Blue Jays: [Statistics Canada table 17-10-0148-01](https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1710014801), because the US Census does not cover Toronto.

Estimates end at 2025, so 2026 uses the 2025 figure. Two-team metros are not split between teams. The Athletics use San Francisco-Oakland for 2024 and Sacramento for 2025 and 2026.

To load it, run `uv run python scripts/pull_market.py`, then `uv run python scripts/build_all.py`.
