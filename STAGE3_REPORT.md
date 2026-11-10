# Stage 3 report: checks instead of repairs

Branch `stage3-checks`, from `main` at 8463327 (stages 1 and 2). Nothing merged or pushed.

## What changed

- **Checks that only detect** (`check_report` in `app.py`). Sentence-level: first person, stated
  causes, repeated numbers, repeated limit points, untraced numbers / unread links / unfilled IDs,
  derived seat totals, fill gaps in percent instead of points, cleaned-game season figures instead
  of official ones. Whole-answer: a named comparison missing one team's figures, length, "in the
  data" twice in one sentence.
- **Fixed words are never flagged**: `{team_crowd_tv}`, every ready-made phrase, the Nielsen clause,
  and any `*_text` column value a tool returned (headline, network, capped sentences).
- **One Writer retry** lists the problems (quoting each failing sentence). It replaces the fix pass,
  the first-sentence rewrite and the shorten pass. It goes through `run.step`, so required facts are
  protected.
- **After the retry**: code repairs run, then the checks again. A sentence still failing is dropped,
  unless it holds the last copy of a required fact (kept and logged). If every sentence fails, the
  retry's text is kept. Everything is logged in `check_failures` (new in eval output).
- **Repairs removed, because a check now covers their case**: `drop_stated_causes`,
  `drop_repeated_numbers`, `drop_repeated_limits`, `drop_derived_seats`, `drop_unsourced` (function
  kept for tests), and the fill-gap part of `fix_units` (the coverage "64%" part stays).
- **Open-issue fixes**: `nielsen_inline` attaches its clause only to a sentence with a viewer number,
  not one that counts appearances. A fill rate returned as a fraction (1.027) now shows as 102.7%,
  not "1.0%" (a stage 2 formatter bug, found in rerun 1 on `new_opener`).
- **Tests**: `tests/test_checks.py` (10 tests) and 2 new formatter asserts in `tests/test_facts.py`.
  All 61 fast tests pass, including every "good text stays unchanged" case.

## Results vs stage 2

Full run (40 questions, PARALLEL = 2) on the final commit, plus rerun 2 for its 3 failures, also on
the final commit. Files: `evals/stage3_full.json`, `stage3_rerun1.json`, `stage3_rerun2.json`,
`stage3_combined.json`.

| | Stage 2 (combined) | Stage 3 (combined) |
|---|---|---|
| Passed | 40 of 40 | 40 of 40 |
| Full run alone | 34 of 40 (5 quota crashes) | 37 of 40 (2 quota crashes, 1 chart choice) |
| Model fix passes | 6 (`writer_fix_pass`) | 9 (`writer_retry`) |
| Code repairs that changed text | 20 | 19 |
| Sentences dropped by a check | n/a | 1 |
| Answers with no step at all | 25 | 21 |
| Required fact lost / empty answer / "driving both" | 0 / 0 / 0 | 0 / 0 / 0 |

- `drop_repeated_numbers`, `lead_with_team_result` and `network_mix_note` fired 0 times (2, 2, 1 in
  stage 2). The retry fixes more in the Writer's own words, so fewer answers need code repairs. But
  more answers get a retry, so fewer have no step at all.
- `chart_al_east` failed once in the full run (bar chart instead of paired) with no repairs: the
  chart comes from whichever rows the researcher queried. It passed on rerun.

## Decisions made alone

- The official-figure check runs only after the repairs, because `official_season_figures` still
  swaps the number. Running it on the draft would trigger retries the repair already handles.
- Length and "in the data" twice are whole-answer problems: retry and log, never a drop.
- "because / due to" followed by data, source, table, figures or Nielsen explains a data limit, not
  a crowd, so it is not a stated cause (2 false positives in the stage 2 answers otherwise).
- No direction check (up vs down against a gap's sign) yet: the subject of the sentence decides the
  direction, and a regex would misfire.
- A fill value of 2.0 or less is read as a fraction, since no park fills to 2% of capacity.

## Open items

- Dropping a whole sentence can drop one team from a list: `chart_al_east` lost its Rays sentence
  (stated cause). `both_sides` only covers teams the question names, not "the AL East".
- The retry fires on more answers (9 of 40) than the old fix pass (6).
- Vertex 429 quota errors still crash questions at PARALLEL = 2.
- Not checked in the browser yet. Restart the local app first.
