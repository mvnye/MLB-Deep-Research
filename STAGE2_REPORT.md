# Stage 2 report: facts with IDs

Branch `stage2-facts`, created from `compute-write-check` at 27705b7 (stage 1). Nothing merged or
pushed. `main` and `compute-write-check` are untouched.

## 1. Stage 1 baseline

Full agent run on 27705b7, run in a separate checkout of that commit. Results:
`evals/stage1_baseline.json`, log `evals/stage1_baseline_log.txt`.

- **37 of 40 passed**, no researcher failures, no quota errors.
- Failures:
  - `postseason`: used 1 researcher (the case allows 0). The answer itself was right.
  - `held_tigers`: 21.9% and 25.1% "do not appear in any data". The numbers came from the
    "Start with: ..." fix message, which reads `team_season_model` without logging a query.
  - `uncovered_red_sox_day_night`: **the answer was empty**. The fix pass and `drop_unsourced`
    removed every sentence (the Writer's gap of 2.3 points and a raw 869.37 failed the number check).
- Repairs that changed text: 32 in all. `writer_fix_pass` 9, `fix_citation_format` 3,
  `add_thousands` 3, `nielsen_inline` 3, `drop_repeated_numbers` 2, `drop_unasked` 2,
  `gaps_from_shown` 2, `capacity_wording` 1, others 1 each. 22 of 40 answers had no repair.
- The two crowds vs TV answers said "driving both".

## 2. What changed in stage 2

New file `facts.py`, small changes in `tools.py`, `app.py`, `citations.py`, eval script.

- **Query IDs.** Every successful researcher `run_sql` result gets `query_id` (Q1, Q2, ...),
  added in the event loop so parallel researchers never share an ID. A value is cited as
  `{Q3.r1.avg_attendance}` (query, row from 1, column).
- **Code formats values.** Thousands separators, rounding, units (`39,316 fans`,
  `3,094,000 viewers`), fractions to percents (`21.4%`), fill as `89.9% of capacity` (or
  `of listed capacity` over 100), coverage as a whole percent, correlations to two decimals,
  win_pct as the record (`94-68`), dates as `May 1, 2025`, NULL as `unknown`. A renamed column
  (`t.avg_attendance AS avg_24`) is formatted from its source column.
- **Differences in code.** `{diff:A,B}` and `{change:A,B}` (also `{A - B}`) are worked out from
  the values as shown: two fill rates give percentage points, counts keep their unit. Each one
  is logged as a `compute_fact` tool call, so later checks find its number.
- **Researchers** are asked to put the ID after each value. Code also tags any number they leave
  untagged when exactly one column of that researcher's own results matches it, and removes made-up
  IDs (`{attendance.r1.x}`, `{Q6.row_count}`) before tagging again. READY sentences are not tagged.
- **Writer** writes IDs instead of numbers; code fills them (`fill_report`) after every Writer
  call (draft, fix pass, first-sentence rewrite, shorten). A number the Writer types next to its
  ID, or a unit it adds ("fans", "more fans", "of the listed ... capacity"), is not doubled. An ID
  that does not resolve goes to the fix pass with the reason; one still left is dropped with its
  sentence.
- **Exact citations.** Filling adds `query: Q3` to the sentence's citation, and footnotes use
  those queries instead of guessing from the numbers.
- **Ready-made phrases** for absence and limit statements: `{no_tv:2025}`,
  `{no_viewership:2024}`, `{no_season:2019}`, `{postseason}`, `{rsn}`, `{peacock}`,
  `{late_2025}`, `{window}`, `{unknown_audience}`.
- **Team-level crowds vs TV sentence** is a fixed phrase, `{team_crowd_tv}`, in your wording,
  with numbers filled by code: "Across team-seasons, teams with higher average home attendance
  also tended to have larger average national TV audiences (correlation 0.51, n = 56, range 0.29
  to 0.68). This is consistent with team-level factors such as popularity and market size, but the
  data does not show that one causes the other." The code path that appends it after the headline
  uses the same text.
- **Fix found on the way:** when the fix message gives a team opening's numbers, the
  team-season row is now logged as a query, so those numbers have a source (stage 1 `held_tigers`).
- **Tests:** `tests/test_facts.py` (20 tests). The three existing test files pass unchanged,
  including all "good text stays unchanged" cases.
- **Eval output** now has a `facts` entry per question: values filled, gaps derived, phrases,
  raw numbers the Writer typed, and unresolved IDs.

## 3. Stage 2 results

Runs, all on `stage2-facts`:

| Run | Questions | Passed | Notes |
|---|---|---|---|
| Rerun 1 (after first commit) | 9 | 8 | `held_tigers` failed; led to the logged-row fix |
| Full run | 40 | 34 | 5 crashes were Vertex 429 quota errors; 2 real failures (`chart_al_east`, `uncovered_espn_sundays`: made-up IDs) |
| Rerun 2 (after fixes) | 10 | 9 | the 1 failure was a 429 quota crash |
| Rerun 3 | 3 | 3 | `plain_lookup`, `cubs_white_sox`, `held_tigers` |

Files: `evals/stage2_first_full.json` (+ `_log.txt`), `evals/stage2_rerun2.json`,
`evals/stage2_rerun3.json`, and `evals/stage2_combined.json` (the full run with each question's
latest rerun swapped in). `evals/latest_results.json` holds only rerun 3.

Combined, compared with stage 1:

| | Stage 1 | Stage 2 (combined) |
|---|---|---|
| Passed | 37 of 40 | 40 of 40 |
| Repairs that changed text | 32 | 26 |
| Answers with no repair | 22 | 25 |
| `writer_fix_pass` | 9 | 6 |
| `add_thousands`, `capacity_wording` | 3, 1 | 0, 0 |
| `nielsen_inline` | 3 | 1 |
| `gaps_from_shown` | 2 | 2 (Writer typed a gap instead of `{diff}`) |
| Required fact lost (final check) | 0 | 0 |
| "driving both" in an answer | 2 | 0 |

- The 3 uncovered questions pass: `uncovered_40k_games`, `uncovered_red_sox_day_night` (was
  empty in stage 1), `uncovered_espn_sundays` (1,786,136 viewers across 22 games).
- Fact use: 286 values filled from IDs, 5 gaps derived, 8 phrases, 0 unresolved, and 57 numbers
  the Writer still typed itself (about 17%; many are the fixed crowds vs TV headline numbers,
  which come from a READY sentence).
- `held_network` (open issue): the answer now names every network with its average, not only Fox
  (but see the garbled-text note below).

Caveat: the 40 of 40 is a combined result. 11 questions come from reruns after the full run; the
other 29 come from the full run, before the last fix commits. Those fixes are small and covered by
fast tests, but no single run has all 40 on the final commit.

Found after the runs: the full-run `held_network` answer passed its checks but reads garbled
("...in 2026[1]. led all national networks..."). The Writer used an ID for `network_text`, a
column holding a finished sentence, so a whole sentence was pasted mid-sentence. Fixed in the last
commit: a `_text` column cannot be filled as a value, and the fix pass is told to copy the sentence
word for word. Not rerun with a model (no partial reruns left); covered by a fast test.

## 4. Decisions made on my own

- ID format `{Q3.r1.column}` with rows from 1; gaps as `{diff:A,B}` and `{change:A,B}`.
- Values carry their unit when filled ("fans", "viewers", "% of capacity"); small counts stay
  bare, and the Writer writes the noun.
- Percents and gaps fill as positive numbers; the Writer says the direction in words from the
  sign in the findings. No code check of direction yet (a stage 3 check).
- Writers may still type a number without an ID; the existing number checks still apply to it.
  "Every number from a fact ID" is measured (`raw_numbers`), not enforced.
- Code tags untagged researcher numbers from their own results, because the cheaper researcher
  model often leaves IDs out or invents them.
- Changed the eval checks `sample_crowd_tv` and `crowd_tv_popular_teams` from `prove` to
  `does not show that one causes`, to match your new sentence. `require_no_proof_note` and
  `TEAM_READING` accept the new wording.
- The model-wording check ignores the fixed team-level sentence ("causes the other" is not a
  model claim).
- "There is no 2024 viewership in the data" gets no footnote of its own.
- Saved the baseline log as `.txt` because `*.log` is git-ignored.
- Ran the baseline in a temporary checkout of 27705b7 so stage 2 work could go on at the same time
  (removed afterwards).

## 5. Reverted or unfinished

- Nothing reverted.
- Not a full fresh 40-question run on the final commit (see caveat above). The 429 quota errors
  hit 5 questions in the full run; rerunning them used 2 of the 3 partial reruns.
- Repairs not yet made redundant: `gaps_from_shown` still fires when the Writer types a gap.
  `fix_units`, `records_not_rates`, `official_season_figures`, `drop_derived_seats`,
  `absent_in_data` fired 0 times in both runs, so the runs cannot show a change for them.
- Writer prose issues stage 2 does not address: repeated points ("all 81 games ... every single
  one of their 81 home games"), a missing comma, "44.9 percentage points gap", and
  `cubs_white_sox` once more leaving out the White Sox average (18,072). These are stage 3 checks.
- The context panel shows researcher findings with their IDs in braces. The answer itself never
  shows them.

## 6. Ready for a browser check?

- **Stage 1** (`compute-write-check`, 27705b7): yes. Baseline 37 of 40; the empty Red Sox answer
  and the Tigers sourcing failure are real stage 1 bugs worth knowing about before merging it alone.
- **Stage 2** (`stage2-facts`, latest commit): yes. Look at `held_network` ("Which network had the
  biggest national TV audiences in 2026?") first, since its fix was not rerun with a model. Restart the local app first (it does not reload).
  Worth checking in the browser: the crowds vs TV chip (new team-level wording), Cubs vs White
  Sox, the Rays drop chip, and one "how many people watched" game question, looking at footnotes
  and the "Show the agents' work" panel.
