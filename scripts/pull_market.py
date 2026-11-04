"""Build data/raw/market/team_markets.csv: metro population per team and season.

Sources:
  US teams    US Census Bureau, Vintage 2025 metro population estimates (cbsa-est2025-alldata.csv)
  Blue Jays   Statistics Canada table 17-10-0148-01 (the US Census does not cover Toronto)

Estimates end at 2025, so 2026 uses the 2025 figure. Two-team metros are not split between
teams: teams_in_market records the sharing.

Usage: uv run python scripts/pull_market.py
"""

import csv
import io
import zipfile
from collections import Counter
from pathlib import Path

import httpx

OUT = Path(__file__).resolve().parent.parent / "data" / "raw" / "market" / "team_markets.csv"
CENSUS_URL = ("https://www2.census.gov/programs-surveys/popest/datasets/2020-2025/metro/totals/"
              "cbsa-est2025-alldata.csv")
STATCAN_URL = "https://www150.statcan.gc.ca/n1/tbl/csv/17100148-eng.zip"
STATCAN_GEO = "Toronto (CMA), Ontario"
SEASONS = [2024, 2025, 2026]
LATEST_ESTIMATE = 2025
BLUE_JAYS = 141
ATHLETICS = 133

# team_id -> Census metro name. The Athletics change market by season, below.
TEAM_METROS = {
    147: "New York-Newark-Jersey City, NY-NJ",
    121: "New York-Newark-Jersey City, NY-NJ",
    119: "Los Angeles-Long Beach-Anaheim, CA",
    108: "Los Angeles-Long Beach-Anaheim, CA",
    112: "Chicago-Naperville-Elgin, IL-IN",
    145: "Chicago-Naperville-Elgin, IL-IN",
    143: "Philadelphia-Camden-Wilmington, PA-NJ-DE-MD",
    111: "Boston-Cambridge-Newton, MA-NH",
    137: "San Francisco-Oakland-Fremont, CA",
    136: "Seattle-Tacoma-Bellevue, WA",
    135: "San Diego-Chula Vista-Carlsbad, CA",
    109: "Phoenix-Mesa-Chandler, AZ",
    115: "Denver-Aurora-Centennial, CO",
    140: "Dallas-Fort Worth-Arlington, TX",
    117: "Houston-Pasadena-The Woodlands, TX",
    118: "Kansas City, MO-KS",
    138: "St. Louis, MO-IL",
    142: "Minneapolis-St. Paul-Bloomington, MN-WI",
    158: "Milwaukee-Waukesha, WI",
    116: "Detroit-Warren-Dearborn, MI",
    114: "Cleveland, OH",
    113: "Cincinnati, OH-KY-IN",
    134: "Pittsburgh, PA",
    110: "Baltimore-Columbia-Towson, MD",
    120: "Washington-Arlington-Alexandria, DC-VA-MD-WV",
    144: "Atlanta-Sandy Springs-Roswell, GA",
    146: "Miami-Fort Lauderdale-West Palm Beach, FL",
    139: "Tampa-St. Petersburg-Clearwater, FL",
}
ATHLETICS_METRO = {2024: "San Francisco-Oakland-Fremont, CA",
                   2025: "Sacramento-Roseville-Folsom, CA",
                   2026: "Sacramento-Roseville-Folsom, CA"}


def census_population(client: httpx.Client) -> dict[tuple[str, int], int]:
    text = client.get(CENSUS_URL).raise_for_status().content.decode("latin-1")
    pops = {}
    for r in csv.DictReader(io.StringIO(text)):
        if r["LSAD"] == "Metropolitan Statistical Area":
            for year in (2024, 2025):
                pops[(r["NAME"], year)] = int(r[f"POPESTIMATE{year}"])
    return pops


def toronto_population(client: httpx.Client) -> dict[int, int]:
    data = client.get(STATCAN_URL).raise_for_status().content
    with zipfile.ZipFile(io.BytesIO(data)) as z, z.open("17100148.csv") as f:
        rows = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8-sig"))
        return {int(r["REF_DATE"]): int(float(r["VALUE"])) for r in rows
                if r["GEO"] == STATCAN_GEO and r["Gender"] == "Total - gender"
                and r["Age group"] == "All ages"}


def main() -> None:
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        census = census_population(client)
        toronto = toronto_population(client)

    rows = []
    for season in SEASONS:
        year = min(season, LATEST_ESTIMATE)
        metros = {**TEAM_METROS, ATHLETICS: ATHLETICS_METRO[season]}
        shared = Counter(metros.values())
        for team_id, metro in metros.items():
            rows.append({"team_id": team_id, "season": season, "market": metro,
                         "population": census[(metro, year)], "population_year": year,
                         "source": "US Census Vintage 2025", "teams_in_market": shared[metro]})
        rows.append({"team_id": BLUE_JAYS, "season": season, "market": "Toronto CMA, Ontario",
                     "population": toronto[year], "population_year": year,
                     "source": "Statistics Canada 17-10-0148-01", "teams_in_market": 1})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
