"""Rebuild data/mlb.duckdb from the raw files: load the data, then fit the models.

Usage: uv run python scripts/build_all.py
"""

import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent

for script in ("load_duckdb.py", "build_models.py"):
    print(f"== {script}", flush=True)
    subprocess.run([sys.executable, str(SCRIPTS / script)], check=True)
