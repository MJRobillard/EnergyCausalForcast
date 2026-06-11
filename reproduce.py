"""
Full reproduction pipeline for Ma et al. 2024 (arXiv:2512.11653).

Steps:
  1. Fetch WAUE hourly load from EIA API
  2. Fetch weather data from Open-Meteo (ERA5)
  3. Train SCM and evaluate on train/test/CV splits

Usage:
    pip install -r requirements.txt
    python reproduce.py [--skip-fetch]   # --skip-fetch if data already downloaded
"""

import argparse
import subprocess
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data" / "raw"


def run(cmd: list[str]):
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Skip data download (use existing CSVs)")
    args = parser.parse_args()

    if not args.skip_fetch:
        print("=== Step 1: Fetch WAUE load data from EIA ===")
        run([sys.executable, "data/fetch_load.py"])

        print("\n=== Step 2: Fetch weather data from Open-Meteo ===")
        run([sys.executable, "data/fetch_weather.py"])
    else:
        print("Skipping fetch — using existing data in data/raw/")
        for fname in ["waue_load.csv", "waue_weather.csv"]:
            if not (DATA_DIR / fname).exists():
                print(f"ERROR: {DATA_DIR / fname} not found. Run without --skip-fetch first.")
                sys.exit(1)

    print("\n=== Step 3: Train SCM and evaluate ===")
    run([sys.executable, "model/train.py"])


if __name__ == "__main__":
    main()
