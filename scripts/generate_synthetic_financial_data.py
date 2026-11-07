"""Write fictional Stage 9 CSV fixtures; never connect to any database."""

from __future__ import annotations

import argparse
import csv
from datetime import date
from decimal import Decimal
import json
import os
from pathlib import Path
import tempfile
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.synthetic_financial_data import SCENARIO, build_dataset

OUTPUT_DIR = ROOT / "tests" / "fixtures"
TRANSACTIONS_PATH = OUTPUT_DIR / "synthetic_financial_transactions.csv"
CONTRIBUTIONS_PATH = OUTPUT_DIR / "synthetic_savings_contributions.csv"
SCENARIO_PATH = OUTPUT_DIR / "synthetic_financial_scenario.json"


def _json_value(value):
    if isinstance(value, (date, Decimal)):
        return value.isoformat() if isinstance(value, date) else str(value)
    raise TypeError(f"Unsupported scenario value: {type(value).__name__}")


def _atomic_csv(path: Path, rows: tuple[dict, ...], columns: tuple[str, ...]) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent,
                                     prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: row[key].isoformat() if isinstance(row[key], date)
                else format(row[key], "f") if isinstance(row[key], Decimal)
                else row[key]
                for key in columns
            })
    os.replace(temporary, path)


def _atomic_json(path: Path, value: dict) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, default=_json_value)
        stream.write("\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", required=True, choices=("test", "development"),
                        help="Explicitly confirm this is a non-production fixture generation run.")
    parser.add_argument("--replace-existing", action="store_true",
                        help="Replace only the three known generated fixture files.")
    args = parser.parse_args()
    if os.getenv("FLASK_CONFIG", "").lower() == "production":
        parser.error("Synthetic fixture generation is disabled when FLASK_CONFIG=production.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Refuse symlinked fixture targets so replacement cannot escape the fixture directory.
    paths = (TRANSACTIONS_PATH, CONTRIBUTIONS_PATH, SCENARIO_PATH)
    if any(path.is_symlink() for path in paths):
        parser.error("A generated fixture path is a symlink; refusing to write.")
    if any(path.exists() for path in paths) and not args.replace_existing:
        parser.error("Generated fixture files already exist. Pass --replace-existing to replace these synthetic fixtures.")
    if OUTPUT_DIR.resolve() != (ROOT / "tests" / "fixtures").resolve():
        parser.error("Fixture output directory did not resolve to tests/fixtures.")

    dataset = build_dataset()
    _atomic_csv(
        TRANSACTIONS_PATH, dataset.transactions,
        ("id", "user_key", "transaction_date", "transaction_type", "category",
         "category_type", "amount", "description"),
    )
    _atomic_csv(
        CONTRIBUTIONS_PATH, dataset.contributions,
        ("user_key", "goal_name", "target_amount", "amount", "contributed_on", "note"),
    )
    _atomic_json(SCENARIO_PATH, dict(SCENARIO))
    print(f"Wrote {len(dataset.transactions)} synthetic transactions for {len(dataset.users)} users.")
    print(f"Wrote {len(dataset.contributions)} savings contributions (seed {dataset.seed}).")
    print(f"Fixture-only output directory: {OUTPUT_DIR}")
    print("No database connection was opened.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
