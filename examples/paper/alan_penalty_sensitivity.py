from __future__ import annotations

import csv
from pathlib import Path

from rpqubo.examples import reproduce_alan_penalty_sensitivity


def main() -> None:
    rows = reproduce_alan_penalty_sensitivity()
    out = Path("outputs/paper/alan_penalty_sensitivity.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(out)


if __name__ == "__main__":
    main()
