from __future__ import annotations

import csv
from pathlib import Path

from rpqubo.examples import reproduce_example2_zoom


def main() -> None:
    rows = reproduce_example2_zoom()
    out = Path("outputs/paper/ex2_miqp_zoom.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(out)


if __name__ == "__main__":
    main()
