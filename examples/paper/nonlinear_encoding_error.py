from __future__ import annotations

import csv
from pathlib import Path

from rpqubo.encodings import nonlinear_error_table


def main() -> None:
    rows = nonlinear_error_table([0.6, 1.5], [1, 2, 3])
    out = Path("outputs/paper/nonlinear_encoding_error.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(out)


if __name__ == "__main__":
    main()
