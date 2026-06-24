from __future__ import annotations

import csv
from pathlib import Path

from rpqubo.encodings import nonlinear_error_table


def test_paper_nonlinear_error_values_with_tolerance() -> None:
    rows = nonlinear_error_table([0.6, 1.5], [1, 2, 3])
    lookup = {(row["a"], row["J"]): row for row in rows}
    assert abs(lookup[(0.6, 2.0)]["max_abs_error"] - 0.180553375376858) <= 1e-12
    assert abs(lookup[(1.5, 3.0)]["rmse"] - 0.04677738492225125) <= 1e-12


def test_alan_sensitivity_csv_schema_is_stable() -> None:
    path = Path(__file__).resolve().parents[1] / "data" / "alan_penalty_sensitivity.csv"
    with path.open() as f:
        reader = csv.DictReader(f)
        assert {
            "lambda_input",
            "obj",
            "feas",
            "e1_abs",
            "e2_abs",
            "qubo_max_abs_unscaled",
            "qubo_dyn_range_unscaled",
        } <= set(reader.fieldnames or [])
