from __future__ import annotations

import pytest

from rpqubo.examples import reproduce_alan_penalty_sensitivity
from rpqubo.paper_expectations import TABLE9, TABLE9_BY_LAMBDA, compare_table9_row


@pytest.mark.slow
def test_alan_penalty_sensitivity_regenerates_table9() -> None:
    rows = reproduce_alan_penalty_sensitivity()
    by_lambda = {float(row["lambda_input"]): row for row in rows}
    assert set(by_lambda) == set(TABLE9_BY_LAMBDA)
    for expected in TABLE9:
        lambda_value = expected.lambda_input
        row = by_lambda[lambda_value]
        assert row["n_vars"] == 46
        assert row["n_quad"] == 385
        comparison = compare_table9_row(row)
        assert comparison["passed"], comparison
        assert comparison["status"] in {
            "exact_reference_match",
            "within_documented_stochastic_tolerance",
        }
        if not comparison["exact_reference_match"]:
            assert comparison["within_stochastic_tolerance"]


def test_table9_status_is_honest_for_stochastic_alternative() -> None:
    reference = TABLE9_BY_LAMBDA[1000.0]
    row = {
        "lambda_input": 1000.0,
        "objective": reference.objective + 0.01,
        "feasibility": reference.feasibility,
        "e1_abs": reference.e1_abs,
        "e2_abs": reference.e2_abs,
        "n_vars": 46,
        "n_quad": 385,
        "qubo_max_abs_unscaled": reference.qubo_max_abs_unscaled,
        "qubo_dyn_range_unscaled": reference.qubo_dyn_range_unscaled * 1.1,
    }
    comparison = compare_table9_row(row)
    assert comparison["passed"]
    assert comparison["status"] == "within_documented_stochastic_tolerance"
    assert comparison["exact_reference_match"] is False
