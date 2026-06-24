from __future__ import annotations

import pytest

from rpqubo.examples import reproduce_alan_penalty_sensitivity


@pytest.mark.slow
def test_alan_penalty_sensitivity_regenerates_table9() -> None:
    rows = reproduce_alan_penalty_sensitivity()
    by_lambda = {float(row["lambda_input"]): row for row in rows}
    expected = {
        50.0: (1.7200, 1.0e-1, 1.0e-1, 1.78e-15, 2.9721e3, 5.9442e3),
        100.0: (2.1472, 6.0e-2, 6.0e-2, 2.0e-2, 8.0e2, 2.0e4),
        250.0: (3.101376, 0.0, 0.0, 1.78e-15, 2.0e3, 2.0e4),
        500.0: (3.0, 0.0, 0.0, 0.0, 2.97291e4, 5.94582e3),
        1000.0: (2.9424502, 4.416e-3, 4.416e-3, 8.64e-4, 8.0e3, 3.90625e9),
        2500.0: (3.1125214, 2.976e-3, 2.976e-3, 4.48e-4, 2.0e4, 8.0e10),
        5000.0: (3.2480, 0.0, 0.0, 0.0, 4.0e4, 2.0e4),
    }
    for lambda_value, values in expected.items():
        row = by_lambda[lambda_value]
        objective, feasibility, e1_abs, e2_abs, qmax, dyn = values
        assert abs(float(row["objective"]) - objective) <= 1e-6
        assert abs(float(row["feasibility"]) - feasibility) <= 1e-9
        assert abs(float(row["e1_abs"]) - e1_abs) <= 1e-9
        assert abs(float(row["e2_abs"]) - e2_abs) <= 1e-9
        assert abs(float(row["qubo_max_abs_unscaled"]) - qmax) <= max(1e-9, qmax * 1e-9)
        assert abs(float(row["qubo_dyn_range_unscaled"]) - dyn) <= max(1e-6, dyn * 1e-8)
        assert row["n_vars"] == 46
        assert row["n_quad"] == 385
