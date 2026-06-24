from __future__ import annotations

from rpqubo.examples import solve_example1_at_digits, solve_example2_at_digits


def test_exact_solver_returns_known_example1_small_grid_optimum() -> None:
    row = solve_example1_at_digits(1, solver="exact")
    assert row["x1"] == 0.1
    assert row["x2"] == 0.8
    assert abs(row["objective"] - 0.0017451564853001589) <= 1e-15


def test_exact_solver_returns_example2_j2_optimum() -> None:
    row = solve_example2_at_digits(2, 2, solver="exact")
    assert abs(row["x"] - 0.35) <= 1e-12
    assert row["y"] == 0
    assert abs(row["s"] - 0.65) <= 1e-12
    assert row["feasibility"] <= 1e-12
