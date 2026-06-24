"""Canonical paper examples built through the public package API."""

from __future__ import annotations

from typing import Any

from .builders import BuildResult, build_qubo
from .solvers import SolveResult, solve_qubo
from .variables import (
    BinaryVar,
    ContinuousVar,
    LinearConstraint,
    Problem,
    QuadraticObjective,
    Variable,
)


def _solve_and_decode(
    build: BuildResult, solver: str = "neal", solver_options: dict[str, Any] | None = None
) -> tuple[SolveResult, dict[str, float]]:
    result = solve_qubo(build.qubo, solver=solver, **(solver_options or {}))
    decoded = build.decode_sample(result.sample)
    return result, decoded


def example1_problem(digits: int, bounds: dict[str, tuple[float, float]] | None = None) -> Problem:
    bounds = bounds or {"x1": (0.0, 1.0), "x2": (0.0, 1.0)}
    a = 0.1234567
    b = 0.7654321
    return Problem(
        name="example1_unconstrained",
        variables=[
            ContinuousVar("x1", bounds["x1"][0], bounds["x1"][1], digits),
            ContinuousVar("x2", bounds["x2"][0], bounds["x2"][1], digits),
        ],
        objective=QuadraticObjective(
            constant=a * a + b * b,
            linear={"x1": -2.0 * a, "x2": -2.0 * b},
            quadratic={("x1", "x1"): 1.0, ("x2", "x2"): 1.0},
        ),
    )


def solve_example1_at_digits(
    digits: int,
    *,
    bounds: dict[str, tuple[float, float]] | None = None,
    solver: str = "neal",
    solver_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    build = build_qubo(example1_problem(digits, bounds))
    result, decoded = _solve_and_decode(build, solver, solver_options)
    objective = build.objective_value(decoded)
    return {
        "J1": digits,
        "J2": digits,
        "x1": decoded["x1"],
        "x2": decoded["x2"],
        "objective": objective,
        "energy": result.energy,
        "n_vars": len(build.qubo.variables),
        "n_quad": len(build.qubo.quadratic),
        "solver": result.solver,
    }


def reproduce_example1_bit_growth(
    *,
    solver: str = "neal",
    solver_options: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    options = solver_options or {"num_reads": 200, "sweeps": 2500, "seed": 11}
    return [
        solve_example1_at_digits(j, solver=solver, solver_options=options) for j in [2, 4, 6, 8]
    ]


def _anchored_shrink(
    original: tuple[float, float],
    current: tuple[float, float],
    value: float,
    rho: float,
    min_width: float,
) -> tuple[float, float]:
    base_l, base_u = original
    lower, upper = current
    width = max(min_width, rho * (upper - lower))
    xhat = 0.0 if upper <= lower else min(1.0, max(0.0, (value - lower) / (upper - lower)))
    new_l = value - width * xhat
    new_u = new_l + width
    if new_l < base_l:
        new_l = base_l
        new_u = base_l + width
    if new_u > base_u:
        new_u = base_u
        new_l = base_u - width
    return max(base_l, new_l), min(base_u, new_u)


def reproduce_example1_zoom(
    *,
    solver: str = "neal",
    solver_options: dict[str, Any] | None = None,
    max_iters: int = 5,
) -> list[dict[str, Any]]:
    options = solver_options or {"num_reads": 300, "sweeps": 4000, "seed": 11}
    original = {"x1": (0.0, 1.0), "x2": (0.0, 1.0)}
    bounds = dict(original)
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for iteration in range(max_iters):
        row = solve_example1_at_digits(1, bounds=bounds, solver=solver, solver_options=options)
        row["iter"] = iteration
        row["action"] = "baseline" if iteration == 0 else "accepted_zoom"
        if best is None or row["objective"] < best["objective"]:
            best = row
            rows.append(row)
        else:
            break
        bounds = {
            name: _anchored_shrink(original[name], bounds[name], row[name], 0.2, 1e-4)
            for name in ["x1", "x2"]
        }
    if rows:
        rows[-1]["action"] = "accepted_zoom_best" if len(rows) > 1 else rows[-1]["action"]
    return rows


def example2_problem(
    digits_x: int,
    digits_s: int | None = None,
    bounds_x: tuple[float, float] = (0.0, 1.0),
    bounds_s: tuple[float, float] = (0.0, 1.0),
) -> Problem:
    digits_s = digits_x if digits_s is None else digits_s
    return Problem(
        name="example2_miqp",
        variables=[
            ContinuousVar("x", bounds_x[0], bounds_x[1], digits_x),
            BinaryVar("y"),
        ],
        objective=QuadraticObjective(
            constant=0.35 * 0.35,
            linear={"x": -0.7, "y": 0.2},
            quadratic={("x", "x"): 1.0},
        ),
        constraints=[
            LinearConstraint(
                name="ineq",
                linear={"x": 1.0, "y": 0.8},
                sense="<=",
                rhs=1.0,
                penalty=100.0,
                slack_name="s",
                slack_lower=bounds_s[0],
                slack_upper=bounds_s[1],
                slack_digits=digits_s,
            )
        ],
    )


def solve_example2_at_digits(
    digits_x: int,
    digits_s: int | None = None,
    *,
    solver: str = "neal",
    solver_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    build = build_qubo(example2_problem(digits_x, digits_s))
    result, decoded = _solve_and_decode(build, solver, solver_options)
    return {
        "Jx": digits_x,
        "Js": digits_x if digits_s is None else digits_s,
        "x": decoded["x"],
        "y": int(round(decoded["y"])),
        "s": decoded["s"],
        "objective": build.objective_value(decoded),
        "feasibility": build.feasibility(decoded),
        "violation": max(0.0, decoded["x"] + 0.8 * decoded["y"] - 1.0),
        "residual": decoded["x"] + 0.8 * decoded["y"] + decoded["s"] - 1.0,
        "energy": result.energy,
        "n_vars": len(build.qubo.variables),
        "n_quad": len(build.qubo.quadratic),
        "solver": result.solver,
    }


def solve_example2_at_box(
    bounds_x: tuple[float, float],
    bounds_s: tuple[float, float],
    *,
    digits_x: int = 1,
    digits_s: int = 1,
    solver: str = "neal",
    solver_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    build = build_qubo(example2_problem(digits_x, digits_s, bounds_x, bounds_s))
    result, decoded = _solve_and_decode(build, solver, solver_options)
    return {
        "x": decoded["x"],
        "y": int(round(decoded["y"])),
        "s": decoded["s"],
        "objective": build.objective_value(decoded),
        "feasibility": build.feasibility(decoded),
        "violation": max(0.0, decoded["x"] + 0.8 * decoded["y"] - 1.0),
        "residual": decoded["x"] + 0.8 * decoded["y"] + decoded["s"] - 1.0,
        "energy": result.energy,
        "n_vars": len(build.qubo.variables),
        "n_quad": len(build.qubo.quadratic),
        "solver": result.solver,
        "bounds_x": bounds_x,
        "bounds_s": bounds_s,
    }


def reproduce_example2_zoom(
    *,
    solver: str = "neal",
    solver_options: dict[str, Any] | None = None,
    max_iters: int = 5,
) -> list[dict[str, Any]]:
    options = solver_options or {"num_reads": 300, "sweeps": 4000, "seed": 13}
    original_x = (0.0, 1.0)
    original_s = (0.0, 1.0)
    bx = original_x
    bs = original_s
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for iteration in range(max_iters):
        row = solve_example2_at_box(bx, bs, solver=solver, solver_options=options)
        row["iter"] = iteration
        row["action"] = "baseline" if iteration == 0 else "accepted_zoom"
        better = best is None or (
            row["feasibility"] <= 5e-3 and row["objective"] < best["objective"] - 1e-8
        )
        if not better:
            break
        best = row
        rows.append(row)
        bx = _anchored_shrink(original_x, bx, row["x"], 0.2, 1e-4)
        bs = _anchored_shrink(original_s, bs, row["s"], 0.2, 1e-4)
    if rows:
        rows[-1]["action"] = "accepted_zoom_best" if len(rows) > 1 else rows[-1]["action"]
    return rows


def reproduce_example2_bit_growth(
    *,
    solver: str = "neal",
    solver_options: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    options = solver_options or {"num_reads": 200, "sweeps": 2500, "seed": 13}
    return [
        solve_example2_at_digits(1, 1, solver=solver, solver_options=options),
        solve_example2_at_digits(2, 2, solver=solver, solver_options=options),
        solve_example2_at_digits(4, 4, solver=solver, solver_options=options),
    ]


def alan_problem(
    digits_x: int,
    digits_s: int,
    penalty: float = 500.0,
    x_bounds: dict[str, tuple[float, float]] | None = None,
    slack_bounds: dict[str, tuple[float, float]] | None = None,
) -> Problem:
    x_bounds = x_bounds or {f"x{i}": (0.0, 1.0) for i in range(1, 5)}
    slack_bounds = slack_bounds or {f"s{i}": (0.0, 1.0) for i in range(1, 5)}
    variables: list[Variable] = [
        ContinuousVar(f"x{i}", x_bounds[f"x{i}"][0], x_bounds[f"x{i}"][1], digits_x)
        for i in range(1, 5)
    ]
    variables.extend(BinaryVar(f"y{i}") for i in range(1, 5))
    constraints = [
        LinearConstraint(
            name="e1",
            linear={f"x{i}": 1.0 for i in range(1, 5)},
            sense="==",
            rhs=1.0,
            penalty=penalty,
        ),
        LinearConstraint(
            name="e2",
            linear={"x1": 8.0, "x2": 9.0, "x3": 12.0, "x4": 7.0},
            sense="==",
            rhs=10.0,
            penalty=penalty,
        ),
    ]
    for i in range(1, 5):
        constraints.append(
            LinearConstraint(
                name=f"link{i}",
                linear={f"x{i}": 1.0, f"y{i}": -1.0},
                sense="<=",
                rhs=0.0,
                penalty=penalty,
                slack_name=f"s{i}",
                slack_lower=slack_bounds[f"s{i}"][0],
                slack_upper=slack_bounds[f"s{i}"][1],
                slack_digits=digits_s,
            )
        )
    constraints.append(
        LinearConstraint(
            name="card",
            linear={f"y{i}": 1.0 for i in range(1, 5)},
            sense="<=",
            rhs=3.0,
            penalty=penalty,
            slack_name="sc",
            slack_upper=3.0,
            slack_type="integer",
        )
    )
    return Problem(
        name="alan",
        variables=variables,
        objective=QuadraticObjective(
            quadratic={
                ("x1", "x1"): 4.0,
                ("x1", "x2"): 6.0,
                ("x1", "x3"): -2.0,
                ("x2", "x2"): 6.0,
                ("x2", "x3"): 2.0,
                ("x3", "x3"): 10.0,
            }
        ),
        constraints=constraints,
    )


def solve_alan_at_digits(
    digits_x: int,
    digits_s: int,
    *,
    penalty: float = 500.0,
    solver: str = "neal",
    solver_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    build = build_qubo(alan_problem(digits_x, digits_s, penalty))
    result, decoded = _solve_and_decode(build, solver, solver_options)
    y_sum = sum(decoded[f"y{i}"] for i in range(1, 5))
    link_v = max(max(0.0, decoded[f"x{i}"] - decoded[f"y{i}"]) for i in range(1, 5))
    card_v = max(0.0, y_sum - 3.0)
    e1 = sum(decoded[f"x{i}"] for i in range(1, 5)) - 1.0
    e2 = 8 * decoded["x1"] + 9 * decoded["x2"] + 12 * decoded["x3"] + 7 * decoded["x4"] - 10
    return {
        "Jx": digits_x,
        "Js": digits_s,
        "x": [decoded[f"x{i}"] for i in range(1, 5)],
        "y": [int(round(decoded[f"y{i}"])) for i in range(1, 5)],
        "objective": build.objective_value(decoded),
        "feasibility": max(abs(e1), abs(e2), link_v, card_v),
        "e1_abs": abs(e1),
        "e2_abs": abs(e2),
        "link_violation": link_v,
        "card_violation": card_v,
        "energy": result.energy,
        "n_vars": len(build.qubo.variables),
        "n_quad": len(build.qubo.quadratic),
        "solver": result.solver,
    }


def solve_alan_at_box(
    x_bounds: dict[str, tuple[float, float]],
    slack_bounds: dict[str, tuple[float, float]],
    *,
    digits_x: int = 1,
    digits_s: int = 1,
    penalty: float = 500.0,
    solver: str = "neal",
    solver_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    build = build_qubo(alan_problem(digits_x, digits_s, penalty, x_bounds, slack_bounds))
    result, decoded = _solve_and_decode(build, solver, solver_options)
    y_sum = sum(decoded[f"y{i}"] for i in range(1, 5))
    link_v = max(max(0.0, decoded[f"x{i}"] - decoded[f"y{i}"]) for i in range(1, 5))
    card_v = max(0.0, y_sum - 3.0)
    e1 = sum(decoded[f"x{i}"] for i in range(1, 5)) - 1.0
    e2 = 8 * decoded["x1"] + 9 * decoded["x2"] + 12 * decoded["x3"] + 7 * decoded["x4"] - 10
    return {
        "x": [decoded[f"x{i}"] for i in range(1, 5)],
        "y": [int(round(decoded[f"y{i}"])) for i in range(1, 5)],
        "objective": build.objective_value(decoded),
        "feasibility": max(abs(e1), abs(e2), link_v, card_v),
        "e1_abs": abs(e1),
        "e2_abs": abs(e2),
        "link_violation": link_v,
        "card_violation": card_v,
        "energy": result.energy,
        "n_vars": len(build.qubo.variables),
        "n_quad": len(build.qubo.quadratic),
        "solver": result.solver,
        "decoded": decoded,
    }


def reproduce_alan_zoom(
    *,
    solver: str = "neal",
    solver_options: dict[str, Any] | None = None,
    penalty: float = 500.0,
) -> list[dict[str, Any]]:
    options = solver_options or {"num_reads": 300, "sweeps": 4000, "seed": 13}
    original_x = {f"x{i}": (0.0, 1.0) for i in range(1, 5)}
    original_s = {f"s{i}": (0.0, 1.0) for i in range(1, 5)}
    base = solve_alan_at_box(
        dict(original_x),
        dict(original_s),
        penalty=penalty,
        solver=solver,
        solver_options=options,
    )
    base["iter"] = 0
    base["action"] = "baseline"
    x_bounds = {
        name: _anchored_shrink(original_x[name], original_x[name], base["decoded"][name], 0.2, 1e-4)
        for name in original_x
    }
    s_bounds = {
        name: _anchored_shrink(original_s[name], original_s[name], base["decoded"][name], 0.2, 1e-4)
        for name in original_s
    }
    zoom = solve_alan_at_box(
        x_bounds,
        s_bounds,
        penalty=penalty,
        solver=solver,
        solver_options=options,
    )
    zoom["iter"] = 1
    zoom["action"] = "accepted_zoom" if zoom["objective"] <= base["objective"] else "zoom_trial"
    for row in (base, zoom):
        row.pop("decoded", None)
    return [base, zoom]


def reproduce_alan_penalty_sensitivity(
    *,
    solver: str = "neal",
    solver_options: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    options = solver_options or {"num_reads": 300, "sweeps": 4000, "seed": 13}
    rows: list[dict[str, Any]] = []
    for penalty in [50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0]:
        row = solve_alan_at_digits(
            1,
            1,
            penalty=penalty,
            solver=solver,
            solver_options=options,
        )
        row["lambda"] = penalty
        rows.append(row)
    return rows


def reproduce_alan_bit_growth(
    *,
    solver: str = "neal",
    solver_options: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    options = solver_options or {"num_reads": 300, "sweeps": 3000, "seed": 7}
    rows = []
    for jx, js in [(1, 1), (2, 2), (3, 3), (4, 4), (5, 4), (8, 8)]:
        rows.append(solve_alan_at_digits(jx, js, solver=solver, solver_options=options))
    return rows
