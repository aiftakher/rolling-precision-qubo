"""Canonical paper examples built through the public package API."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import Any, Literal

from .builders import BuildResult, build_qubo
from .paper_reference import (
    alan_penalty_sensitivity_reference,
    alan_table6_config,
    rolling_alan_zoom_reference,
    rolling_example2_zoom_reference,
    solve_alan_bit_growth_reference_at_precision,
)
from .rolling import (
    AcceptanceConfig,
    Box,
    ZoomMove,
    rolling_zoom_in,
)
from .solvers import SolveResult, solve_qubo
from .variables import (
    BinaryVar,
    ContinuousVar,
    LinearConstraint,
    Problem,
    QuadraticObjective,
    Variable,
)

CardinalitySlackEncoding = Literal["sbe", "integer"]


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
    #objective = build.objective_value(decoded)
    a = 0.1234567
    b = 0.7654321
    objective = (decoded["x1"] - a) ** 2 + (decoded["x2"] - b) ** 2
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
    max_iters: int = 15,
) -> list[dict[str, Any]]:
    if solver == "neal" and solver_options is None:
        report = rolling_example2_zoom_reference()
        rows: list[dict[str, Any]] = []
        saw_backtrack = False
        for row in report["history"]:
            if row["action"] not in {"baseline", "accepted_zoom", "backtrack"}:
                continue
            rows.append(row)
            if row["action"] == "backtrack":
                saw_backtrack = True
                continue
            if saw_backtrack and row["action"] == "accepted_zoom":
                break
        return rows[: max_iters + 1]

    options = solver_options or {
        "num_reads": 300,
        "sweeps": 4000,
        "seed": 13,
    }

    initial_box: Box = {
        "x": (0.0, 1.0),
        "s": (0.0, 1.0),
    }

    def solve_at(box: Box) -> dict[str, Any]:
        result = solve_example2_at_box(
            bounds_x=box["x"],
            bounds_s=box["s"],
            digits_x=1,
            digits_s=1,
            solver=solver,
            solver_options=options,
        )

        # rolling_zoom_in can use this directly instead of a point_getter.
        result["point"] = {
            "x": result["x"],
            "s": result["s"],
        }
        return result

    report = rolling_zoom_in(
        initial_box,
        solve_at,
        candidate_moves=[
            ZoomMove("zoom_xs", ("x", "s")),
            ZoomMove("zoom_x", ("x",)),
            ZoomMove("zoom_s", ("s",)),
        ],
        rho=0.2,
        min_width={
            "x": 1e-4,
            "s": 1e-4,
        },
        acceptance=AcceptanceConfig(
            criterion="feasibility_first",
            feasibility_tol=5e-3,
            feasibility_eps=1e-6,
            objective_eps=1e-8,
        ),
        max_iters=max_iters,
    )

    # report = rolling_zoom_in(...)
    # print(report["incumbent"])
    # print(report["termination_reason"])
    # print(report["accepted_count"])
    # print(report["backtrack_count"])

    # Return only rows useful for the paper table/report.
    return [
        row
        for row in report["history"]
        if row["action"] in {
            "baseline",
            "accepted_zoom",
            "backtrack",
        }
    ]


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
    *,
    penalties: Mapping[str, float] | None = None,
    x_bounds: dict[str, tuple[float, float]] | None = None,
    slack_bounds: dict[str, tuple[float, float]] | None = None,
    cardinality_slack_encoding: CardinalitySlackEncoding = "sbe",
) -> Problem:
    """Build the Alan MINLPLib problem.

    Parameters
    ----------
    digits_x:
        Number of decimal SBE digits used for x1, ..., x4.

    digits_s:
        Number of decimal SBE digits used for the four linking slacks
        s1, ..., s4. It is also used for the cardinality slack when
        ``cardinality_slack_encoding="sbe"``.

    penalties:
        Penalty weights with the required keys:

        - ``"e1"``: sum(x_i) = 1
        - ``"e2"``: weighted-sum equality
        - ``"link"``: x_i <= y_i constraints
        - ``"card"``: sum(y_i) <= 3

    x_bounds:
        Current bounds for x1, ..., x4. These may be tightened during zoom-in.

    slack_bounds:
        Current bounds for the four linking slacks s1, ..., s4.
        These may be tightened during zoom-in.

    cardinality_slack_encoding:
        ``"sbe"`` represents sc in [0, 3] using decimal SBE with
        ``4 * digits_s + 1`` bits. Use this for the bit-growth
        formulation and Table 5 QUBO sizes.

        ``"integer"`` represents sc in {0, 1, 2, 3} using two bounded
        binary bits. Use this for the legacy constant-size zoom and
        penalty-sensitivity implementation.
    """
    if digits_x < 1:
        raise ValueError("digits_x must be at least 1")
    if digits_s < 1:
        raise ValueError("digits_s must be at least 1")

    if cardinality_slack_encoding not in {"sbe", "integer"}:
        raise ValueError(
            "cardinality_slack_encoding must be either 'sbe' or 'integer'"
        )

    required_penalties = {"e1", "e2", "link", "card"}

    if penalties is None:
        penalty_values = {
            "e1": 500.0,
            "e2": 500.0,
            "link": 500.0,
            "card": 500.0,
        }
    else:
        missing = required_penalties - set(penalties)
        unknown = set(penalties) - required_penalties

        if missing:
            raise ValueError(
                f"Missing Alan penalty values: {sorted(missing)}"
            )
        if unknown:
            raise ValueError(
                f"Unknown Alan penalty names: {sorted(unknown)}"
            )

        penalty_values = {
            name: float(penalties[name])
            for name in required_penalties
        }

    for name, value in penalty_values.items():
        if not isfinite(value) or value <= 0.0:
            raise ValueError(
                f"Penalty {name!r} must be finite and strictly positive; "
                f"received {value}"
            )

    expected_x_names = {f"x{i}" for i in range(1, 5)}
    expected_slack_names = {f"s{i}" for i in range(1, 5)}

    if x_bounds is None:
        x_bounds = {
            name: (0.0, 1.0)
            for name in sorted(expected_x_names)
        }
    else:
        x_bounds = {
            name: (float(lower), float(upper))
            for name, (lower, upper) in x_bounds.items()
        }

    if slack_bounds is None:
        slack_bounds = {
            name: (0.0, 1.0)
            for name in sorted(expected_slack_names)
        }
    else:
        slack_bounds = {
            name: (float(lower), float(upper))
            for name, (lower, upper) in slack_bounds.items()
        }

    missing_x = expected_x_names - set(x_bounds)
    extra_x = set(x_bounds) - expected_x_names
    if missing_x or extra_x:
        raise ValueError(
            "x_bounds must contain exactly x1, x2, x3, and x4. "
            f"Missing={sorted(missing_x)}, extra={sorted(extra_x)}"
        )

    missing_s = expected_slack_names - set(slack_bounds)
    extra_s = set(slack_bounds) - expected_slack_names
    if missing_s or extra_s:
        raise ValueError(
            "slack_bounds must contain exactly s1, s2, s3, and s4. "
            f"Missing={sorted(missing_s)}, extra={sorted(extra_s)}"
        )

    for name, (lower, upper) in {
        **x_bounds,
        **slack_bounds,
    }.items():
        if not isfinite(lower) or not isfinite(upper):
            raise ValueError(
                f"{name}: bounds must be finite; received {(lower, upper)}"
            )
        if upper < lower:
            raise ValueError(
                f"{name}: upper bound must be at least the lower bound; "
                f"received {(lower, upper)}"
            )

    # Original decision variables. Slack variables are created automatically
    # by build_qubo() from the LinearConstraint configurations below.
    variables: list[Variable] = [
        ContinuousVar(
            name=f"x{i}",
            lower=x_bounds[f"x{i}"][0],
            upper=x_bounds[f"x{i}"][1],
            digits=digits_x,
            encoding="sbe",
        )
        for i in range(1, 5)
    ]

    variables.extend(
        BinaryVar(name=f"y{i}")
        for i in range(1, 5)
    )

    constraints: list[LinearConstraint] = [
        # x1 + x2 + x3 + x4 = 1
        LinearConstraint(
            name="e1",
            linear={
                "x1": 1.0,
                "x2": 1.0,
                "x3": 1.0,
                "x4": 1.0,
            },
            sense="==",
            rhs=1.0,
            penalty=penalty_values["e1"],
        ),

        # 8*x1 + 9*x2 + 12*x3 + 7*x4 = 10
        LinearConstraint(
            name="e2",
            linear={
                "x1": 8.0,
                "x2": 9.0,
                "x3": 12.0,
                "x4": 7.0,
            },
            sense="==",
            rhs=10.0,
            penalty=penalty_values["e2"],
        ),
    ]

    # Linking inequalities:
    #
    #     x_i <= y_i
    #
    # are rewritten as:
    #
    #     x_i - y_i + s_i = 0,
    #
    # where s_i is an SBE-encoded slack in its current zoom box.
    for i in range(1, 5):
        slack_name = f"s{i}"
        slack_lower, slack_upper = slack_bounds[slack_name]

        constraints.append(
            LinearConstraint(
                name=f"link{i}",
                linear={
                    f"x{i}": 1.0,
                    f"y{i}": -1.0,
                },
                sense="<=",
                rhs=0.0,
                penalty=penalty_values["link"],
                slack_name=slack_name,
                slack_lower=slack_lower,
                slack_upper=slack_upper,
                slack_digits=digits_s,
                slack_encoding="sbe",
                slack_type="continuous",
            )
        )

    # Cardinality inequality:
    #
    #     y1 + y2 + y3 + y4 <= 3
    #
    # becomes:
    #
    #     y1 + y2 + y3 + y4 + sc = 3.
    #
    if cardinality_slack_encoding == "sbe":
        cardinality_constraint = LinearConstraint(
            name="card",
            linear={
                "y1": 1.0,
                "y2": 1.0,
                "y3": 1.0,
                "y4": 1.0,
            },
            sense="<=",
            rhs=3.0,
            penalty=penalty_values["card"],
            slack_name="sc",
            slack_lower=0.0,
            slack_upper=3.0,
            slack_digits=digits_s,
            slack_encoding="sbe",
            slack_type="continuous",
        )
    else:
        cardinality_constraint = LinearConstraint(
            name="card",
            linear={
                "y1": 1.0,
                "y2": 1.0,
                "y3": 1.0,
                "y4": 1.0,
            },
            sense="<=",
            rhs=3.0,
            penalty=penalty_values["card"],
            slack_name="sc",
            slack_lower=0.0,
            slack_upper=3.0,
            slack_type="integer",
        )

    constraints.append(cardinality_constraint)

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

    common_penalties = {
        "e1": penalty,
        "e2": penalty,
        "link": penalty,
        "card": penalty,
    }
    build = build_qubo(
        alan_problem(
            digits_x,
            digits_s,
            penalties=common_penalties,
            cardinality_slack_encoding="sbe",
        )
    )
    result, decoded = _solve_and_decode(build, solver, solver_options)
    x_values = {f"x{i}": decoded[f"x{i}"] for i in range(1, 5)}
    y_values = {f"y{i}": int(round(decoded[f"y{i}"])) for i in range(1, 5)}
    s_values = {f"s{i}": decoded[f"s{i}"] for i in range(1, 5)}
    y_sum = sum(y_values.values())
    link_v = max(max(0.0, x_values[f"x{i}"] - y_values[f"y{i}"]) for i in range(1, 5))
    card_v = max(0.0, y_sum - 3.0)
    e1 = sum(x_values.values()) - 1.0
    e2 = 8 * decoded["x1"] + 9 * decoded["x2"] + 12 * decoded["x3"] + 7 * decoded["x4"] - 10
    link_residual = max(
        abs(x_values[f"x{i}"] - y_values[f"y{i}"] + s_values[f"s{i}"]) for i in range(1, 5)
    )
    card_residual = y_sum + decoded["sc"] - 3.0
    objective = (
        4 * decoded["x1"] ** 2
        + 6 * decoded["x1"] * decoded["x2"]
        - 2 * decoded["x1"] * decoded["x3"]
        + 6 * decoded["x2"] ** 2
        + 2 * decoded["x2"] * decoded["x3"]
        + 10 * decoded["x3"] ** 2
    )
    return {
        "Jx": digits_x,
        "Js": digits_s,
        "x": x_values,
        "y": y_values,
        "s": s_values,
        "sc": decoded["sc"],
        "objective": objective,
        "feasibility": max(abs(e1), abs(e2), link_v, card_v),
        "e1": e1,
        "e2": e2,
        "e1_abs": abs(e1),
        "e2_abs": abs(e2),
        "link_violation": link_v,
        "link_residual": link_residual,
        "card_violation": card_v,
        "card_residual": card_residual,
        "energy": result.energy,
        "n_vars": len(build.qubo.variables),
        "n_quad": len(build.qubo.quadratic),
        "solver": result.solver,
        "solver_metadata": result.metadata,
    }


def solve_alan_at_box(
    x_bounds: dict[str, tuple[float, float]],
    slack_bounds: dict[str, tuple[float, float]],
    *,
    digits_x: int = 1,
    digits_s: int = 1,
    penalty: float = 500.0,
    penalties: Mapping[str, float] | None = None,
    solver: str = "neal",
    solver_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    common_penalties = penalties or {
        "e1": penalty,
        "e2": penalty,
        "link": penalty,
        "card": penalty,
    }

    build = build_qubo(
        alan_problem(
            digits_x,
            digits_s,
            penalties=common_penalties,
            x_bounds=x_bounds,
            slack_bounds=slack_bounds,
            cardinality_slack_encoding="integer",
        )
    )

    result, decoded = _solve_and_decode(build, solver, solver_options)
    x_values = {f"x{i}": decoded[f"x{i}"] for i in range(1, 5)}
    y_values = {f"y{i}": int(round(decoded[f"y{i}"])) for i in range(1, 5)}
    s_values = {f"s{i}": decoded[f"s{i}"] for i in range(1, 5)}
    y_sum = sum(y_values.values())
    link_v = max(max(0.0, x_values[f"x{i}"] - y_values[f"y{i}"]) for i in range(1, 5))
    card_v = max(0.0, y_sum - 3.0)
    e1 = sum(x_values.values()) - 1.0
    e2 = 8 * decoded["x1"] + 9 * decoded["x2"] + 12 * decoded["x3"] + 7 * decoded["x4"] - 10
    link_residual = max(
        abs(x_values[f"x{i}"] - y_values[f"y{i}"] + s_values[f"s{i}"]) for i in range(1, 5)
    )
    card_residual = y_sum + decoded["sc"] - 3.0
    objective = (
        4 * decoded["x1"] ** 2
        + 6 * decoded["x1"] * decoded["x2"]
        - 2 * decoded["x1"] * decoded["x3"]
        + 6 * decoded["x2"] ** 2
        + 2 * decoded["x2"] * decoded["x3"]
        + 10 * decoded["x3"] ** 2
    )
    return {
        "x": x_values,
        "y": y_values,
        "s": s_values,
        "sc": decoded["sc"],
        "objective": objective,
        "feasibility": max(abs(e1), abs(e2), link_v, card_v),
        "e1": e1,
        "e2": e2,
        "e1_abs": abs(e1),
        "e2_abs": abs(e2),
        "link_violation": link_v,
        "link_residual": link_residual,
        "card_violation": card_v,
        "card_residual": card_residual,
        "energy": result.energy,
        "n_vars": len(build.qubo.variables),
        "n_quad": len(build.qubo.quadratic),
        "solver": result.solver,
        "solver_metadata": result.metadata,
        "decoded": decoded,
    }



def reproduce_alan_zoom(
    *,
    solver: str = "neal",
    solver_options: dict[str, Any] | None = None,
    max_iters: int = 15,
    mode: str = "paper_table6_reference",
) -> list[dict[str, Any]]:
    if solver != "neal" or solver_options is not None:
        raise ValueError("Alan paper-reference zoom uses the configured neal settings")
    config = alan_table6_config(mode)
    report = rolling_alan_zoom_reference(config=config)
    return [
        row
        for row in report["history"]
        if row["action"] in {"baseline", "accepted_zoom", "backtrack"}
    ][: max_iters + 1]


def reproduce_alan_penalty_sensitivity(
    *,
    solver: str = "neal",
    solver_options: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if solver != "neal" or solver_options is not None:
        raise ValueError("Alan penalty sensitivity uses the configured neal settings")
    return alan_penalty_sensitivity_reference()


def _reproduce_alan_zoom_generic(
    *,
    solver: str = "neal",
    solver_options: dict[str, Any] | None = None,
    max_iters: int = 15,
) -> list[dict[str, Any]]:
    options = solver_options or {
        "num_reads": 300,
        "sweeps": 4000,
        "seed": 13,
    }

    initial_box: Box = {
        **{f"x{i}": (0.0, 1.0) for i in range(1, 5)},
        **{f"s{i}": (0.0, 1.0) for i in range(1, 5)},
    }

    x_names = tuple(f"x{i}" for i in range(1, 5))
    slack_names = tuple(f"s{i}" for i in range(1, 5))

    base_penalties = {
        "e1": 200.0,
        "e2": 200.0,
        "link": 300.0,
        "card": 200.0,
    }

    def solve_at(box: Box) -> dict[str, Any]:
        x_bounds = {
            name: box[name]
            for name in x_names
        }
        slack_bounds = {
            name: box[name]
            for name in slack_names
        }

        result = solve_alan_at_box(
            x_bounds=x_bounds,
            slack_bounds=slack_bounds,
            digits_x=1,
            digits_s=1,
            penalties=base_penalties,
            solver=solver,
            solver_options=options,
        )

        # Adjust these two lines if your result currently uses different keys.
        x_values = result["x"]
        slack_values = result["s"]

        result["point"] = {
            **x_values,
            **slack_values,
        }
        return result

    report = rolling_zoom_in(
        initial_box,
        solve_at,
        candidate_moves=[
            ZoomMove(
                "zoom_xs",
                x_names + slack_names,
            ),
            ZoomMove(
                "zoom_x",
                x_names,
            ),
            ZoomMove(
                "zoom_s",
                slack_names,
            ),
        ],
        rho=0.2,
        min_width={
            **{name: 1e-4 for name in x_names},
            **{name: 1e-4 for name in slack_names},
        },
        acceptance=AcceptanceConfig(
            criterion="feasibility_first",
            feasibility_tol=5e-3,
            feasibility_eps=1e-6,
            objective_eps=1e-8,
        ),
        max_iters=max_iters,
    )

    return [
        row
        for row in report["history"]
        if row["action"] in {
            "baseline",
            "accepted_zoom",
            "backtrack",
        }
    ]




def reproduce_alan_bit_growth(
    *,
    solver: str = "neal",
    solver_options: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if solver != "neal" or solver_options is not None:
        options = solver_options or {"num_reads": 300, "sweeps": 3000, "seed": 7}
        rows = []
        for jx, js in [(1, 1), (2, 2), (3, 3), (4, 4), (5, 4), (8, 8)]:
            rows.append(solve_alan_at_digits(jx, js, solver=solver, solver_options=options))
        return rows
    rows = []
    for jx, js in [(1, 1), (2, 2), (3, 3), (4, 4), (5, 4), (8, 8)]:
        rows.append(solve_alan_bit_growth_reference_at_precision(jx, js))
    return rows
