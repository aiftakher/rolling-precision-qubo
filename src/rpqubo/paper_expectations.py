"""Immutable expectations for paper-example reproducibility checks."""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
TABLE9_REFERENCE_PATH = REPO_ROOT / "data" / "alan_penalty_sensitivity.csv"
TABLE9_REFERENCE_SHA256 = "1d4b2a414798a6549b5bd25e6bbb50c73880a86e1028d971957793c361f5a556"

VALUE_ATOL = 1e-9
OBJECTIVE_ATOL = 1e-6
FEASIBILITY_ATOL = 1e-9

TABLE1 = (
    {
        "J1": 2,
        "J2": 2,
        "x1": 0.12,
        "x2": 0.77,
        "objective": 3.28144853000016e-05,
        "n_vars": 18,
        "n_quad": 72,
    },
    {
        "J1": 4,
        "J2": 4,
        "x1": 0.1235,
        "x2": 0.7654,
        "objective": 2.9052999999924134e-09,
        "n_vars": 34,
        "n_quad": 272,
    },
    {
        "J1": 6,
        "J2": 6,
        "x1": 0.123457,
        "x2": 0.765432,
        "objective": 9.999999998077112e-14,
        "n_vars": 50,
        "n_quad": 600,
    },
    {
        "J1": 8,
        "J2": 8,
        "x1": 0.1234567,
        "x2": 0.7654321,
        "objective": 4.949639957075196e-32,
        "n_vars": 66,
        "n_quad": 1056,
    },
)

TABLE2 = (
    {"x1": 0.1, "x2": 0.8, "action": "baseline"},
    {"x1": 0.12, "x2": 0.76, "action": "accepted_zoom"},
    {"x1": 0.124, "x2": 0.764, "action": "accepted_zoom"},
    {"x1": 0.1232, "x2": 0.7656, "action": "accepted_zoom"},
    {"x1": 0.12352, "x2": 0.76544, "action": "accepted_zoom_best"},
)

TABLE3 = (
    {"Jx": 1, "Js": 1, "objective": 0.0025, "feasibility": 0.0, "n_vars": 11, "n_quad": 55},
    {
        "Jx": 2,
        "Js": 2,
        "x": 0.35,
        "y": 0,
        "s": 0.65,
        "objective": 0.0,
        "feasibility": 0.0,
        "n_vars": 19,
        "n_quad": 171,
    },
    {
        "Jx": 4,
        "Js": 4,
        "x": 0.35,
        "y": 0,
        "s": 0.65,
        "objective": 0.0,
        "feasibility": 0.0,
        "n_vars": 35,
        "n_quad": 595,
    },
)
TABLE3_J1_TIED_STATES = ((0.4, 0, 0.6), (0.3, 0, 0.7))

TABLE4 = (
    {"x": 0.4, "action": "baseline"},
    {"x": 0.36, "action": "accepted_zoom"},
    {"x": 0.352, "action": "accepted_zoom"},
    {"x": 0.352, "action": "backtrack"},
    {"x": 0.3496, "action": "accepted_zoom"},
)

TABLE5 = (
    {
        "Jx": 1,
        "Js": 1,
        "x": (0.3, 0.0, 0.4, 0.4),
        "objective": 1.72,
        "feasibility": 0.1,
        "n_vars": 49,
        "n_quad": 406,
    },
    {
        "Jx": 2,
        "Js": 2,
        "x": (0.34, 0.33, 0.36, 0.0),
        "objective": 3.0778,
        "feasibility": 0.03,
        "n_vars": 85,
        "n_quad": 1248,
    },
    {
        "Jx": 3,
        "Js": 3,
        "x": (0.297, 0.264, 0.435, 0.004),
        "objective": 3.105,
        "feasibility": 0.004,
        "n_vars": 121,
        "n_quad": 2554,
    },
    {
        "Jx": 4,
        "Js": 4,
        "x": (0.498, 0.087, 0.434, 0.0031),
        "objective": 2.824198,
        "feasibility": 0.0221,
        "n_vars": 157,
        "n_quad": 4324,
    },
    {
        "Jx": 5,
        "Js": 4,
        "x": (0.315, 0.0033, 0.5333, 0.15),
        "objective": 2.91483202,
        "feasibility": 0.0033,
        "n_vars": 173,
        "n_quad": 5820,
    },
    {
        "Jx": 8,
        "Js": 8,
        "x": (0.3967, 0.1813, 0.4269, 0.01),
        "objective": 2.89675954,
        "feasibility": 0.0149,
        "n_vars": 301,
        "n_quad": 16044,
    },
)

TABLE6_ACTIONS = ("baseline", "accepted_zoom", "backtrack")
TABLE6_BASELINE_X = (0.5, 0.0, 0.5, 0.0)
TABLE6_ACCEPTED_X = (0.4, 0.0, 0.52, 0.08)
TABLE6_ACCEPTED_OBJECTIVE = 2.928

NONLINEAR = (
    {"a": 0.6, "J": 1.0, "max_abs_error": 1.1102230246251565e-16, "rmse": 5.021172554038838e-17},
    {"a": 0.6, "J": 2.0, "max_abs_error": 0.180553375376858, "rmse": 0.10947070387325102},
    {"a": 0.6, "J": 3.0, "max_abs_error": 0.23437195366963415, "rmse": 0.13947465241652124},
    {"a": 1.5, "J": 1.0, "max_abs_error": 1.1102230246251565e-16, "rmse": 5.021172554038838e-17},
    {"a": 1.5, "J": 2.0, "max_abs_error": 0.10422259449009119, "rmse": 0.042623506543928644},
    {"a": 1.5, "J": 3.0, "max_abs_error": 0.11683159184881531, "rmse": 0.04677738492225123},
)


@dataclass(frozen=True)
class Table9Expectation:
    lambda_input: float
    objective: float
    feasibility: float
    e1_abs: float
    e2_abs: float
    n_vars: int
    n_quad: int
    qubo_max_abs_unscaled: float
    qubo_dyn_range_unscaled: float


TABLE9 = (
    Table9Expectation(
        50.0,
        1.7200000000000004,
        0.10000000000000009,
        0.10000000000000009,
        1.7763568394002505e-15,
        46,
        385,
        2972.1000000000004,
        5944.2,
    ),
    Table9Expectation(
        100.0,
        2.1472,
        0.06000000000000005,
        0.06000000000000005,
        0.019999999999997797,
        46,
        385,
        800.0,
        19999.99999999999,
    ),
    Table9Expectation(250.0, 3.101376, 0.0, 0.0, 0.0, 46, 385, 2000.0, 19999.99999999999),
    Table9Expectation(500.0, 3.0, 0.0, 0.0, 0.0, 46, 385, 29729.100000000002, 5945.82),
    Table9Expectation(
        1000.0,
        2.946161074240001,
        0.004054000000000002,
        0.004054000000000002,
        0.0007019999999986481,
        46,
        385,
        8000.0,
        12500000000.001886,
    ),
    Table9Expectation(
        2500.0,
        3.232179980861439,
        0.0029824000000000517,
        0.0029824000000000517,
        0.00046080000000259247,
        46,
        385,
        20000.0,
        3906250000.000318,
    ),
    Table9Expectation(5000.0, 3.248, 0.0, 0.0, 0.0, 46, 385, 40000.0, 19999.999999999993),
)
TABLE9_BY_LAMBDA = {row.lambda_input: row for row in TABLE9}


def reference_table9_sha256(path: Path = TABLE9_REFERENCE_PATH) -> str | None:
    """Return the SHA256 for the accepted Table 9 CSV when it is available."""

    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_table9_reference_csv(path: Path = TABLE9_REFERENCE_PATH) -> tuple[Table9Expectation, ...]:
    """Load accepted Table 9 expectations from the immutable reference CSV."""

    rows: list[Table9Expectation] = []
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            rows.append(
                Table9Expectation(
                    lambda_input=float(row["lambda_input"]),
                    objective=float(row["obj"]),
                    feasibility=float(row["feas"]),
                    e1_abs=float(row["e1_abs"]),
                    e2_abs=float(row["e2_abs"]),
                    n_vars=int(row["n_vars"]),
                    n_quad=int(row["n_quad"]),
                    qubo_max_abs_unscaled=float(row["qubo_max_abs_unscaled"]),
                    qubo_dyn_range_unscaled=float(row["qubo_dyn_range_unscaled"]),
                )
            )
    return tuple(rows)


def _float(row: Mapping[str, Any], *names: str) -> float:
    for name in names:
        if name in row:
            return float(row[name])
    raise KeyError(names[0])


def _int(row: Mapping[str, Any], name: str) -> int:
    return int(row[name])


def _rel_close(observed: float, expected: float, rel_tol: float, abs_tol: float = 1e-12) -> bool:
    return abs(observed - expected) <= max(abs_tol, abs(expected) * rel_tol)


def compare_table9_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Classify one generated Table 9 row against the accepted-paper reference."""

    lambda_input = _float(row, "lambda_input")
    expected = TABLE9_BY_LAMBDA[lambda_input]
    observed = {
        "objective": _float(row, "objective", "obj"),
        "feasibility": _float(row, "feasibility", "feas"),
        "e1_abs": _float(row, "e1_abs"),
        "e2_abs": _float(row, "e2_abs"),
        "n_vars": _int(row, "n_vars"),
        "n_quad": _int(row, "n_quad"),
        "qubo_max_abs_unscaled": _float(row, "qubo_max_abs_unscaled"),
        "qubo_dyn_range_unscaled": _float(row, "qubo_dyn_range_unscaled"),
    }
    deltas = {
        "objective": observed["objective"] - expected.objective,
        "feasibility": observed["feasibility"] - expected.feasibility,
        "e1_abs": observed["e1_abs"] - expected.e1_abs,
        "e2_abs": observed["e2_abs"] - expected.e2_abs,
        "qubo_max_abs_unscaled": observed["qubo_max_abs_unscaled"] - expected.qubo_max_abs_unscaled,
        "qubo_dyn_range_unscaled": observed["qubo_dyn_range_unscaled"]
        - expected.qubo_dyn_range_unscaled,
    }
    dims_match = observed["n_vars"] == expected.n_vars and observed["n_quad"] == expected.n_quad
    exact = (
        dims_match
        and abs(deltas["objective"]) <= OBJECTIVE_ATOL
        and abs(deltas["feasibility"]) <= FEASIBILITY_ATOL
        and abs(deltas["e1_abs"]) <= FEASIBILITY_ATOL
        and abs(deltas["e2_abs"]) <= FEASIBILITY_ATOL
        and _rel_close(
            observed["qubo_max_abs_unscaled"],
            expected.qubo_max_abs_unscaled,
            1e-8,
        )
        and _rel_close(
            observed["qubo_dyn_range_unscaled"],
            expected.qubo_dyn_range_unscaled,
            1e-8,
        )
    )
    stochastic = (
        dims_match
        and abs(deltas["objective"]) <= max(0.15, 0.05 * abs(expected.objective))
        and observed["feasibility"] <= max(0.01, expected.feasibility + 0.005)
        and observed["e1_abs"] <= max(0.01, expected.e1_abs + 0.005)
        and observed["e2_abs"] <= max(0.01, expected.e2_abs + 0.005)
    )
    status = (
        "exact_reference_match"
        if exact
        else "within_documented_stochastic_tolerance"
        if stochastic
        else "failed"
    )
    return {
        "lambda_input": lambda_input,
        "passed": exact or stochastic,
        "status": status,
        "exact_reference_match": exact,
        "within_stochastic_tolerance": stochastic,
        "details": {
            "observed": observed,
            "reference": expected.__dict__,
            "deltas": deltas,
        },
    }
