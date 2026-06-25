"""Metrics and residual helpers."""

from __future__ import annotations

from .qubo import QUBO, coefficient_stats


def qubo_summary(qubo: QUBO) -> dict[str, object]:
    stats = coefficient_stats(qubo)
    return {
        "n_variables": int(stats["n_variables"]),
        "n_linear": int(stats["n_linear"]),
        "n_quadratic": int(stats["n_quadratic"]),
        "min_coeff": stats["min_coeff"],
        "max_coeff": stats["max_coeff"],
        "max_abs": stats["max_abs"],
        "min_nonzero_abs": stats["min_nonzero_abs"],
        "dynamic_range": stats["dynamic_range"],
        "density": stats["density"],
        "offset": qubo.offset,
        "variable_groups": qubo.variable_groups,
    }
