"""QUBO solvers with optional dependencies."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata
from itertools import product
from math import isfinite
from random import Random
from typing import Any, cast

from .qubo import QUBO


@dataclass
class SolveResult:
    sample: dict[str, int]
    energy: float
    solver: str
    metadata: dict[str, object] = field(default_factory=dict)


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("dimod", "dwave-neal", "dwave-samplers", "numpy"):
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            continue
    return versions


def _ordered_variables(qubo: QUBO) -> list[str]:
    return qubo.ordered_variables


def _positive_int(value: int, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _finite_positive_float(value: float, name: str) -> float:
    value = float(value)
    if not isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _base_metadata(qubo: QUBO) -> dict[str, object]:
    return {
        "package_versions": _package_versions(),
        "bqm_variable_order": _ordered_variables(qubo),
    }


def solve_exact(qubo: QUBO, *, max_variables: int = 24) -> SolveResult:
    max_variables = _positive_int(max_variables, "max_variables")
    variables = _ordered_variables(qubo)
    if len(variables) > max_variables:
        raise ValueError(
            f"Exact enumeration requested for {len(variables)} variables; limit is {max_variables}"
        )
    best_energy = float("inf")
    best_sample: dict[str, int] = {}
    for values in product([0, 1], repeat=len(variables)):
        sample = dict(zip(variables, values))
        energy = qubo.energy(sample)
        if energy < best_energy:
            best_energy = energy
            best_sample = sample
    return SolveResult(
        sample=best_sample,
        energy=best_energy,
        solver="exact",
        metadata={**_base_metadata(qubo), "num_evaluated": 2 ** len(variables)},
    )


def solve_random(qubo: QUBO, *, num_reads: int = 1000, seed: int | None = None) -> SolveResult:
    num_reads = _positive_int(num_reads, "num_reads")
    variables = _ordered_variables(qubo)
    rng = Random(seed)
    best_energy = float("inf")
    best_sample: dict[str, int] = {}
    for _ in range(num_reads):
        sample = {var: rng.randrange(2) for var in variables}
        energy = qubo.energy(sample)
        if energy < best_energy:
            best_energy = energy
            best_sample = sample
    return SolveResult(
        sample=best_sample,
        energy=best_energy,
        solver="random",
        metadata={**_base_metadata(qubo), "num_reads": num_reads, "seed": seed},
    )


def solve_neal(
    qubo: QUBO,
    *,
    num_reads: int = 200,
    sweeps: int = 2500,
    seed: int | None = None,
) -> SolveResult:
    num_reads = _positive_int(num_reads, "num_reads")
    sweeps = _positive_int(sweeps, "sweeps")
    try:
        from neal import SimulatedAnnealingSampler
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("dwave-neal is required for solver='neal'") from exc
    sampler = SimulatedAnnealingSampler()
    bqm = qubo.to_dimod_bqm()
    sampleset = sampler.sample(bqm, num_reads=num_reads, sweeps=sweeps, seed=seed)
    best = sampleset.first
    return SolveResult(
        sample={str(k): int(v) for k, v in dict(best.sample).items()},
        energy=float(best.energy),
        solver="neal",
        metadata={
            **_base_metadata(qubo),
            "num_reads": num_reads,
            "sweeps": sweeps,
            "seed": seed,
            "neal_variable_order": [str(v) for v in bqm.variables],
        },
    )


def _int_option(options: Mapping[str, object], name: str, default: int) -> int:
    return int(cast(Any, options.get(name, default)))


def _seed_option(options: Mapping[str, object]) -> int | None:
    value = options.get("seed")
    return None if value is None else int(cast(Any, value))


def solve_qubo(qubo: QUBO, solver: str = "exact", **options: object) -> SolveResult:
    solver = solver.lower()
    if solver == "exact":
        return solve_exact(qubo, max_variables=_int_option(options, "max_variables", 24))
    if solver == "random":
        return solve_random(
            qubo,
            num_reads=_int_option(options, "num_reads", 1000),
            seed=_seed_option(options),
        )
    if solver == "neal":
        return solve_neal(
            qubo,
            num_reads=_int_option(options, "num_reads", 200),
            sweeps=_int_option(options, "sweeps", 2500),
            seed=_seed_option(options),
        )
    raise ValueError(f"Unknown solver {solver!r}")


def solved_payload(
    qubo: QUBO, result: SolveResult, decoded: Mapping[str, float] | None = None
) -> dict[str, object]:
    payload: dict[str, object] = {
        "solver": result.solver,
        "energy": result.energy,
        "sample": result.sample,
        "metadata": result.metadata,
    }
    if decoded is not None:
        payload["decoded"] = dict(decoded)
    payload["qubo_energy_check"] = qubo.energy(result.sample)
    return payload
