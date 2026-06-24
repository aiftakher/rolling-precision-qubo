"""QUBO solvers with optional dependencies."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import product
from random import Random
from typing import Any, cast

from .qubo import QUBO


@dataclass
class SolveResult:
    sample: dict[str, int]
    energy: float
    solver: str
    metadata: dict[str, object] = field(default_factory=dict)


def solve_exact(qubo: QUBO, *, max_variables: int = 24) -> SolveResult:
    variables = sorted(qubo.variables)
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
        metadata={"num_evaluated": 2 ** len(variables)},
    )


def solve_random(qubo: QUBO, *, num_reads: int = 1000, seed: int | None = None) -> SolveResult:
    variables = sorted(qubo.variables)
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
        metadata={"num_reads": num_reads, "seed": seed},
    )


def solve_neal(
    qubo: QUBO,
    *,
    num_reads: int = 200,
    sweeps: int = 2500,
    seed: int | None = None,
) -> SolveResult:
    try:
        from neal import SimulatedAnnealingSampler
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("dwave-neal is required for solver='neal'") from exc
    sampler = SimulatedAnnealingSampler()
    sampleset = sampler.sample(qubo.to_dimod_bqm(), num_reads=num_reads, sweeps=sweeps, seed=seed)
    best = sampleset.first
    return SolveResult(
        sample={str(k): int(v) for k, v in dict(best.sample).items()},
        energy=float(best.energy),
        solver="neal",
        metadata={"num_reads": num_reads, "sweeps": sweeps, "seed": seed},
    )


def solve_dwave_placeholder(qubo: QUBO, **_: object) -> SolveResult:
    raise RuntimeError(
        "D-Wave hardware/hybrid backends are optional. Install the dwave extra "
        "and configure credentials through the standard D-Wave environment or config."
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
    if solver in {"dwave-hybrid", "dwave-hardware", "dwave"}:
        return solve_dwave_placeholder(qubo, **options)
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
