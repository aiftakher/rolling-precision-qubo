"""Rosenberg quadratization utilities."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product

from .qubo import QUBO

Polynomial = Mapping[tuple[str, ...], float]


@dataclass
class QuadratizationResult:
    qubo: QUBO
    ancillas: set[str]
    strength: float


def evaluate_polynomial(terms: Polynomial, sample: Mapping[str, int]) -> float:
    total = 0.0
    for monomial, coeff in terms.items():
        value = 1
        for var in monomial:
            value *= int(sample.get(var, 0))
        total += coeff * value
    return total


def default_rosenberg_strength(terms: Polynomial, margin: float = 1.0) -> float:
    return sum(abs(c) for c in terms.values()) + abs(margin)


def rosenberg_reduce_to_qubo(
    terms: Polynomial, strength: float | None = None, *, reuse_pairs: bool = True
) -> QuadratizationResult:
    """Reduce pseudo-Boolean terms to QUBO with Rosenberg ancillas."""

    penalty = default_rosenberg_strength(terms) if strength is None else float(strength)
    if penalty <= 0:
        raise ValueError("Rosenberg strength must be positive")
    qubo = QUBO()
    ancillas: set[str] = set()
    pair_cache: dict[tuple[str, str], str] = {}
    counter = 0

    def new_ancilla(a: str, b: str) -> str:
        nonlocal counter
        first, second = sorted((a, b))
        key = (first, second)
        if reuse_pairs and key in pair_cache:
            return pair_cache[key]
        counter += 1
        name = f"anc_{key[0]}__{key[1]}__{counter}"
        pair_cache[key] = name
        ancillas.add(name)
        qubo.add_quadratic(key[0], key[1], penalty)
        qubo.add_quadratic(key[0], name, -2.0 * penalty)
        qubo.add_quadratic(key[1], name, -2.0 * penalty)
        qubo.add_linear(name, 3.0 * penalty)
        return name

    work = [(tuple(sorted(set(mon))), float(coeff)) for mon, coeff in terms.items()]
    while work:
        monomial, coeff = work.pop()
        degree = len(monomial)
        if degree == 0:
            qubo.add_offset(coeff)
        elif degree == 1:
            qubo.add_linear(monomial[0], coeff)
        elif degree == 2:
            qubo.add_quadratic(monomial[0], monomial[1], coeff)
        else:
            a, b, *rest = monomial
            y = new_ancilla(a, b)
            work.append((tuple(sorted((y, *rest))), coeff))
    return QuadratizationResult(qubo=qubo, ancillas=ancillas, strength=penalty)


def brute_force_minimum(terms: Polynomial) -> tuple[float, dict[str, int]]:
    variables = sorted({var for mon in terms for var in mon})
    best_energy = float("inf")
    best_sample: dict[str, int] = {}
    for values in product([0, 1], repeat=len(variables)):
        sample = dict(zip(variables, values))
        energy = evaluate_polynomial(terms, sample)
        if energy < best_energy:
            best_energy = energy
            best_sample = sample
    return best_energy, best_sample


def validate_quadratization(terms: Polynomial, strength: float | None = None) -> bool:
    original_energy, _ = brute_force_minimum(terms)
    result = rosenberg_reduce_to_qubo(terms, strength=strength)
    variables = sorted(result.qubo.variables)
    best = float("inf")
    for values in product([0, 1], repeat=len(variables)):
        sample = dict(zip(variables, values))
        best = min(best, result.qubo.energy(sample))
    return abs(best - original_energy) <= 1e-8
