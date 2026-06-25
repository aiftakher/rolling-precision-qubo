from __future__ import annotations

from itertools import product

from rpqubo.quadratization import brute_force_minimum, rosenberg_reduce_to_qubo


def test_rosenberg_reduction_preserves_minimum_with_adaptive_strength() -> None:
    terms = {("a", "b", "c"): 10000.0, ("a",): -4000.0, ("b",): -4000.0, ("c",): -4000.0}
    original, _ = brute_force_minimum(terms)
    reduced = rosenberg_reduce_to_qubo(terms)
    best = float("inf")
    for values in product([0, 1], repeat=len(reduced.qubo.variables)):
        sample = dict(zip(sorted(reduced.qubo.variables), values))
        best = min(best, reduced.qubo.energy(sample))
    assert abs(best - original) <= 1e-8


def test_rosenberg_ancilla_satisfied_at_optimum_when_strength_large() -> None:
    terms = {("a", "b", "c"): -3.0, ("a",): 0.5}
    reduced = rosenberg_reduce_to_qubo(terms, strength=100.0)
    best = float("inf")
    best_sample = {}
    variables = sorted(reduced.qubo.variables)
    for values in product([0, 1], repeat=len(variables)):
        sample = dict(zip(variables, values))
        energy = reduced.qubo.energy(sample)
        if energy < best:
            best = energy
            best_sample = sample
    for anc in reduced.ancillas:
        parts = anc.split("__")
        a = parts[0].replace("anc_", "")
        b = parts[1]
        assert best_sample[anc] == best_sample[a] * best_sample[b]
