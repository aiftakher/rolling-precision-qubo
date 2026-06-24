from __future__ import annotations

from itertools import product

from rpqubo.io import export_qubo, load_qubo
from rpqubo.qubo import QUBO


def _fixture_qubo() -> QUBO:
    return QUBO(
        linear={"a": -1.25, "b": 0.5, "c": 2.0},
        quadratic={("a", "b"): 2.5, ("b", "c"): -0.75},
        offset=1.125,
        variable_order=["a", "b", "c"],
    )


def _assert_roundtrip(original: QUBO, reloaded: QUBO) -> None:
    assert reloaded.linear == original.linear
    assert reloaded.quadratic == original.quadratic
    assert reloaded.offset == original.offset
    for bits in product([0, 1], repeat=3):
        sample = dict(zip(("a", "b", "c"), bits))
        assert abs(reloaded.energy(sample) - original.energy(sample)) <= 1e-12


def test_qubo_roundtrips_preserve_coefficients_and_energies(tmp_path) -> None:
    original = _fixture_qubo()
    cases = [
        ("json", "json", None),
        ("csv", "csv", None),
        ("coo", "coo", None),
        ("npz", "npz", None),
        ("dimod", "dimod", None),
        ("bqm", "bqm", None),
    ]
    for label, suffix, fmt in cases:
        path = tmp_path / f"qubo.{suffix}"
        export_qubo(original, path, fmt=fmt)
        reloaded = load_qubo(path)
        _assert_roundtrip(original, reloaded)
        if label in {"dimod", "bqm"}:
            assert reloaded.quadratic[("a", "b")] == 2.5


def test_direct_dimod_roundtrip_keeps_ab_interaction(tmp_path) -> None:
    original = _fixture_qubo()
    path = tmp_path / "direct.bqm"
    export_qubo(original, path)
    reloaded = load_qubo(path)
    _assert_roundtrip(original, reloaded)
    assert ("a", "b") in reloaded.quadratic
    assert reloaded.quadratic[("a", "b")] == 2.5
