from __future__ import annotations

from rpqubo.examples import (
    reproduce_alan_bit_growth,
    reproduce_alan_zoom,
    reproduce_example1_bit_growth,
    reproduce_example1_zoom,
    reproduce_example2_bit_growth,
    reproduce_example2_zoom,
)


def test_paper_structural_sizes() -> None:
    assert [(r["n_vars"], r["n_quad"]) for r in reproduce_example1_bit_growth()] == [
        (18, 72),
        (34, 272),
        (50, 600),
        (66, 1056),
    ]
    assert {(r["n_vars"], r["n_quad"]) for r in reproduce_example1_zoom()} == {(10, 20)}
    assert [(r["n_vars"], r["n_quad"]) for r in reproduce_example2_bit_growth()] == [
        (11, 55),
        (19, 171),
        (35, 595),
    ]
    assert {(r["n_vars"], r["n_quad"]) for r in reproduce_example2_zoom()} == {(11, 55)}
    assert [(r["n_vars"], r["n_quad"]) for r in reproduce_alan_bit_growth()] == [
        (49, 406),
        (85, 1248),
        (121, 2554),
        (157, 4324),
        (173, 5820),
        (301, 16044),
    ]
    assert {(r["n_vars"], r["n_quad"]) for r in reproduce_alan_zoom()} == {(46, 385)}
