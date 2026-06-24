from __future__ import annotations

import pytest

from rpqubo.examples import (
    reproduce_alan_bit_growth,
    reproduce_alan_zoom,
    reproduce_example1_bit_growth,
    reproduce_example1_zoom,
    reproduce_example2_bit_growth,
    reproduce_example2_zoom,
)


def _x(row: dict[str, object]) -> list[float]:
    values = row["x"]
    assert isinstance(values, dict)
    return [float(values[f"x{i}"]) for i in range(1, 5)]


@pytest.mark.slow
def test_tables_1_to_6_reference_rows() -> None:
    assert [
        (r["J1"], round(r["x1"], 7), round(r["x2"], 7), r["n_vars"], r["n_quad"])
        for r in reproduce_example1_bit_growth()
    ] == [
        (2, 0.12, 0.77, 18, 72),
        (4, 0.1235, 0.7654, 34, 272),
        (6, 0.123457, 0.765432, 50, 600),
        (8, 0.1234567, 0.7654321, 66, 1056),
    ]

    assert [(round(r["x1"], 6), round(r["x2"], 6)) for r in reproduce_example1_zoom()] == [
        (0.1, 0.8),
        (0.12, 0.76),
        (0.124, 0.764),
        (0.1232, 0.7656),
        (0.12352, 0.76544),
    ]

    table3_rows = reproduce_example2_bit_growth()
    table3 = [(r["Jx"], r["Js"], round(r["x"], 6), r["y"], round(r["s"], 6)) for r in table3_rows]
    assert table3[0] in {(1, 1, 0.4, 0, 0.6), (1, 1, 0.3, 0, 0.7)}
    assert table3[1:] == [
        (2, 2, 0.35, 0, 0.65),
        (4, 4, 0.35, 0, 0.65),
    ]
    first = table3_rows[0]
    assert abs(float(first["objective"]) - 0.0025) <= 1e-9
    assert abs(float(first["feasibility"])) <= 1e-9
    assert first["n_vars"] == 11
    assert first["n_quad"] == 55

    table4 = reproduce_example2_zoom()
    assert [r["action"] for r in table4] == [
        "baseline",
        "accepted_zoom",
        "accepted_zoom",
        "backtrack",
        "accepted_zoom",
    ]
    assert [round(r["x"], 6) for r in table4] == [0.4, 0.36, 0.352, 0.352, 0.3496]

    table5 = reproduce_alan_bit_growth()
    expected_x = [
        [0.3, 0.0, 0.4, 0.4],
        [0.34, 0.33, 0.36, 0.0],
        [0.297, 0.264, 0.435, 0.004],
        [0.498, 0.087, 0.434, 0.0031],
        [0.315, 0.0033, 0.5333, 0.15],
        [0.3967, 0.1813, 0.4269, 0.01],
    ]
    for row, expected in zip(table5, expected_x):
        assert [round(v, 6) for v in _x(row)] == [round(v, 6) for v in expected]

    table6 = reproduce_alan_zoom()
    assert [r["action"] for r in table6] == ["baseline", "accepted_zoom", "backtrack"]
    assert [round(v, 6) for v in _x(table6[0])] == [0.5, 0.0, 0.5, 0.0]
    assert [round(v, 6) for v in _x(table6[1])] == [0.4, 0.0, 0.52, 0.08]
    assert abs(float(table6[1]["objective"]) - 2.928) <= 1e-12
