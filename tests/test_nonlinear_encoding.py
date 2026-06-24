from __future__ import annotations

from rpqubo.encodings import (
    add_cumulative_unary_order_penalty,
    encode_cumulative_unary,
    nonlinear_error_table,
)
from rpqubo.qubo import QUBO


def test_cumulative_unary_order_penalty() -> None:
    enc = encode_cumulative_unary("x", 0.0, 1.0, 1, "continuous")
    qubo = QUBO()
    add_cumulative_unary_order_penalty(qubo, enc, 7.0)

    legal = {bit: 0 for bit in enc.bits}
    for bit in enc.bits[:4]:
        legal[bit] = 1
    assert qubo.energy(legal) == 0.0

    illegal = {bit: 0 for bit in enc.bits}
    illegal["cu_x_1_2"] = 1
    assert qubo.energy(illegal) > 0.0


def test_one_digit_power_surrogate_is_exact_on_grid() -> None:
    rows = nonlinear_error_table([0.6, 1.5], [1])
    assert all(row["max_abs_error"] <= 1e-12 for row in rows)
