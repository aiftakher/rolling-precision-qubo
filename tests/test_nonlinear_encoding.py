from __future__ import annotations

import pytest

from rpqubo.builders import build_qubo
from rpqubo.encodings import (
    add_cumulative_unary_order_penalty,
    cumulative_unary_order_pairs,
    encode_cumulative_unary,
    nonlinear_error_table,
)
from rpqubo.io import export_model, load_model
from rpqubo.qubo import QUBO
from rpqubo.variables import ContinuousVar, Problem


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


def test_j2_cumulative_unary_order_pairs_include_every_tail_endpoint() -> None:
    enc = encode_cumulative_unary("x", 0.0, 1.0, 2, "continuous")
    pairs = cumulative_unary_order_pairs(enc)
    assert len(pairs) == 18
    assert ("cu_x_1_9", "cu_x_tail_J2") in pairs
    assert ("cu_x_2_9", "cu_x_tail_J2") in pairs


def test_j2_cumulative_unary_penalty_catches_each_digit_and_tail() -> None:
    enc = encode_cumulative_unary("x", 0.0, 1.0, 2, "continuous")
    qubo = QUBO()
    add_cumulative_unary_order_penalty(qubo, enc, 5.0)

    legal = {bit: 0 for bit in enc.bits}
    for digit in (1, 2):
        for index in range(1, 10):
            legal[f"cu_x_{digit}_{index}"] = 1
    legal["cu_x_tail_J2"] = 1
    assert qubo.energy(legal) == 0.0

    illegal_digit_1 = {bit: 0 for bit in enc.bits}
    illegal_digit_1["cu_x_1_2"] = 1
    assert qubo.energy(illegal_digit_1) > 0.0

    illegal_digit_2 = {bit: 0 for bit in enc.bits}
    illegal_digit_2["cu_x_2_2"] = 1
    assert qubo.energy(illegal_digit_2) > 0.0

    tail_without_digit_1_endpoint = dict(legal)
    tail_without_digit_1_endpoint["cu_x_1_9"] = 0
    assert qubo.energy(tail_without_digit_1_endpoint) > 0.0

    tail_without_digit_2_endpoint = dict(legal)
    tail_without_digit_2_endpoint["cu_x_2_9"] = 0
    assert qubo.energy(tail_without_digit_2_endpoint) > 0.0


def test_builder_requires_and_inserts_cumulative_unary_ordering_penalty() -> None:
    missing = Problem(variables=[ContinuousVar("x", 0.0, 1.0, 2, encoding="cumulative_unary")])
    with pytest.raises(ValueError, match="ordering_penalty"):
        build_qubo(missing)

    problem = Problem(
        variables=[
            ContinuousVar(
                "x",
                0.0,
                1.0,
                2,
                encoding="cumulative_unary",
                ordering_penalty=4.0,
            )
        ]
    )
    result = build_qubo(problem)
    legal = {bit: 0 for bit in result.qubo.variables}
    illegal = dict(legal)
    illegal["cu_x_2_2"] = 1
    assert result.qubo.energy(legal) == 0.0
    assert result.qubo.energy(illegal) > 0.0


def test_model_bundle_preserves_cumulative_unary_ordering_penalty(tmp_path) -> None:
    problem = Problem(
        variables=[
            ContinuousVar(
                "x",
                0.0,
                1.0,
                2,
                encoding="cumulative_unary",
                ordering_penalty=4.0,
            )
        ]
    )
    result = build_qubo(problem)
    path = tmp_path / "model.model"
    export_model(result, path)
    loaded = load_model(path)
    loaded_var = loaded.problem.variables[0]
    assert isinstance(loaded_var, ContinuousVar)
    assert loaded_var.ordering_penalty == 4.0


def test_one_digit_power_surrogate_is_exact_on_grid() -> None:
    rows = nonlinear_error_table([0.6, 1.5], [1])
    assert all(row["max_abs_error"] <= 1e-12 for row in rows)


def test_multi_digit_power_surrogate_is_approximate() -> None:
    rows = nonlinear_error_table([0.6, 1.5], [2, 3])
    assert all(row["max_abs_error"] > 0.0 for row in rows)
