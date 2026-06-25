from __future__ import annotations

from rpqubo.encodings import (
    brute_force_decoded_values,
    encode_digit_sum_unary,
    encode_sbe,
    nonlinear_coefficients,
    nonlinear_error_table,
    sbe_grid_values,
)


def test_sbe_endpoints_and_bounds() -> None:
    enc = encode_sbe("x", 0.0, 1.0, 2, "continuous")
    values = brute_force_decoded_values(enc)
    assert 0.0 in values
    assert 1.0 in values
    assert min(values) >= 0.0
    assert max(values) <= 1.0


def test_sbe_all_decimal_grid_values_are_representable() -> None:
    for digits in [1, 2, 3]:
        enc = encode_sbe("x", 0.0, 1.0, digits, "continuous")
        values = brute_force_decoded_values(enc)
        expected = {round(v, 12) for v in sbe_grid_values(digits)}
        assert expected <= values


def test_sbe_scales_to_bounds() -> None:
    enc = encode_sbe("x", 2.0, 5.0, 1, "continuous")
    sample = {bit: 0 for bit in enc.bits}
    assert enc.decode(sample) == 2.0
    sample = {bit: 1 for bit in enc.bits}
    assert abs(enc.decode(sample) - 5.0) <= 1e-12


def test_digit_sum_unary_counts_and_endpoint() -> None:
    enc = encode_digit_sum_unary("x", 0.0, 1.0, 3, "continuous")
    sample = {bit: 0 for bit in enc.bits}
    for digit, count in [(1, 3), (2, 4), (3, 2)]:
        for i in range(1, count + 1):
            sample[f"u_x_{digit}_{i}"] = 1
    assert abs(enc.decode(sample) - 0.342) <= 1e-12
    sample = {bit: 1 for bit in enc.bits}
    assert abs(enc.decode(sample) - 1.0) <= 1e-12


def test_one_digit_nonlinear_coefficients_have_endpoint_q() -> None:
    coeffs, q = nonlinear_coefficients(1, 1.5)
    assert len(coeffs) == 9
    assert abs(sum(row["coefficient"] for row in coeffs) + q - 1.0) <= 1e-12


def test_nonlinear_error_table_schema() -> None:
    rows = nonlinear_error_table([0.6, 1.5], [1, 2])
    assert {"a", "J", "grid_size", "max_abs_error", "rmse"} <= rows[0].keys()
    assert rows[0]["max_abs_error"] < 1e-12
