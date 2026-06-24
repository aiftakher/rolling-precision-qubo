"""Small expression helpers used by builders and tests."""

from __future__ import annotations

from collections.abc import Mapping


def affine_value(constant: float, coeffs: Mapping[str, float], sample: Mapping[str, int]) -> float:
    return constant + sum(coeff * float(sample.get(bit, 0)) for bit, coeff in coeffs.items())


def square_of_affine_value(
    constant: float, coeffs: Mapping[str, float], sample: Mapping[str, int]
) -> float:
    value = affine_value(constant, coeffs, sample)
    return value * value


def product_of_affines_value(
    c1: float,
    a1: Mapping[str, float],
    c2: float,
    a2: Mapping[str, float],
    sample: Mapping[str, int],
) -> float:
    return affine_value(c1, a1, sample) * affine_value(c2, a2, sample)
