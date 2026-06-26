from __future__ import annotations

from itertools import product

import pytest

from rpqubo.encodings import encode_integer
from rpqubo.variables import IntegerVar


def decoded_values(var: IntegerVar) -> set[int]:
    enc = encode_integer(var)
    values = set()
    for bits in product([0, 1], repeat=len(enc.bits)):
        values.add(int(enc.decode(dict(zip(enc.bits, bits)))))
    return values


def test_power_of_two_range_exact_coverage() -> None:
    assert decoded_values(IntegerVar("i", 3, 10)) == set(range(3, 11))


def test_non_power_of_two_strict_bounds_do_not_exceed_domain() -> None:
    assert decoded_values(IntegerVar("i", 0, 5, strict_bounds=True)) == set(range(0, 6))


def test_non_strict_binary_reports_and_rejects_out_of_range_decode() -> None:
    enc = encode_integer(IntegerVar("i", 0, 5, strict_bounds=False))
    assert enc.metadata["invalid_values"] == [6, 7]
    sample = {bit: 1 for bit in enc.bits}
    with pytest.raises(ValueError):
        enc.decode(sample)
