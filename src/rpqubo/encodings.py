"""Binary encodings for bounded variables and nonlinear digit surrogates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import product
from math import isfinite, sqrt
from typing import Any

from .variables import BinaryVar, IntegerVar, SlackVar, Variable

DECIMAL_SBE_WEIGHTS = (1, 2, 3, 3)


@dataclass
class AffineEncoding:
    """Affine map from binary variables to one original variable."""

    name: str
    offset: float
    weights: dict[str, float]
    lower: float
    upper: float
    kind: str
    encoding: str
    bits: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.bits:
            self.bits = list(self.weights)

    def decode(self, sample: Mapping[str, int], *, validate: bool = True) -> float:
        value = self.offset
        for bit, coeff in self.weights.items():
            value += coeff * float(sample.get(bit, 0))
        if validate and self.kind == "integer":
            rounded = round(value)
            if abs(value - rounded) > 1e-9:
                raise ValueError(f"{self.name}: decoded non-integer value {value}")
            if rounded < self.lower or rounded > self.upper:
                raise ValueError(
                    f"{self.name}: decoded value {rounded} outside [{self.lower}, {self.upper}]"
                )
            return float(rounded)
        if validate and (value < self.lower - 1e-9 or value > self.upper + 1e-9):
            raise ValueError(
                f"{self.name}: decoded value {value} outside [{self.lower}, {self.upper}]"
            )
        return value

    def grid_summary(self) -> dict[str, object]:
        digits = self.metadata.get("digits")
        if self.kind in {"continuous", "slack"} and isinstance(digits, int):
            step = (self.upper - self.lower) / (10**digits)
            return {
                "grid_size": 10**digits + 1,
                "step": step,
                "min": self.lower,
                "max": self.upper,
            }
        if self.kind == "integer":
            return {
                "grid_size": int(self.upper - self.lower + 1),
                "step": 1,
                "min": int(self.lower),
                "max": int(self.upper),
                "invalid_values": self.metadata.get("invalid_values", []),
            }
        return {"grid_size": 2, "step": 1, "min": 0, "max": 1}

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "encoding": self.encoding,
            "offset": self.offset,
            "weights": self.weights,
            "binary_variables": self.bits,
            "bounds": {"lower": self.lower, "upper": self.upper},
            "grid": self.grid_summary(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AffineEncoding:
        bounds = data.get("bounds", {})
        if not isinstance(bounds, Mapping):
            raise ValueError("encoding bounds must be a mapping")
        bits = data.get("binary_variables", data.get("bits", []))
        return cls(
            name=str(data["name"]),
            offset=float(data["offset"]),
            weights={str(k): float(v) for k, v in dict(data["weights"]).items()},
            lower=float(bounds.get("lower", data.get("lower", 0.0))),
            upper=float(bounds.get("upper", data.get("upper", 1.0))),
            kind=str(data["kind"]),
            encoding=str(data["encoding"]),
            bits=[str(bit) for bit in bits],
            metadata=dict(data.get("metadata", {})),
        )


def sbe_bit_name(name: str, digit: int, index: int) -> str:
    return f"z_{name}_{digit}_{index}"


def sbe_tail_name(name: str, digits: int) -> str:
    return f"z_{name}_tail_J{digits}"


def encode_sbe(name: str, lower: float, upper: float, digits: int, kind: str) -> AffineEncoding:
    if digits < 1:
        raise ValueError("SBE requires digits >= 1")
    if upper < lower:
        raise ValueError("upper bound must be >= lower bound")
    width = upper - lower
    weights: dict[str, float] = {}
    for j in range(1, digits + 1):
        place = 10.0 ** (-j)
        for k, w in enumerate(DECIMAL_SBE_WEIGHTS, start=1):
            weights[sbe_bit_name(name, j, k)] = width * place * float(w)
    weights[sbe_tail_name(name, digits)] = width * (10.0 ** (-digits))
    return AffineEncoding(
        name=name,
        offset=lower,
        weights=weights,
        lower=lower,
        upper=upper,
        kind=kind,
        encoding="sbe",
        metadata={"digits": digits, "digit_weights": DECIMAL_SBE_WEIGHTS},
    )


def unary_bit_name(name: str, digit: int, index: int) -> str:
    return f"u_{name}_{digit}_{index}"


def cumulative_unary_bit_name(name: str, digit: int, index: int) -> str:
    return f"cu_{name}_{digit}_{index}"


def encode_digit_sum_unary(
    name: str, lower: float, upper: float, digits: int, kind: str
) -> AffineEncoding:
    if digits < 1:
        raise ValueError("unary encoding requires digits >= 1")
    width = upper - lower
    weights: dict[str, float] = {}
    for j in range(1, digits + 1):
        place = 10.0 ** (-j)
        for i in range(1, 10):
            weights[unary_bit_name(name, j, i)] = width * place
    weights[f"u_{name}_tail_J{digits}"] = width * (10.0 ** (-digits))
    return AffineEncoding(
        name=name,
        offset=lower,
        weights=weights,
        lower=lower,
        upper=upper,
        kind=kind,
        encoding="unary",
        metadata={"digits": digits, "endpoint_bit": True},
    )


def encode_cumulative_unary(
    name: str, lower: float, upper: float, digits: int, kind: str
) -> AffineEncoding:
    if digits < 1:
        raise ValueError("cumulative unary encoding requires digits >= 1")
    width = upper - lower
    weights: dict[str, float] = {}
    for j in range(1, digits + 1):
        place = 10.0 ** (-j)
        for i in range(1, 10):
            weights[cumulative_unary_bit_name(name, j, i)] = width * place
    weights[f"cu_{name}_tail_J{digits}"] = width * (10.0 ** (-digits))
    return AffineEncoding(
        name=name,
        offset=lower,
        weights=weights,
        lower=lower,
        upper=upper,
        kind=kind,
        encoding="cumulative_unary",
        metadata={"digits": digits, "endpoint_bit": True},
    )


def bounded_binary_weights(range_size: int) -> list[int]:
    """Return binary-like weights whose subset sums never exceed range_size."""

    if range_size < 0:
        raise ValueError("range_size must be nonnegative")
    if range_size == 0:
        return []
    weights: list[int] = []
    total = 0
    power = 1
    while total + power < range_size:
        weights.append(power)
        total += power
        power *= 2
    if total < range_size:
        weights.append(range_size - total)
    return weights


def standard_binary_weights(range_size: int) -> list[int]:
    if range_size < 0:
        raise ValueError("range_size must be nonnegative")
    bits = 0
    while 2**bits - 1 < range_size:
        bits += 1
    return [2**i for i in range(bits)]


def encode_integer(var: IntegerVar) -> AffineEncoding:
    if var.encoding != "binary":
        raise ValueError(f"Unsupported integer encoding {var.encoding!r}")
    range_size = var.upper - var.lower
    if var.strict_bounds:
        raw_weights = bounded_binary_weights(range_size)
        invalid_values: list[int] = []
        encoding = "bounded_binary"
    else:
        raw_weights = standard_binary_weights(range_size)
        max_value = var.lower + sum(raw_weights)
        invalid_values = list(range(var.upper + 1, max_value + 1))
        encoding = "binary"
    weights = {f"z_{var.name}_b{i}": float(w) for i, w in enumerate(raw_weights)}
    return AffineEncoding(
        name=var.name,
        offset=float(var.lower),
        weights=weights,
        lower=float(var.lower),
        upper=float(var.upper),
        kind="integer",
        encoding=encoding,
        metadata={
            "strict_bounds": var.strict_bounds,
            "integer_weights": raw_weights,
            "invalid_values": invalid_values,
        },
    )


def encode_variable(var: Variable) -> AffineEncoding:
    if isinstance(var, BinaryVar):
        return AffineEncoding(
            name=var.name,
            offset=0.0,
            weights={var.name: 1.0},
            lower=0.0,
            upper=1.0,
            kind="binary",
            encoding="native",
        )
    if isinstance(var, IntegerVar):
        return encode_integer(var)
    kind = "slack" if isinstance(var, SlackVar) else "continuous"
    if var.encoding == "sbe":
        return encode_sbe(var.name, var.lower, var.upper, var.digits, kind)
    if var.encoding in {"unary", "digit_sum_unary"}:
        return encode_digit_sum_unary(var.name, var.lower, var.upper, var.digits, kind)
    if var.encoding == "cumulative_unary":
        return encode_cumulative_unary(var.name, var.lower, var.upper, var.digits, kind)
    raise ValueError(f"Unsupported encoding for {var.name}: {var.encoding!r}")


def sbe_grid_values(digits: int) -> list[float]:
    return [k / (10**digits) for k in range(10**digits + 1)]


def brute_force_decoded_values(encoding: AffineEncoding, max_bits: int = 20) -> set[float]:
    if len(encoding.bits) > max_bits:
        raise ValueError("Too many bits for brute-force decoding")
    values: set[float] = set()
    for bits in product([0, 1], repeat=len(encoding.bits)):
        values.add(round(encoding.decode(dict(zip(encoding.bits, bits))), 12))
    return values


def nonlinear_coefficients(digits: int, exponent: float) -> tuple[list[dict[str, float]], float]:
    """Compute cumulative-unary coefficients for x^a on the decimal grid."""

    coeffs: list[dict[str, float]] = []
    total = 0.0
    for j in range(1, digits + 1):
        for i in range(1, 10):
            p = (i * 10.0 ** (-j)) ** exponent - ((i - 1) * 10.0 ** (-j)) ** exponent
            coeffs.append({"digit": float(j), "index": float(i), "coefficient": p})
            total += p
    return coeffs, 1.0 - total


def nonlinear_surrogate_value(k: int, digits: int, exponent: float) -> float:
    """Evaluate the paper's multi-digit nonlinear surrogate at k / 10^digits."""

    if k < 0 or k > 10**digits:
        raise ValueError("k is outside the encoded grid")
    if k == 10**digits:
        digit_values = [9] * digits
        endpoint = 1
    else:
        digit_values = [int(ch) for ch in f"{k:0{digits}d}"]
        endpoint = 0
    coeffs, q = nonlinear_coefficients(digits, exponent)
    value = q * endpoint
    for row in coeffs:
        j = int(row["digit"])
        i = int(row["index"])
        if i <= digit_values[j - 1]:
            value += row["coefficient"]
    return value


def nonlinear_error_table(exponents: list[float], digits_list: list[int]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for exponent in exponents:
        for digits in digits_list:
            errors: list[float] = []
            max_abs = -1.0
            max_k = 0
            max_hat = 0.0
            max_true = 0.0
            for k in range(10**digits + 1):
                x = k / (10**digits)
                hat = nonlinear_surrogate_value(k, digits, exponent)
                true = x**exponent
                err = hat - true
                errors.append(err)
                if abs(err) > max_abs:
                    max_abs = abs(err)
                    max_k = k
                    max_hat = hat
                    max_true = true
            rows.append(
                {
                    "a": exponent,
                    "J": float(digits),
                    "grid_size": float(10**digits + 1),
                    "max_abs_error": max_abs,
                    "rmse": sqrt(sum(e * e for e in errors) / len(errors)),
                    "max_error_point": max_k / (10**digits),
                    "surrogate_at_max": max_hat,
                    "true_at_max": max_true,
                }
            )
    return rows


def cumulative_unary_order_pairs(encoding: AffineEncoding) -> list[tuple[str, str]]:
    """Return adjacent cumulative-unary pairs that must satisfy next <= current."""

    if encoding.encoding != "cumulative_unary":
        raise ValueError("ordering penalty requires a cumulative_unary encoding")
    digits_raw = encoding.metadata.get("digits")
    if not isinstance(digits_raw, int):
        raise ValueError("cumulative_unary encoding metadata must include integer digits")
    pairs: list[tuple[str, str]] = []
    for digit in range(1, digits_raw + 1):
        bits = [cumulative_unary_bit_name(encoding.name, digit, index) for index in range(1, 10)]
        pairs.extend(zip(bits, bits[1:]))
    tail = f"cu_{encoding.name}_tail_J{digits_raw}"
    if tail in encoding.weights:
        pairs.append((cumulative_unary_bit_name(encoding.name, digits_raw, 9), tail))
    return pairs


def add_cumulative_unary_order_penalty(
    qubo: Any,
    encoding: AffineEncoding,
    lambda_order: float,
) -> None:
    """Add lambda * sum(z_next - z_current*z_next) for illegal unary transitions."""

    if not isfinite(lambda_order) or lambda_order <= 0.0:
        raise ValueError("lambda_order must be finite and positive")
    for current, next_bit in cumulative_unary_order_pairs(encoding):
        qubo.add_linear(next_bit, lambda_order)
        qubo.add_quadratic(current, next_bit, -lambda_order)


def add_nonlinear_surrogate(
    qubo: Any,
    encoding: AffineEncoding,
    exponent: float,
    *,
    lambda_order: float,
) -> None:
    """Add the paper's cumulative-unary power surrogate and ordering penalty."""

    if encoding.encoding != "cumulative_unary":
        raise ValueError("nonlinear surrogate requires cumulative_unary encoding")
    if encoding.lower != 0.0 or encoding.upper != 1.0:
        raise ValueError("nonlinear surrogate currently supports encodings on [0, 1]")
    if not isfinite(exponent):
        raise ValueError("exponent must be finite")

    add_cumulative_unary_order_penalty(qubo, encoding, lambda_order)
    digits_raw = encoding.metadata.get("digits")
    if not isinstance(digits_raw, int):
        raise ValueError("encoding metadata must include integer digits")
    coeffs, endpoint_coeff = nonlinear_coefficients(digits_raw, exponent)
    for row in coeffs:
        bit = cumulative_unary_bit_name(encoding.name, int(row["digit"]), int(row["index"]))
        qubo.add_linear(bit, row["coefficient"])
    qubo.add_linear(f"cu_{encoding.name}_tail_J{digits_raw}", endpoint_coeff)
    qubo.metadata.setdefault("nonlinear_surrogates", [])
    surrogates = qubo.metadata["nonlinear_surrogates"]
    if isinstance(surrogates, list):
        surrogates.append(
            {
                "variable": encoding.name,
                "exponent": exponent,
                "digits": digits_raw,
                "exact": digits_raw == 1,
                "lambda_order": lambda_order,
            }
        )
