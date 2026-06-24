"""Low-level QUBO algebra."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import fsum, isfinite
from typing import Any

from .encodings import AffineEncoding


@dataclass
class QUBO:
    """Sparse QUBO/BQM representation with separate offset."""

    linear: dict[str, float] = field(default_factory=dict)
    quadratic: dict[tuple[str, str], float] = field(default_factory=dict)
    offset: float = 0.0
    variable_groups: dict[str, list[str]] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)
    variable_order: list[str] = field(default_factory=list)

    def copy(self) -> QUBO:
        """Return an independent copy preserving metadata and ordering."""

        return QUBO(
            linear=dict(self.linear),
            quadratic=dict(self.quadratic),
            offset=self.offset,
            variable_groups={name: list(bits) for name, bits in self.variable_groups.items()},
            metadata=dict(self.metadata),
            variable_order=list(self.variable_order),
        )

    def add_linear(self, var: str, coeff: float) -> None:
        coeff = float(coeff)
        if not isfinite(coeff):
            raise ValueError(f"Non-finite linear coefficient for {var!r}: {coeff}")
        if coeff == 0.0:
            return

        value = fsum((self.linear.get(var, 0.0), coeff))
        if value == 0.0:
            self.linear.pop(var, None)
        else:
            self.linear[var] = value

    def add_quadratic(self, u: str, v: str, coeff: float) -> None:
        coeff = float(coeff)
        if not isfinite(coeff):
            raise ValueError(f"Non-finite quadratic coefficient ({u!r}, {v!r}): {coeff}")
        if coeff == 0.0:
            return
        if u == v:
            self.add_linear(u, coeff)
            return

        key: tuple[str, str] = (u, v) if u <= v else (v, u)
        value = fsum((self.quadratic.get(key, 0.0), coeff))
        if value == 0.0:
            self.quadratic.pop(key, None)
        else:
            self.quadratic[key] = value

    def prune(self, tolerance: float) -> None:
        if tolerance < 0:
            raise ValueError("tolerance must be nonnegative")
        self.linear = {
            key: value for key, value in self.linear.items()
            if abs(value) > tolerance
        }
        self.quadratic = {
            key: value for key, value in self.quadratic.items()
            if abs(value) > tolerance
        }

    def add_offset(self, coeff: float) -> None:
        coeff = float(coeff)
        if not isfinite(coeff):
            raise ValueError(f"Non-finite offset coefficient: {coeff}")
        self.offset = fsum((self.offset, coeff))

    @property
    def variables(self) -> set[str]:
        variables = set(self.variable_order)
        variables.update(self.linear)
        for u, v in self.quadratic:
            variables.update((u, v))
        for group in self.variable_groups.values():
            variables.update(group)
        return variables

    @property
    def ordered_variables(self) -> list[str]:
        """Return explicit order first, then remaining labels deterministically."""

        ordered: list[str] = []
        seen: set[str] = set()
        for var in self.variable_order:
            if var not in seen:
                ordered.append(var)
                seen.add(var)
        for var in sorted(self.variables - seen):
            ordered.append(var)
        return ordered

    def energy(self, sample: Mapping[str, int]) -> float:
        total = self.offset
        for var, coeff in self.linear.items():
            total += coeff * float(sample.get(var, 0))
        for (u, v), coeff in self.quadratic.items():
            total += coeff * float(sample.get(u, 0)) * float(sample.get(v, 0))
        return total

    def to_dimod_bqm(self):
        try:
            import dimod
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("dimod is required for BQM conversion") from exc
        bqm = dimod.BinaryQuadraticModel({}, {}, self.offset, vartype=dimod.BINARY)
        for var in self.ordered_variables:
            bqm.add_variable(var, self.linear.get(var, 0.0))
        for (u, v), bias in self.quadratic.items():
            bqm.add_interaction(u, v, bias)
        return bqm

    def to_dict(self) -> dict[str, object]:
        return {
            "linear": self.linear,
            "quadratic": [
                {"u": u, "v": v, "bias": bias} for (u, v), bias in sorted(self.quadratic.items())
            ],
            "offset": self.offset,
            "variable_groups": self.variable_groups,
            "metadata": self.metadata,
            "variable_order": self.variable_order,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> QUBO:
        quadratic: dict[tuple[str, str], float] = {}
        raw_quad = data.get("quadratic", [])
        if isinstance(raw_quad, Mapping):
            for key, value in raw_quad.items():
                u, v = str(key).split(",", 1)
                quadratic[(u, v)] = float(value)
        else:
            for row in raw_quad:  # type: ignore[assignment]
                quadratic[(str(row["u"]), str(row["v"]))] = float(row["bias"])
        return cls(
            linear={str(k): float(v) for k, v in dict(data.get("linear", {})).items()},
            quadratic=quadratic,
            offset=float(data.get("offset", 0.0)),
            variable_groups={
                str(k): [str(x) for x in v]
                for k, v in dict(data.get("variable_groups", {})).items()
            },
            metadata=dict(data.get("metadata", {})),
            variable_order=[str(x) for x in data.get("variable_order", [])],
        )


def add_affine(qubo: QUBO, encoding: AffineEncoding, weight: float) -> None:
    qubo.add_offset(weight * encoding.offset)
    for bit, coeff in encoding.weights.items():
        qubo.add_linear(bit, weight * coeff)


def add_square_of_affine(
    qubo: QUBO, constant: float, coeffs: Mapping[str, float], weight: float = 1.0
) -> None:
    qubo.add_offset(weight * constant * constant)
    items = list(coeffs.items())
    for bit, coeff in items:
        qubo.add_linear(bit, weight * (coeff * coeff + 2.0 * constant * coeff))
    for i, (bi, ai) in enumerate(items):
        for bj, aj in items[i + 1 :]:
            qubo.add_quadratic(bi, bj, weight * 2.0 * ai * aj)


def add_product_of_affines(
    qubo: QUBO,
    c1: float,
    a1: Mapping[str, float],
    c2: float,
    a2: Mapping[str, float],
    weight: float = 1.0,
) -> None:
    qubo.add_offset(weight * c1 * c2)
    for bit, coeff in a2.items():
        qubo.add_linear(bit, weight * c1 * coeff)
    for bit, coeff in a1.items():
        qubo.add_linear(bit, weight * c2 * coeff)
    for bi, ai in a1.items():
        for bj, aj in a2.items():
            qubo.add_quadratic(bi, bj, weight * ai * aj)


def add_square_of_encoding(qubo: QUBO, encoding: AffineEncoding, weight: float = 1.0) -> None:
    add_square_of_affine(qubo, encoding.offset, encoding.weights, weight)


def add_product_of_encodings(
    qubo: QUBO, left: AffineEncoding, right: AffineEncoding, weight: float = 1.0
) -> None:
    add_product_of_affines(qubo, left.offset, left.weights, right.offset, right.weights, weight)


def coefficient_stats(qubo: QUBO, include_offset: bool = False) -> dict[str, float]:
    vals = list(qubo.linear.values()) + list(qubo.quadratic.values())
    if include_offset:
        vals.append(qubo.offset)
    nonzero = [abs(v) for v in vals if abs(v) > 0.0]
    n = len(qubo.variables)
    possible_quad = n * (n - 1) / 2 if n > 1 else 0
    if not nonzero:
        return {
            "n_variables": float(n),
            "n_linear": float(len(qubo.linear)),
            "n_quadratic": float(len(qubo.quadratic)),
            "min_coeff": 0.0,
            "max_coeff": 0.0,
            "max_abs": 0.0,
            "min_nonzero_abs": 0.0,
            "dynamic_range": 0.0,
            "density": 0.0,
        }
    return {
        "n_variables": float(n),
        "n_linear": float(len(qubo.linear)),
        "n_quadratic": float(len(qubo.quadratic)),
        "min_coeff": min(vals) if vals else 0.0,
        "max_coeff": max(vals) if vals else 0.0,
        "max_abs": max(nonzero),
        "min_nonzero_abs": min(nonzero),
        "dynamic_range": max(nonzero) / min(nonzero),
        "density": len(qubo.quadratic) / possible_quad if possible_quad else 0.0,
    }


def rescaled_qubo(
    qubo: QUBO,
    target_max_abs: float = 10.0,
    *,
    include_offset: bool = False,
) -> tuple[QUBO, float]:
    if not isfinite(target_max_abs) or target_max_abs <= 0.0:
        raise ValueError("target_max_abs must be finite and strictly positive")
    stats = coefficient_stats(qubo, include_offset=include_offset)
    max_abs = stats["max_abs"]
    scaled = qubo.copy()
    if max_abs == 0.0 or max_abs <= target_max_abs:
        scaled.metadata["rescale_factor"] = 1.0
        return scaled, 1.0
    factor = max_abs / target_max_abs
    scaled.linear = {k: v / factor for k, v in scaled.linear.items()}
    scaled.quadratic = {k: v / factor for k, v in scaled.quadratic.items()}
    scaled.offset /= factor
    scaled.metadata["rescale_factor"] = factor
    return scaled, factor
