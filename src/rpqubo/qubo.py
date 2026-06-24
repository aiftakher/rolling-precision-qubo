"""Low-level QUBO algebra."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
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

    def add_linear(self, var: str, coeff: float) -> None:
        if abs(coeff) <= 0.0:
            return
        self.linear[var] = self.linear.get(var, 0.0) + coeff
        if abs(self.linear[var]) < 1e-15:
            del self.linear[var]

    def add_quadratic(self, u: str, v: str, coeff: float) -> None:
        if abs(coeff) <= 0.0:
            return
        if u == v:
            self.add_linear(u, coeff)
            return
        a, b = (u, v) if u < v else (v, u)
        key = (a, b)
        self.quadratic[key] = self.quadratic.get(key, 0.0) + coeff
        if abs(self.quadratic[key]) < 1e-15:
            del self.quadratic[key]

    def add_offset(self, coeff: float) -> None:
        self.offset += coeff

    @property
    def variables(self) -> set[str]:
        vars_ = set(self.linear)
        for u, v in self.quadratic:
            vars_.add(u)
            vars_.add(v)
        return vars_

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
        return dimod.BinaryQuadraticModel(
            self.linear, self.quadratic, self.offset, vartype=dimod.BINARY
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "linear": self.linear,
            "quadratic": [
                {"u": u, "v": v, "bias": bias} for (u, v), bias in sorted(self.quadratic.items())
            ],
            "offset": self.offset,
            "variable_groups": self.variable_groups,
            "metadata": self.metadata,
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


def rescaled_qubo(qubo: QUBO, target_max_abs: float = 10.0) -> tuple[QUBO, float]:
    stats = coefficient_stats(qubo, include_offset=False)
    max_abs = stats["max_abs"]
    if max_abs == 0.0 or max_abs <= target_max_abs:
        return qubo, 1.0
    factor = max_abs / target_max_abs
    scaled = QUBO(
        linear={k: v / factor for k, v in qubo.linear.items()},
        quadratic={k: v / factor for k, v in qubo.quadratic.items()},
        offset=qubo.offset / factor,
        variable_groups={k: list(v) for k, v in qubo.variable_groups.items()},
        metadata={**qubo.metadata, "rescale_factor": factor},
    )
    return scaled, factor
