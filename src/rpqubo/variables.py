"""Variable and problem specifications for rpqubo."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Union


@dataclass(frozen=True)
class BinaryVar:
    """Native binary variable."""

    name: str

    @property
    def lower(self) -> int:
        return 0

    @property
    def upper(self) -> int:
        return 1


@dataclass(frozen=True)
class ContinuousVar:
    """Bounded continuous variable encoded on a decimal grid."""

    name: str
    lower: float
    upper: float
    digits: int
    encoding: str = "sbe"

    def __post_init__(self) -> None:
        if self.upper < self.lower:
            raise ValueError(f"{self.name}: upper bound must be >= lower bound")
        if self.digits < 1:
            raise ValueError(f"{self.name}: digits must be >= 1")


@dataclass(frozen=True)
class SlackVar(ContinuousVar):
    """Bounded slack variable."""

    encoding: str = "sbe"


@dataclass(frozen=True)
class IntegerVar:
    """Bounded integer variable."""

    name: str
    lower: int
    upper: int
    encoding: str = "binary"
    strict_bounds: bool = True

    def __post_init__(self) -> None:
        if self.upper < self.lower:
            raise ValueError(f"{self.name}: upper bound must be >= lower bound")


Variable = Union[BinaryVar, ContinuousVar, IntegerVar, SlackVar]


@dataclass(frozen=True)
class LinearConstraint:
    """Linear constraint with optional slack encoding configuration."""

    name: str
    linear: Mapping[str, float]
    sense: str
    rhs: float
    penalty: float
    slack_name: str | None = None
    slack_lower: float = 0.0
    slack_upper: float | None = None
    slack_digits: int = 1
    slack_encoding: str = "sbe"
    slack_type: str = "continuous"

    def __post_init__(self) -> None:
        if self.sense not in {"<=", ">=", "=="}:
            raise ValueError(f"{self.name}: unsupported sense {self.sense!r}")
        if self.penalty < 0:
            raise ValueError(f"{self.name}: penalty must be nonnegative")


@dataclass
class QuadraticObjective:
    """Objective in original variables.

    Each quadratic term coefficient multiplies exactly `u * v`.
    """

    constant: float = 0.0
    linear: dict[str, float] = field(default_factory=dict)
    quadratic: dict[tuple[str, str], float] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> QuadraticObjective:
        data = data or {}
        linear = {str(k): float(v) for k, v in data.get("linear", {}).items()}
        quadratic: dict[tuple[str, str], float] = {}
        raw_quad = data.get("quadratic", {})
        if isinstance(raw_quad, Mapping):
            for key, value in raw_quad.items():
                if isinstance(key, str):
                    if "*" in key:
                        u, v = key.split("*", 1)
                    elif "," in key:
                        u, v = key.split(",", 1)
                    else:
                        raise ValueError(f"Bad quadratic key {key!r}")
                    quadratic[(u.strip(), v.strip())] = float(value)
                else:
                    u, v = key
                    quadratic[(str(u), str(v))] = float(value)
        else:
            for row in raw_quad:
                vars_ = row.get("vars") or [row.get("u"), row.get("v")]
                if len(vars_) != 2:
                    raise ValueError(f"Bad quadratic row {row!r}")
                quadratic[(str(vars_[0]), str(vars_[1]))] = float(row["coef"])
        return cls(
            constant=float(data.get("constant", 0.0)),
            linear=linear,
            quadratic=quadratic,
        )


@dataclass
class Problem:
    """Structured quadratic problem used by the package builder."""

    variables: list[Variable]
    objective: QuadraticObjective = field(default_factory=QuadraticObjective)
    constraints: list[LinearConstraint] = field(default_factory=list)
    name: str = "problem"

    @property
    def variable_map(self) -> dict[str, Variable]:
        return {v.name: v for v in self.variables}

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> Problem:
        variables: list[Variable] = []
        for row in data.get("variables", []):
            kind = str(row.get("type", row.get("kind", ""))).lower()
            name = str(row["name"])
            if kind in {"binary", "bool"}:
                variables.append(BinaryVar(name))
            elif kind in {"integer", "int"}:
                variables.append(
                    IntegerVar(
                        name=name,
                        lower=int(row["lower"]),
                        upper=int(row["upper"]),
                        encoding=str(row.get("encoding", "binary")),
                        strict_bounds=bool(row.get("strict_bounds", True)),
                    )
                )
            elif kind in {"continuous", "real"}:
                variables.append(
                    ContinuousVar(
                        name=name,
                        lower=float(row["lower"]),
                        upper=float(row["upper"]),
                        digits=int(row.get("digits", 1)),
                        encoding=str(row.get("encoding", "sbe")),
                    )
                )
            elif kind == "slack":
                variables.append(
                    SlackVar(
                        name=name,
                        lower=float(row.get("lower", 0.0)),
                        upper=float(row["upper"]),
                        digits=int(row.get("digits", 1)),
                        encoding=str(row.get("encoding", "sbe")),
                    )
                )
            else:
                raise ValueError(f"Unsupported variable type for {name!r}: {kind!r}")

        constraints = [
            LinearConstraint(
                name=str(row.get("name", f"c{i}")),
                linear={str(k): float(v) for k, v in row.get("linear", {}).items()},
                sense=str(row["sense"]),
                rhs=float(row["rhs"]),
                penalty=float(row.get("penalty", data.get("penalty", 100.0))),
                slack_name=row.get("slack_name"),
                slack_lower=float(row.get("slack_lower", 0.0)),
                slack_upper=(None if row.get("slack_upper") is None else float(row["slack_upper"])),
                slack_digits=int(row.get("slack_digits", 1)),
                slack_encoding=str(row.get("slack_encoding", "sbe")),
                slack_type=str(row.get("slack_type", "continuous")),
            )
            for i, row in enumerate(data.get("constraints", []), start=1)
        ]
        return cls(
            name=str(data.get("name", "problem")),
            variables=variables,
            objective=QuadraticObjective.from_mapping(data.get("objective", {})),
            constraints=constraints,
        )


def variable_bounds(var: Variable) -> tuple[float, float]:
    """Return numeric lower and upper bounds for a variable spec."""

    if isinstance(var, BinaryVar):
        return (0.0, 1.0)
    return (float(var.lower), float(var.upper))
