"""Variable and problem specifications for rpqubo."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Union


@dataclass(frozen=True)
class BinaryVar:
    """Native binary variable."""

    name: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("binary variable name must be nonempty")

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
    ordering_penalty: float | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("continuous variable name must be nonempty")
        if not isfinite(self.lower) or not isfinite(self.upper):
            raise ValueError(f"{self.name}: bounds must be finite")
        if self.upper < self.lower:
            raise ValueError(f"{self.name}: upper bound must be >= lower bound")
        if self.digits < 1:
            raise ValueError(f"{self.name}: digits must be >= 1")
        if self.encoding == "cumulative_unary":
            if self.ordering_penalty is None:
                return
            if not isfinite(self.ordering_penalty) or self.ordering_penalty <= 0.0:
                raise ValueError(f"{self.name}: ordering_penalty must be finite and positive")
        elif self.ordering_penalty is not None:
            raise ValueError(
                f"{self.name}: ordering_penalty is only supported for cumulative_unary"
            )


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
        if not self.name:
            raise ValueError("integer variable name must be nonempty")
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
    slack_ordering_penalty: float | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("constraint name must be nonempty")
        if self.sense not in {"<=", ">=", "=="}:
            raise ValueError(f"{self.name}: unsupported sense {self.sense!r}")
        if not isfinite(self.rhs):
            raise ValueError(f"{self.name}: rhs must be finite")
        if not isfinite(self.penalty) or self.penalty <= 0.0:
            raise ValueError(f"{self.name}: penalty must be finite and positive")
        if self.sense != "==" and self.slack_encoding == "cumulative_unary":
            if self.slack_ordering_penalty is None:
                return
            if not isfinite(self.slack_ordering_penalty) or self.slack_ordering_penalty <= 0.0:
                raise ValueError(f"{self.name}: slack_ordering_penalty must be finite and positive")
        elif self.sense != "==" and self.slack_ordering_penalty is not None:
            raise ValueError(
                f"{self.name}: slack_ordering_penalty is only supported for cumulative_unary"
            )


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

    def validate(self) -> None:
        """Validate names, references, encodings, bounds, and generated bits."""

        if not self.name:
            raise ValueError("problem name must be nonempty")
        if not self.variables:
            raise ValueError("problem must contain at least one variable")

        names: list[str] = []
        for var in self.variables:
            if not var.name:
                raise ValueError("variable names must be nonempty")
            names.append(var.name)
            if isinstance(var, BinaryVar):
                continue
            if not isfinite(float(var.lower)) or not isfinite(float(var.upper)):
                raise ValueError(f"{var.name}: bounds must be finite")
            if var.upper < var.lower:
                raise ValueError(f"{var.name}: upper bound must be >= lower bound")
            if isinstance(var, IntegerVar):
                if var.encoding != "binary":
                    raise ValueError(f"{var.name}: unsupported integer encoding {var.encoding!r}")
                if not var.strict_bounds:
                    raise ValueError(
                        f"{var.name}: strict_bounds=False is disabled until invalid-state "
                        "penalties are implemented"
                    )
            elif var.encoding not in {"sbe", "unary", "digit_sum_unary", "cumulative_unary"}:
                raise ValueError(f"{var.name}: unsupported encoding {var.encoding!r}")
            elif var.encoding == "cumulative_unary":
                if var.ordering_penalty is None:
                    raise ValueError(f"{var.name}: cumulative_unary requires ordering_penalty")
                if not isfinite(var.ordering_penalty) or var.ordering_penalty <= 0.0:
                    raise ValueError(f"{var.name}: ordering_penalty must be finite and positive")
            elif var.ordering_penalty is not None:
                raise ValueError(
                    f"{var.name}: ordering_penalty is only supported for cumulative_unary"
                )

        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate variable names: {duplicates}")

        variable_names = set(names)
        objective_names = set(self.objective.linear)
        for pair in self.objective.quadratic:
            if len(pair) != 2:
                raise ValueError(f"bad quadratic objective key {pair!r}")
            objective_names.update(pair)
        unknown_objective = sorted(objective_names - variable_names)
        if unknown_objective:
            raise ValueError(f"objective references unknown variables: {unknown_objective}")
        if not isfinite(self.objective.constant):
            raise ValueError("objective constant must be finite")
        for name, coeff in self.objective.linear.items():
            if not isfinite(coeff):
                raise ValueError(f"objective coefficient for {name!r} must be finite")
        for pair, coeff in self.objective.quadratic.items():
            if not isfinite(coeff):
                raise ValueError(f"objective coefficient for {pair!r} must be finite")

        constraint_names: list[str] = []
        generated_slack_names: set[str] = set()
        bit_names = set(variable_names)

        from .encodings import encode_variable

        for var in self.variables:
            enc = encode_variable(var)
            for bit in enc.bits:
                if isinstance(var, BinaryVar) and bit == var.name:
                    continue
                if bit in bit_names:
                    raise ValueError(f"generated bit name collision: {bit!r}")
                bit_names.add(bit)

        for constraint in self.constraints:
            constraint_names.append(constraint.name)
            if constraint.sense not in {"<=", ">=", "=="}:
                raise ValueError(f"{constraint.name}: unsupported sense {constraint.sense!r}")
            if constraint.slack_type not in {"continuous", "integer"}:
                raise ValueError(
                    f"{constraint.name}: unsupported slack_type {constraint.slack_type!r}"
                )
            if constraint.slack_encoding not in {
                "sbe",
                "unary",
                "digit_sum_unary",
                "cumulative_unary",
            }:
                raise ValueError(
                    f"{constraint.name}: unsupported slack encoding {constraint.slack_encoding!r}"
                )
            if not isfinite(constraint.rhs):
                raise ValueError(f"{constraint.name}: rhs must be finite")
            if not isfinite(constraint.penalty) or constraint.penalty <= 0.0:
                raise ValueError(f"{constraint.name}: penalty must be finite and positive")
            if not isfinite(constraint.slack_lower):
                raise ValueError(f"{constraint.name}: slack lower bound must be finite")
            if constraint.slack_upper is not None and not isfinite(constraint.slack_upper):
                raise ValueError(f"{constraint.name}: slack upper bound must be finite")
            if constraint.slack_digits < 1:
                raise ValueError(f"{constraint.name}: slack digits must be >= 1")
            if constraint.sense != "==" and constraint.slack_encoding == "cumulative_unary":
                if constraint.slack_ordering_penalty is None:
                    raise ValueError(
                        f"{constraint.name}: cumulative_unary slack requires slack_ordering_penalty"
                    )
                if (
                    not isfinite(constraint.slack_ordering_penalty)
                    or constraint.slack_ordering_penalty <= 0.0
                ):
                    raise ValueError(
                        f"{constraint.name}: slack_ordering_penalty must be finite and positive"
                    )
            elif constraint.sense != "==" and constraint.slack_ordering_penalty is not None:
                raise ValueError(
                    f"{constraint.name}: slack_ordering_penalty is only supported for "
                    "cumulative_unary"
                )
            unknown = sorted(set(constraint.linear) - variable_names)
            if unknown:
                raise ValueError(f"{constraint.name}: references unknown variables {unknown}")
            for name, coeff in constraint.linear.items():
                if not isfinite(coeff):
                    raise ValueError(f"{constraint.name}: coefficient for {name!r} must be finite")

            if constraint.sense != "==":
                slack_name = constraint.slack_name or f"s_{constraint.name}"
                if slack_name in variable_names or slack_name in generated_slack_names:
                    raise ValueError(
                        f"{constraint.name}: generated slack name collision {slack_name!r}"
                    )
                generated_slack_names.add(slack_name)
                if constraint.slack_type == "integer":
                    upper = constraint.slack_upper
                    if upper is None:
                        raise ValueError(
                            f"{constraint.name}: integer slacks require explicit slack_upper"
                        )
                    if (
                        float(constraint.slack_lower).is_integer() is False
                        or float(upper).is_integer() is False
                    ):
                        raise ValueError(
                            f"{constraint.name}: integer slack bounds must be integral"
                        )
                    slack: Variable = IntegerVar(
                        slack_name,
                        int(round(constraint.slack_lower)),
                        int(round(upper)),
                    )
                else:
                    upper = constraint.slack_upper
                    if upper is None:
                        upper = max(constraint.slack_lower, 1.0)
                    slack = SlackVar(
                        slack_name,
                        constraint.slack_lower,
                        upper,
                        constraint.slack_digits,
                        constraint.slack_encoding,
                        constraint.slack_ordering_penalty,
                    )
                enc = encode_variable(slack)
                for bit in enc.bits:
                    if bit in bit_names:
                        raise ValueError(f"generated bit name collision: {bit!r}")
                    bit_names.add(bit)

        duplicate_constraints = sorted(
            {name for name in constraint_names if constraint_names.count(name) > 1}
        )
        if duplicate_constraints:
            raise ValueError(f"duplicate constraint names: {duplicate_constraints}")

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
                        ordering_penalty=(
                            None
                            if row.get("ordering_penalty") is None
                            else float(row["ordering_penalty"])
                        ),
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
                        ordering_penalty=(
                            None
                            if row.get("ordering_penalty") is None
                            else float(row["ordering_penalty"])
                        ),
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
                slack_ordering_penalty=(
                    None
                    if row.get("slack_ordering_penalty") is None
                    else float(row["slack_ordering_penalty"])
                ),
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
