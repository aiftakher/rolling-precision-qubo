"""Build QUBO objects from structured problem specifications."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import fsum

from .encodings import (
    AffineEncoding,
    add_cumulative_unary_order_penalty,
    encode_variable,
)
from .qubo import QUBO, add_affine, add_product_of_encodings, add_square_of_affine
from .variables import (
    ContinuousVar,
    IntegerVar,
    LinearConstraint,
    Problem,
    SlackVar,
    Variable,
    variable_bounds,
)


@dataclass
class ConstraintBuildInfo:
    name: str
    sense: str
    rhs: float
    penalty: float
    residual_constant: float
    residual_coeffs: dict[str, float]
    slack_variable: Variable | None = None


@dataclass
class BuildResult:
    problem: Problem
    qubo: QUBO
    encodings: dict[str, AffineEncoding]
    constraints: list[ConstraintBuildInfo] = field(default_factory=list)
    unscaled_qubo: QUBO | None = None
    rescale_factor: float = 1.0

    def decode_sample(
        self, sample: Mapping[str, int], *, validate: bool = True
    ) -> dict[str, float]:
        return {name: enc.decode(sample, validate=validate) for name, enc in self.encodings.items()}

    def objective_value(self, decoded: Mapping[str, float]) -> float:
        obj = self.problem.objective
        terms = [obj.constant]
        terms.extend(coeff * decoded[name] for name, coeff in obj.linear.items())
        terms.extend(coeff * decoded[u] * decoded[v] for (u, v), coeff in obj.quadratic.items())
        return fsum(terms)

    def residuals(self, decoded: Mapping[str, float]) -> dict[str, float]:
        values: dict[str, float] = {}
        for info in self.constraints:
            lhs = 0.0
            constraint = next(c for c in self.problem.constraints if c.name == info.name)
            for name, coeff in constraint.linear.items():
                lhs += coeff * decoded[name]
            if constraint.sense == "==":
                values[constraint.name] = lhs - constraint.rhs
            elif constraint.sense == "<=":
                values[constraint.name] = max(0.0, lhs - constraint.rhs)
                if info.slack_variable is not None:
                    values[f"{constraint.name}_slack_residual"] = (
                        lhs + decoded[info.slack_variable.name] - constraint.rhs
                    )
            else:
                values[constraint.name] = max(0.0, constraint.rhs - lhs)
                if info.slack_variable is not None:
                    values[f"{constraint.name}_slack_residual"] = (
                        lhs - decoded[info.slack_variable.name] - constraint.rhs
                    )
        return values

    def feasibility(self, decoded: Mapping[str, float]) -> float:
        residuals = self.residuals(decoded)
        return max((abs(v) for v in residuals.values()), default=0.0)


def _add_objective(result: BuildResult) -> None:
    qubo = result.qubo
    obj = result.problem.objective
    qubo.add_offset(obj.constant)
    for name, coeff in obj.linear.items():
        add_affine(qubo, result.encodings[name], coeff)
    for (u, v), coeff in obj.quadratic.items():
        left = result.encodings[u]
        right = result.encodings[v]
        if u == v:
            add_square_of_affine(qubo, left.offset, left.weights, coeff)
        else:
            add_product_of_encodings(qubo, left, right, coeff)


def _linear_bounds(
    linear: Mapping[str, float], variables: Mapping[str, Variable]
) -> tuple[float, float]:
    low = 0.0
    high = 0.0
    for name, coeff in linear.items():
        lower, upper = variable_bounds(variables[name])
        if coeff >= 0:
            low += coeff * lower
            high += coeff * upper
        else:
            low += coeff * upper
            high += coeff * lower
    return low, high


def _make_slack(constraint: LinearConstraint, variables: Mapping[str, Variable]) -> Variable | None:
    if constraint.sense == "==":
        return None
    low, high = _linear_bounds(constraint.linear, variables)
    if constraint.sense == "<=":
        upper = max(0.0, constraint.rhs - low)
    else:
        upper = max(0.0, high - constraint.rhs)
    if constraint.slack_upper is not None:
        upper = constraint.slack_upper
    name = constraint.slack_name or f"s_{constraint.name}"
    if constraint.slack_type == "integer":
        return IntegerVar(
            name=name,
            lower=int(round(constraint.slack_lower)),
            upper=int(round(upper)),
            strict_bounds=True,
        )
    return SlackVar(
        name=name,
        lower=constraint.slack_lower,
        upper=upper,
        digits=constraint.slack_digits,
        encoding=constraint.slack_encoding,
        ordering_penalty=constraint.slack_ordering_penalty,
    )


def _constraint_residual_affine(
    constraint: LinearConstraint,
    encodings: Mapping[str, AffineEncoding],
    slack: Variable | None,
) -> tuple[float, dict[str, float]]:
    constant = -constraint.rhs
    coeffs: dict[str, float] = {}
    for name, coeff in constraint.linear.items():
        enc = encodings[name]
        constant += coeff * enc.offset
        for bit, weight in enc.weights.items():
            coeffs[bit] = coeffs.get(bit, 0.0) + coeff * weight

    if slack is not None:
        s_enc = encodings[slack.name]
        sign = 1.0 if constraint.sense == "<=" else -1.0
        constant += sign * s_enc.offset
        for bit, weight in s_enc.weights.items():
            coeffs[bit] = coeffs.get(bit, 0.0) + sign * weight
    return constant, coeffs


def build_qubo(problem: Problem, *, rescale: float | None = None) -> BuildResult:
    problem.validate()
    if rescale is not None and rescale <= 0.0:
        raise ValueError("rescale must be strictly positive")

    variables = list(problem.variables)
    variable_map = problem.variable_map
    generated_slacks: list[Variable] = []
    for constraint in problem.constraints:
        slack = _make_slack(constraint, variable_map)
        if slack is not None:
            generated_slacks.append(slack)
            variable_map = {**variable_map, slack.name: slack}
    variables.extend(generated_slacks)

    encodings = {var.name: encode_variable(var) for var in variables}
    variable_order = [bit for var in variables for bit in encodings[var.name].bits]
    qubo = QUBO(
        variable_groups={name: enc.bits for name, enc in encodings.items()},
        variable_order=variable_order,
    )
    expanded_problem = Problem(
        name=problem.name,
        variables=variables,
        objective=problem.objective,
        constraints=problem.constraints,
    )
    result = BuildResult(problem=expanded_problem, qubo=qubo, encodings=encodings)
    for var in variables:
        encoding = encodings[var.name]
        if encoding.encoding == "cumulative_unary":
            if not isinstance(var, (ContinuousVar, SlackVar)):
                raise ValueError(f"{var.name}: cumulative_unary is only valid for grid variables")
            strength = getattr(var, "ordering_penalty", None)
            if strength is None:
                raise ValueError(f"{var.name}: cumulative_unary requires ordering_penalty")
            add_cumulative_unary_order_penalty(qubo, encoding, float(strength))
    _add_objective(result)

    for constraint in problem.constraints:
        slack_name = constraint.slack_name or f"s_{constraint.name}"
        slack = next((s for s in generated_slacks if s.name == slack_name), None)
        constant, coeffs = _constraint_residual_affine(constraint, encodings, slack)
        add_square_of_affine(qubo, constant, coeffs, constraint.penalty)
        result.constraints.append(
            ConstraintBuildInfo(
                name=constraint.name,
                sense=constraint.sense,
                rhs=constraint.rhs,
                penalty=constraint.penalty,
                residual_constant=constant,
                residual_coeffs=coeffs,
                slack_variable=slack,
            )
        )

    if rescale is not None:
        from .qubo import rescaled_qubo

        result.unscaled_qubo = result.qubo.copy()
        result.qubo, factor = rescaled_qubo(result.unscaled_qubo, target_max_abs=rescale)
        result.rescale_factor = factor
    return result


def build_qubo_from_mapping(
    data: Mapping[str, object], *, rescale: float | None = None
) -> BuildResult:
    return build_qubo(Problem.from_mapping(data), rescale=rescale)
