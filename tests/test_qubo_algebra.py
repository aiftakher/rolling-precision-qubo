from __future__ import annotations

from itertools import product

from rpqubo.builders import build_qubo
from rpqubo.qubo import QUBO, add_product_of_affines, add_square_of_affine
from rpqubo.variables import ContinuousVar, LinearConstraint, Problem, QuadraticObjective


def test_square_of_affine_expansion_matches_direct() -> None:
    q = QUBO()
    coeffs = {"a": 0.25, "b": -0.5}
    add_square_of_affine(q, 0.75, coeffs, 3.0)
    for a, b in product([0, 1], repeat=2):
        sample = {"a": a, "b": b}
        direct = 3.0 * (0.75 + 0.25 * a - 0.5 * b) ** 2
        assert abs(q.energy(sample) - direct) <= 1e-12


def test_product_of_affines_expansion_matches_direct() -> None:
    q = QUBO()
    add_product_of_affines(q, 0.2, {"a": 0.5}, -0.1, {"a": 0.25, "b": 0.75}, 2.0)
    for a, b in product([0, 1], repeat=2):
        sample = {"a": a, "b": b}
        direct = 2.0 * (0.2 + 0.5 * a) * (-0.1 + 0.25 * a + 0.75 * b)
        assert abs(q.energy(sample) - direct) <= 1e-12


def test_encoded_penalty_objective_matches_direct_decoded_value() -> None:
    problem = Problem(
        variables=[ContinuousVar("x", 0.0, 1.0, 1)],
        objective=QuadraticObjective(
            constant=0.35**2,
            linear={"x": -0.7},
            quadratic={("x", "x"): 1.0},
        ),
        constraints=[LinearConstraint("c", {"x": 1.0}, "<=", 0.5, penalty=10.0, slack_name="s")],
    )
    build = build_qubo(problem)
    sample = {bit: 0 for bit in build.qubo.variables}
    sample["z_x_1_2"] = 1
    sample["z_s_1_3"] = 1
    decoded = build.decode_sample(sample)
    direct = build.objective_value(decoded) + 10.0 * (decoded["x"] + decoded["s"] - 0.5) ** 2
    assert abs(build.qubo.energy(sample) - direct) <= 1e-12
