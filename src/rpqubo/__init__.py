"""Rolling-precision QUBO package."""

from .builders import BuildResult, build_qubo, build_qubo_from_mapping
from .encodings import (
    AffineEncoding,
    encode_variable,
    nonlinear_coefficients,
    nonlinear_error_table,
)
from .io import export_qubo, load_qubo
from .qubo import QUBO
from .solvers import SolveResult, solve_qubo
from .variables import BinaryVar, ContinuousVar, IntegerVar, SlackVar

__all__ = [
    "AffineEncoding",
    "BinaryVar",
    "BuildResult",
    "ContinuousVar",
    "IntegerVar",
    "QUBO",
    "SlackVar",
    "SolveResult",
    "build_qubo",
    "build_qubo_from_mapping",
    "encode_variable",
    "export_qubo",
    "load_qubo",
    "nonlinear_coefficients",
    "nonlinear_error_table",
    "solve_qubo",
]

__version__ = "0.1.0"
