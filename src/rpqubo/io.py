"""Import and export helpers for QUBO files."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from .builders import BuildResult, ConstraintBuildInfo
from .encodings import AffineEncoding
from .qubo import QUBO
from .variables import (
    BinaryVar,
    ContinuousVar,
    IntegerVar,
    Problem,
    SlackVar,
    Variable,
)


def load_problem_file(path: str | Path) -> Mapping[str, Any]:
    path = Path(path)
    text = path.read_text()
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml  # type: ignore[import-untyped]

        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, Mapping):
        raise ValueError(f"{path}: problem file must contain an object/mapping")
    return cast(Mapping[str, Any], data)


def export_qubo(qubo: QUBO, path: str | Path, fmt: str | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = (fmt or path.suffix.lstrip(".") or "json").lower()
    if fmt == "json":
        path.write_text(json.dumps(qubo.to_dict(), indent=2, sort_keys=True) + "\n")
        return
    if fmt in {"csv", "coo"}:
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["u", "v", "bias"])
            writer.writeheader()
            for var, bias in sorted(qubo.linear.items()):
                writer.writerow({"u": var, "v": var, "bias": bias})
            for (u, v), bias in sorted(qubo.quadratic.items()):
                writer.writerow({"u": u, "v": v, "bias": bias})
            if qubo.offset:
                writer.writerow({"u": "__offset__", "v": "__offset__", "bias": qubo.offset})
        return
    if fmt == "npz":
        import numpy as np

        variables = sorted(qubo.variables)
        index = {v: i for i, v in enumerate(variables)}
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        for var, bias in qubo.linear.items():
            rows.append(index[var])
            cols.append(index[var])
            data.append(bias)
        for (u, v), bias in qubo.quadratic.items():
            rows.append(index[u])
            cols.append(index[v])
            data.append(bias)
        np.savez(
            path,
            variables=np.array(variables),
            rows=np.array(rows),
            cols=np.array(cols),
            data=np.array(data),
            offset=np.array([qubo.offset]),
        )
        return
    if fmt in {"dimod", "bqm"}:
        with path.open("wb") as f:
            f.write(qubo.to_dimod_bqm().to_file().read())
        return
    raise ValueError(f"Unsupported QUBO export format {fmt!r}")


def load_qubo(path: str | Path) -> QUBO:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        return QUBO.from_dict(json.loads(path.read_text()))
    if suffix in {".csv", ".coo"}:
        linear: dict[str, float] = {}
        quadratic: dict[tuple[str, str], float] = {}
        offset = 0.0
        with path.open() as f:
            for row in csv.DictReader(f):
                u = row["u"]
                v = row["v"]
                bias = float(row["bias"])
                if u == "__offset__" and v == "__offset__":
                    offset += bias
                elif u == v:
                    linear[u] = linear.get(u, 0.0) + bias
                else:
                    first, second = sorted((u, v))
                    key = (first, second)
                    quadratic[key] = quadratic.get(key, 0.0) + bias
        return QUBO(linear=linear, quadratic=quadratic, offset=offset)
    if suffix == ".npz":
        import numpy as np

        npz = np.load(path, allow_pickle=False)
        variables = [str(v) for v in npz["variables"]]
        linear_npz: dict[str, float] = {}
        quadratic_npz: dict[tuple[str, str], float] = {}
        for row, col, bias in zip(npz["rows"], npz["cols"], npz["data"]):
            u = variables[int(row)]
            v = variables[int(col)]
            if u == v:
                linear_npz[u] = linear_npz.get(u, 0.0) + float(bias)
            else:
                first, second = sorted((u, v))
                quadratic_npz[(first, second)] = float(bias)
        return QUBO(
            linear=linear_npz,
            quadratic=quadratic_npz,
            offset=float(cast(Any, npz["offset"])[0]),
            variable_order=variables,
        )
    if suffix in {".dimod", ".bqm"}:
        import dimod

        with path.open("rb") as file:
            bqm = dimod.BinaryQuadraticModel.from_file(file)
        return QUBO(
            linear={str(variable): float(bias) for variable, bias in bqm.linear.items()},
            quadratic={
                (min(str(u), str(v)), max(str(u), str(v))): float(bias)
                for (u, v), bias in bqm.quadratic.items()
            },
            offset=float(bqm.offset),
            variable_order=[str(variable) for variable in bqm.variables],
        )
    raise ValueError(f"Unsupported QUBO file type: {path}")


def _variable_to_mapping(var: Variable) -> dict[str, object]:
    if isinstance(var, BinaryVar):
        return {"name": var.name, "type": "binary"}
    if isinstance(var, IntegerVar):
        return {
            "name": var.name,
            "type": "integer",
            "lower": var.lower,
            "upper": var.upper,
            "encoding": var.encoding,
            "strict_bounds": var.strict_bounds,
        }
    if isinstance(var, SlackVar):
        data: dict[str, object] = {
            "name": var.name,
            "type": "slack",
            "lower": var.lower,
            "upper": var.upper,
            "digits": var.digits,
            "encoding": var.encoding,
        }
        if var.ordering_penalty is not None:
            data["ordering_penalty"] = var.ordering_penalty
        return data
    data = {
        "name": var.name,
        "type": "continuous",
        "lower": var.lower,
        "upper": var.upper,
        "digits": var.digits,
        "encoding": var.encoding,
    }
    if var.ordering_penalty is not None:
        data["ordering_penalty"] = var.ordering_penalty
    return data


def _problem_to_mapping(problem: Problem) -> dict[str, object]:
    return {
        "name": problem.name,
        "variables": [_variable_to_mapping(var) for var in problem.variables],
        "objective": {
            "constant": problem.objective.constant,
            "linear": problem.objective.linear,
            "quadratic": [
                {"vars": [u, v], "coef": coeff}
                for (u, v), coeff in problem.objective.quadratic.items()
            ],
        },
        "constraints": [asdict(constraint) for constraint in problem.constraints],
    }


def _constraint_info_to_mapping(info: ConstraintBuildInfo) -> dict[str, object]:
    data = asdict(info)
    slack = info.slack_variable
    data["slack_variable"] = None if slack is None else _variable_to_mapping(slack)
    return data


def export_model(build_result: BuildResult, path: str | Path) -> None:
    """Export a reloadable model bundle with QUBO, encodings, and metadata."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "rpqubo-model-v1",
        "problem": _problem_to_mapping(build_result.problem),
        "qubo": build_result.qubo.to_dict(),
        "unscaled_qubo": (
            None if build_result.unscaled_qubo is None else build_result.unscaled_qubo.to_dict()
        ),
        "encodings": {
            name: encoding.to_dict() for name, encoding in build_result.encodings.items()
        },
        "constraints": [_constraint_info_to_mapping(info) for info in build_result.constraints],
        "variable_order": build_result.qubo.variable_order,
        "rescale_factor": build_result.rescale_factor,
        "metadata": dict(build_result.qubo.metadata),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _constraint_info_from_mapping(data: Mapping[str, Any]) -> ConstraintBuildInfo:
    slack_raw = data.get("slack_variable")
    slack: Variable | None = None
    if isinstance(slack_raw, Mapping):
        slack = _variable_from_mapping(slack_raw)
    return ConstraintBuildInfo(
        name=str(data["name"]),
        sense=str(data["sense"]),
        rhs=float(data["rhs"]),
        penalty=float(data["penalty"]),
        residual_constant=float(data["residual_constant"]),
        residual_coeffs={str(k): float(v) for k, v in dict(data["residual_coeffs"]).items()},
        slack_variable=slack,
    )


def _variable_from_mapping(row: Mapping[str, Any]) -> Variable:
    kind = str(row.get("type", row.get("kind", ""))).lower()
    name = str(row["name"])
    if kind in {"binary", "bool"}:
        return BinaryVar(name)
    if kind in {"integer", "int"}:
        return IntegerVar(
            name,
            int(row["lower"]),
            int(row["upper"]),
            str(row.get("encoding", "binary")),
            bool(row.get("strict_bounds", True)),
        )
    if kind == "slack":
        return SlackVar(
            name,
            float(row.get("lower", 0.0)),
            float(row["upper"]),
            int(row.get("digits", 1)),
            str(row.get("encoding", "sbe")),
            (None if row.get("ordering_penalty") is None else float(row["ordering_penalty"])),
        )
    return ContinuousVar(
        name,
        float(row["lower"]),
        float(row["upper"]),
        int(row.get("digits", 1)),
        str(row.get("encoding", "sbe")),
        None if row.get("ordering_penalty") is None else float(row["ordering_penalty"]),
    )


def load_model(path: str | Path) -> BuildResult:
    """Load a model bundle exported by :func:`export_model`."""

    data = load_problem_file(path)
    if data.get("format") != "rpqubo-model-v1":
        raise ValueError("unsupported model bundle format")
    problem = Problem.from_mapping(cast(Mapping[str, Any], data["problem"]))
    qubo = QUBO.from_dict(cast(Mapping[str, Any], data["qubo"]))
    unscaled_raw = data.get("unscaled_qubo")
    unscaled_qubo = None
    if isinstance(unscaled_raw, Mapping):
        unscaled_qubo = QUBO.from_dict(unscaled_raw)
    encodings = {
        str(name): AffineEncoding.from_dict(cast(Mapping[str, Any], payload))
        for name, payload in dict(data.get("encodings", {})).items()
    }
    constraints = [
        _constraint_info_from_mapping(cast(Mapping[str, Any], row))
        for row in data.get("constraints", [])
    ]
    return BuildResult(
        problem=problem,
        qubo=qubo,
        encodings=encodings,
        constraints=constraints,
        unscaled_qubo=unscaled_qubo,
        rescale_factor=float(data.get("rescale_factor", 1.0)),
    )
