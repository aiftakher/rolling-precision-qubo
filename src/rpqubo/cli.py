"""Command-line interface for rpqubo."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

from . import paper_expectations as expect
from .builders import build_qubo_from_mapping
from .encodings import encode_variable, nonlinear_error_table
from .examples import (
    reproduce_alan_bit_growth,
    reproduce_alan_penalty_sensitivity,
    reproduce_alan_zoom,
    reproduce_example1_bit_growth,
    reproduce_example1_zoom,
    reproduce_example2_bit_growth,
    reproduce_example2_zoom,
)
from .io import export_model, export_qubo, load_model, load_problem_file, load_qubo
from .metrics import qubo_summary
from .paper_config import (
    ALAN_BIT_GROWTH,
    ALAN_PENALTY_SENSITIVITY,
    ALAN_TABLE6_REFERENCE,
    EXAMPLE1_BIT_GROWTH,
    EXAMPLE1_ZOOM,
    EXAMPLE2_BIT_GROWTH,
    EXAMPLE2_ZOOM,
    package_versions,
)
from .solvers import solve_qubo, solved_payload
from .variables import BinaryVar, ContinuousVar, IntegerVar, SlackVar, Variable


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _cmd_encode_variable(args: argparse.Namespace) -> None:
    kind = args.type.lower()
    encoding = args.encoding
    if encoding is None:
        encoding = "binary" if kind == "integer" else "sbe"
    var: Variable
    if kind == "binary":
        var = BinaryVar(args.name)
    elif kind == "integer":
        var = IntegerVar(
            args.name,
            int(args.lower),
            int(args.upper),
            encoding=encoding,
            strict_bounds=True,
        )
    elif kind == "slack":
        var = SlackVar(args.name, float(args.lower), float(args.upper), args.digits, encoding)
    else:
        var = ContinuousVar(args.name, float(args.lower), float(args.upper), args.digits, encoding)
    _print_json(encode_variable(var).to_dict())


def _cmd_build_qubo(args: argparse.Namespace) -> None:
    if args.rescale is not None and args.rescale <= 0.0:
        raise ValueError("--rescale must be positive")
    data = load_problem_file(args.problem)
    result = build_qubo_from_mapping(data, rescale=args.rescale)
    if args.format == "model":
        export_model(result, args.output)
    else:
        export_qubo(result.qubo, args.output, args.format)
    _print_json({"output": args.output, "summary": qubo_summary(result.qubo)})


def _cmd_inspect_qubo(args: argparse.Namespace) -> None:
    _print_json(qubo_summary(load_qubo(args.path)))


def _solver_options(args: argparse.Namespace) -> dict[str, Any]:
    opts: dict[str, Any] = {}
    if args.num_reads is not None:
        if args.num_reads <= 0:
            raise ValueError("--num-reads must be positive")
        opts["num_reads"] = args.num_reads
    if args.sweeps is not None:
        if args.sweeps <= 0:
            raise ValueError("--sweeps must be positive")
        opts["sweeps"] = args.sweeps
    if args.seed is not None:
        opts["seed"] = args.seed
    return opts


def _looks_like_structured_problem(data: Any) -> bool:
    return isinstance(data, dict) and "variables" in data


def _looks_like_qubo_mapping(data: Any) -> bool:
    return isinstance(data, dict) and ("linear" in data or "quadratic" in data)


def _cmd_solve(args: argparse.Namespace) -> None:
    if args.rescale is not None and args.rescale <= 0.0:
        raise ValueError("--rescale must be positive")
    path = Path(args.input)
    decoded = None
    if path.suffix.lower() == ".model":
        build = load_model(path)
        result = solve_qubo(build.qubo, args.solver, **_solver_options(args))
        decoded = build.decode_sample(result.sample)
        payload = solved_payload(build.qubo, result, decoded)
        payload["objective"] = build.objective_value(decoded)
        payload["feasibility"] = build.feasibility(decoded)
        _print_json(payload)
        return
    if path.suffix.lower() in {".json", ".yaml", ".yml"}:
        data = load_problem_file(path)
        if _looks_like_structured_problem(data):
            build = build_qubo_from_mapping(data, rescale=args.rescale)
            result = solve_qubo(build.qubo, args.solver, **_solver_options(args))
            decoded = build.decode_sample(result.sample)
            payload = solved_payload(build.qubo, result, decoded)
            payload["objective"] = build.objective_value(decoded)
            payload["feasibility"] = build.feasibility(decoded)
            _print_json(payload)
            return
        if not _looks_like_qubo_mapping(data):
            raise ValueError(
                "JSON/YAML input is neither a structured problem nor a raw QUBO mapping"
            )
    qubo = load_qubo(path)
    result = solve_qubo(qubo, args.solver, **_solver_options(args))
    _print_json(solved_payload(qubo, result, decoded))


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, sort_keys=True)
                        if isinstance(value, (dict, list, tuple))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def _cmd_reproduce_paper(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    selected = args.example
    written: list[str] = []
    generated: dict[str, list[dict[str, Any]]] = {}

    def write(name: str, rows: list[dict[str, Any]]) -> None:
        path = out_dir / f"{name}.csv"
        _write_rows(path, rows)
        written.append(str(path))
        generated[name] = rows

    if selected in {"ex1", "all"}:
        write("ex1_unconstrained_bit_growth", reproduce_example1_bit_growth())
        write("ex1_unconstrained_zoom", reproduce_example1_zoom())
    if selected in {"ex2", "all"}:
        write("ex2_miqp_bit_growth", reproduce_example2_bit_growth())
        write("ex2_miqp_zoom", reproduce_example2_zoom())
    if selected in {"alan", "all"}:
        write("ex3_alan_bit_growth", reproduce_alan_bit_growth())
        write("ex3_alan_zoom", reproduce_alan_zoom())
    if selected in {"nonlinear", "all"}:
        write("nonlinear_encoding_error", nonlinear_error_table([0.6, 1.5], [1, 2, 3]))
    if selected in {"penalty-sensitivity", "all"}:
        write("alan_penalty_sensitivity", reproduce_alan_penalty_sensitivity())
    report_path = out_dir / "reproducibility_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checks = _paper_checks(generated)
    overall_passed = all(bool(check["passed"]) for check in checks.values())
    all_exact = all(bool(check["exact_reference_match"]) for check in checks.values())
    git_status = _git(["status", "--porcelain"])
    report = {
        "overall_passed": overall_passed,
        "all_exact_reference_matches": all_exact,
        "repository_commit": _git(["rev-parse", "HEAD"]),
        "repository_dirty": None if git_status is None else bool(git_status),
        "python": sys.version,
        "python_requires": ">=3.10",
        "platform": platform.platform(),
        "package_versions": package_versions(),
        "solver_parameters": {
            "example1_bit_growth": EXAMPLE1_BIT_GROWTH.anneal.__dict__,
            "example1_zoom": EXAMPLE1_ZOOM.anneal.__dict__,
            "example2_bit_growth": EXAMPLE2_BIT_GROWTH.anneal.__dict__,
            "example2_zoom": EXAMPLE2_ZOOM.anneal.__dict__,
            "alan_bit_growth": ALAN_BIT_GROWTH.anneal.__dict__,
            "alan_table6_reference": ALAN_TABLE6_REFERENCE.anneal.__dict__,
            "alan_penalty_sensitivity": ALAN_PENALTY_SENSITIVITY.anneal.__dict__,
        },
        "configurations": {
            "table1": EXAMPLE1_BIT_GROWTH.name,
            "table2": EXAMPLE1_ZOOM.name,
            "table3": EXAMPLE2_BIT_GROWTH.name,
            "table4": EXAMPLE2_ZOOM.name,
            "table5": ALAN_BIT_GROWTH.name,
            "table6": ALAN_TABLE6_REFERENCE.name,
            "table9": ALAN_PENALTY_SENSITIVITY.name,
        },
        "checks": checks,
        "written": written,
        "accepted_table9_reference": {
            "path": str(expect.TABLE9_REFERENCE_PATH),
            "sha256": expect.reference_table9_sha256() or expect.TABLE9_REFERENCE_SHA256,
        },
        "neal_heuristic_notice": (
            "neal is a heuristic simulated annealing backend; it does not certify "
            "global optimality and seeded tie-breaking may differ across environments."
        ),
        "known_discrepancies": [
            {
                "table": "Alan Table 6",
                "manuscript": "uniform penalties of 500 are described in prose",
                "reference_code": {"e1": 200.0, "e2": 200.0, "link": 300.0, "card": 200.0},
            }
        ],
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    written.append(str(report_path))
    _print_json({"written": written})
    if args.strict and not overall_passed:
        raise SystemExit(1)


def _git(args: list[str]) -> str | None:
    try:
        proc = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _paper_checks(rows: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, object]]:
    checks: dict[str, dict[str, object]] = {}

    if "ex1_unconstrained_bit_growth" in rows:
        checks["table1"] = _check_table1(rows["ex1_unconstrained_bit_growth"])
    if "ex1_unconstrained_zoom" in rows:
        checks["table2"] = _check_table2(rows["ex1_unconstrained_zoom"])
    if "ex2_miqp_bit_growth" in rows:
        checks["table3"] = _check_table3(rows["ex2_miqp_bit_growth"])
    if "ex2_miqp_zoom" in rows:
        checks["table4"] = _check_table4(rows["ex2_miqp_zoom"])
    if "ex3_alan_bit_growth" in rows:
        checks["table5"] = _check_table5(rows["ex3_alan_bit_growth"])
    if "ex3_alan_zoom" in rows:
        checks["table6"] = _check_table6(rows["ex3_alan_zoom"])
    if "alan_penalty_sensitivity" in rows:
        checks["table9"] = _check_table9(rows["alan_penalty_sensitivity"])
    if "nonlinear_encoding_error" in rows:
        checks["nonlinear"] = _check_nonlinear(rows["nonlinear_encoding_error"])
    return checks


def _check_result(
    passed: bool,
    *,
    details: dict[str, object],
    status: str = "exact_reference_match",
    exact: bool = True,
) -> dict[str, object]:
    return {
        "passed": passed,
        "status": status if passed else "failed",
        "exact_reference_match": bool(passed and exact),
        "details": details,
    }


def _as_float(value: object) -> float:
    return float(cast(Any, value))


def _as_int(value: object) -> int:
    return int(cast(Any, value))


def _close(value: object, expected: float, tolerance: float = expect.VALUE_ATOL) -> bool:
    return abs(_as_float(value) - expected) <= tolerance


def _objective_close(value: object, expected: float) -> bool:
    return abs(_as_float(value) - expected) <= max(
        expect.OBJECTIVE_ATOL,
        abs(expected) * 1e-8,
    )


def _x_values(row: dict[str, Any]) -> tuple[float, float, float, float]:
    raw = row["x"]
    if isinstance(raw, dict):
        return tuple(float(raw[f"x{i}"]) for i in range(1, 5))  # type: ignore[return-value]
    if isinstance(raw, str):
        stripped = raw.strip()
        if stripped.startswith("{"):
            parsed = json.loads(stripped)
            return tuple(float(parsed[f"x{i}"]) for i in range(1, 5))  # type: ignore[return-value]
        values = [float(part) for part in stripped.strip("[]").split(",")]
        return tuple(values[:4])  # type: ignore[return-value]
    raise TypeError("x field must be a mapping or encoded list")


def _check_table1(rows: list[dict[str, Any]]) -> dict[str, object]:
    failures: list[str] = []
    if len(rows) != len(expect.TABLE1):
        failures.append("wrong row count")
    for index, expected in enumerate(expect.TABLE1):
        if index >= len(rows):
            break
        row = rows[index]
        for key in ("J1", "J2", "n_vars", "n_quad"):
            if _as_int(row[key]) != _as_int(expected[key]):
                failures.append(f"row {index} {key}")
        for key in ("x1", "x2"):
            if not _close(row[key], _as_float(expected[key]), 5e-7):
                failures.append(f"row {index} {key}")
        if not _objective_close(row["objective"], _as_float(expected["objective"])):
            failures.append(f"row {index} objective")
    return _check_result(not failures, details={"failures": failures})


def _check_table2(rows: list[dict[str, Any]]) -> dict[str, object]:
    failures: list[str] = []
    if len(rows) != len(expect.TABLE2):
        failures.append("wrong row count")
    for index, expected in enumerate(expect.TABLE2):
        if index >= len(rows):
            break
        row = rows[index]
        if row.get("action") != expected["action"]:
            failures.append(f"row {index} action")
        for key in ("x1", "x2"):
            if not _close(row[key], _as_float(expected[key]), 5e-6):
                failures.append(f"row {index} {key}")
        if int(row["n_vars"]) != 10 or int(row["n_quad"]) != 20:
            failures.append(f"row {index} dimensions")
    return _check_result(not failures, details={"failures": failures})


def _check_table3(rows: list[dict[str, Any]]) -> dict[str, object]:
    failures: list[str] = []
    tied_state: tuple[float, int, float] | None = None
    if len(rows) != len(expect.TABLE3):
        failures.append("wrong row count")
    for index, expected in enumerate(expect.TABLE3):
        if index >= len(rows):
            break
        row = rows[index]
        if _as_int(row["Jx"]) != _as_int(expected["Jx"]) or _as_int(row["Js"]) != _as_int(
            expected["Js"]
        ):
            failures.append(f"row {index} precision")
        if _as_int(row["n_vars"]) != _as_int(expected["n_vars"]) or _as_int(
            row["n_quad"]
        ) != _as_int(expected["n_quad"]):
            failures.append(f"row {index} dimensions")
        if not _objective_close(row["objective"], _as_float(expected["objective"])):
            failures.append(f"row {index} objective")
        if abs(_as_float(row["feasibility"])) > 1e-9:
            failures.append(f"row {index} feasibility")
        if index == 0:
            tied_state = (
                round(_as_float(row["x"]), 10),
                _as_int(row["y"]),
                round(_as_float(row["s"]), 10),
            )
            tied_expected = {
                (round(x, 10), y, round(s, 10)) for x, y, s in expect.TABLE3_J1_TIED_STATES
            }
            if tied_state not in tied_expected:
                failures.append("row 0 tied state")
        else:
            for key in ("x", "s"):
                if not _close(row[key], _as_float(expected[key]), 5e-9):
                    failures.append(f"row {index} {key}")
            if _as_int(row["y"]) != _as_int(expected["y"]):
                failures.append(f"row {index} y")
    exact = tied_state == expect.TABLE3_J1_TIED_STATES[0]
    status = "exact_reference_match" if exact else "accepted_tied_optimum"
    return _check_result(
        not failures,
        details={"failures": failures, "observed_j1_tied_state": tied_state},
        status=status,
        exact=exact,
    )


def _check_table4(rows: list[dict[str, Any]]) -> dict[str, object]:
    failures: list[str] = []
    if len(rows) != len(expect.TABLE4):
        failures.append("wrong row count")
    for index, expected in enumerate(expect.TABLE4):
        if index >= len(rows):
            break
        row = rows[index]
        if row.get("action") != expected["action"]:
            failures.append(f"row {index} action")
        if not _close(row["x"], _as_float(expected["x"]), 5e-9):
            failures.append(f"row {index} x")
        if _as_int(row["n_vars"]) != 11 or _as_int(row["n_quad"]) != 55:
            failures.append(f"row {index} dimensions")
    if rows and abs(_as_float(rows[-1]["feasibility"])) > 1e-9:
        failures.append("final feasibility")
    return _check_result(not failures, details={"failures": failures})


def _check_table5(rows: list[dict[str, Any]]) -> dict[str, object]:
    failures: list[str] = []
    if len(rows) != len(expect.TABLE5):
        failures.append("wrong row count")
    for index, expected in enumerate(expect.TABLE5):
        if index >= len(rows):
            break
        row = rows[index]
        if _as_int(row["Jx"]) != _as_int(expected["Jx"]) or _as_int(row["Js"]) != _as_int(
            expected["Js"]
        ):
            failures.append(f"row {index} precision")
        if _as_int(row["n_vars"]) != _as_int(expected["n_vars"]) or _as_int(
            row["n_quad"]
        ) != _as_int(expected["n_quad"]):
            failures.append(f"row {index} dimensions")
        observed_x = _x_values(row)
        expected_x = cast(tuple[float, float, float, float], expected["x"])
        for observed, target in zip(observed_x, expected_x):
            if abs(observed - target) > 5e-6:
                failures.append(f"row {index} x")
                break
        if not _objective_close(row["objective"], _as_float(expected["objective"])):
            failures.append(f"row {index} objective")
        if abs(_as_float(row["feasibility"]) - _as_float(expected["feasibility"])) > 5e-6:
            failures.append(f"row {index} feasibility")
    return _check_result(not failures, details={"failures": failures})


def _check_table6(rows: list[dict[str, Any]]) -> dict[str, object]:
    failures: list[str] = []
    actions = tuple(str(row.get("action")) for row in rows)
    if actions != expect.TABLE6_ACTIONS:
        failures.append("action sequence")
    if len(rows) >= 2:
        for observed, target in zip(_x_values(rows[0]), expect.TABLE6_BASELINE_X):
            if abs(observed - target) > 5e-9:
                failures.append("baseline x")
                break
        for observed, target in zip(_x_values(rows[1]), expect.TABLE6_ACCEPTED_X):
            if abs(observed - target) > 5e-9:
                failures.append("accepted x")
                break
        if not _objective_close(rows[1]["objective"], expect.TABLE6_ACCEPTED_OBJECTIVE):
            failures.append("accepted objective")
    else:
        failures.append("missing accepted row")
    for index, row in enumerate(rows):
        if _as_int(row["n_vars"]) != 46 or _as_int(row["n_quad"]) != 385:
            failures.append(f"row {index} dimensions")
        if abs(_as_float(row["feasibility"])) > 5e-9:
            failures.append(f"row {index} feasibility")
    return _check_result(not failures, details={"failures": failures})


def _check_table9(rows: list[dict[str, Any]]) -> dict[str, object]:
    comparisons = [expect.compare_table9_row(row) for row in rows]
    observed_lambdas = {_as_float(row["lambda_input"]) for row in rows}
    expected_lambdas = set(expect.TABLE9_BY_LAMBDA)
    missing = sorted(expected_lambdas - observed_lambdas)
    extra = sorted(observed_lambdas - expected_lambdas)
    passed = (
        not missing
        and not extra
        and len(rows) == len(expect.TABLE9)
        and all(bool(row["passed"]) for row in comparisons)
    )
    exact = passed and all(bool(row["exact_reference_match"]) for row in comparisons)
    stochastic = passed and all(
        bool(row["exact_reference_match"]) or bool(row["within_stochastic_tolerance"])
        for row in comparisons
    )
    status = (
        "exact_reference_match"
        if exact
        else "within_documented_stochastic_tolerance"
        if stochastic
        else "failed"
    )
    return _check_result(
        passed,
        details={"rows": comparisons, "missing_lambdas": missing, "extra_lambdas": extra},
        status=status,
        exact=exact,
    )


def _check_nonlinear(rows: list[dict[str, Any]]) -> dict[str, object]:
    failures: list[str] = []
    lookup = {(_as_float(row["a"]), _as_float(row["J"])): row for row in rows}
    for expected in expect.NONLINEAR:
        key = (_as_float(expected["a"]), _as_float(expected["J"]))
        row = lookup.get(key)
        if row is None:
            failures.append(f"missing {key}")
            continue
        for metric in ("max_abs_error", "rmse"):
            if not _close(row[metric], _as_float(expected[metric]), 1e-12):
                failures.append(f"{key} {metric}")
    if len(rows) != len(expect.NONLINEAR):
        failures.append("wrong row count")
    return _check_result(not failures, details={"failures": failures})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rpqubo")
    sub = parser.add_subparsers(dest="command", required=True)

    enc = sub.add_parser("encode-variable")
    enc.add_argument("--name", required=True)
    enc.add_argument("--type", choices=["continuous", "integer", "binary", "slack"], required=True)
    enc.add_argument("--lower", default=0.0)
    enc.add_argument("--upper", default=1.0)
    enc.add_argument("--digits", type=int, default=1)
    enc.add_argument("--encoding", default=None)
    enc.set_defaults(func=_cmd_encode_variable)

    build = sub.add_parser("build-qubo")
    build.add_argument("problem")
    build.add_argument("--output", required=True)
    build.add_argument(
        "--format",
        choices=["json", "csv", "coo", "npz", "dimod", "bqm", "model"],
        default=None,
    )
    build.add_argument("--rescale", type=float, default=None)
    build.set_defaults(func=_cmd_build_qubo)

    inspect = sub.add_parser("inspect-qubo")
    inspect.add_argument("path")
    inspect.set_defaults(func=_cmd_inspect_qubo)

    solve = sub.add_parser("solve")
    solve.add_argument("input")
    solve.add_argument("--solver", choices=["exact", "neal", "random"], default="exact")
    solve.add_argument("--num-reads", type=int, default=None)
    solve.add_argument("--sweeps", type=int, default=None)
    solve.add_argument("--seed", type=int, default=None)
    solve.add_argument("--rescale", type=float, default=None)
    solve.set_defaults(func=_cmd_solve)

    repro = sub.add_parser("reproduce-paper")
    repro.add_argument(
        "--example",
        choices=["ex1", "ex2", "alan", "nonlinear", "penalty-sensitivity", "all"],
        default="all",
    )
    repro.add_argument("--output-dir", default="outputs/paper")
    repro.add_argument("--strict", action="store_true")
    repro.set_defaults(func=_cmd_reproduce_paper)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
