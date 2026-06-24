"""Command-line interface for rpqubo."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

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
        var = ContinuousVar(
            args.name, float(args.lower), float(args.upper), args.digits, encoding
        )
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
    report = {
        "repository_commit": _git(["rev-parse", "HEAD"]),
        "repository_dirty": bool(_git(["status", "--porcelain"])),
        "python": sys.version,
        "package_versions": package_versions(),
        "configurations": {
            "table1": EXAMPLE1_BIT_GROWTH.name,
            "table2": EXAMPLE1_ZOOM.name,
            "table3": EXAMPLE2_BIT_GROWTH.name,
            "table4": EXAMPLE2_ZOOM.name,
            "table5": ALAN_BIT_GROWTH.name,
            "table6": ALAN_TABLE6_REFERENCE.name,
            "table9": ALAN_PENALTY_SENSITIVITY.name,
        },
        "checks": _paper_checks(generated),
        "written": written,
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


def _git(args: list[str]) -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return ""
    return proc.stdout.strip()


def _paper_checks(rows: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, object]]:
    checks: dict[str, dict[str, object]] = {}

    def sizes(name: str) -> list[tuple[int, int]]:
        return [(int(row["n_vars"]), int(row["n_quad"])) for row in rows.get(name, [])]

    checks["table1"] = {
        "passed": sizes("ex1_unconstrained_bit_growth")
        == [(18, 72), (34, 272), (50, 600), (66, 1056)]
    }
    checks["table2"] = {
        "passed": sizes("ex1_unconstrained_zoom") == [(10, 20)] * 5
    }
    checks["table3"] = {
        "passed": sizes("ex2_miqp_bit_growth") == [(11, 55), (19, 171), (35, 595)]
    }
    table4 = rows.get("ex2_miqp_zoom", [])
    checks["table4"] = {
        "passed": [row.get("action") for row in table4]
        == ["baseline", "accepted_zoom", "accepted_zoom", "backtrack", "accepted_zoom"]
    }
    checks["table5"] = {
        "passed": sizes("ex3_alan_bit_growth")
        == [(49, 406), (85, 1248), (121, 2554), (157, 4324), (173, 5820), (301, 16044)]
    }
    table6 = rows.get("ex3_alan_zoom", [])
    checks["table6"] = {
        "passed": len(table6) >= 2
        and table6[0].get("action") == "baseline"
        and table6[1].get("action") == "accepted_zoom"
        and abs(float(table6[1].get("objective", 0.0)) - 2.928) <= 1e-9
    }
    checks["table9"] = {
        "passed": len(rows.get("alan_penalty_sensitivity", [])) == 7
    }
    checks["nonlinear"] = {
        "passed": len(rows.get("nonlinear_encoding_error", [])) == 6
    }
    return checks


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
        choices=["json", "csv", "coo", "npz", "dimod", "model"],
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
    repro.set_defaults(func=_cmd_reproduce_paper)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
