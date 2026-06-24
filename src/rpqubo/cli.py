"""Command-line interface for rpqubo."""

from __future__ import annotations

import argparse
import csv
import json
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
from .io import export_qubo, load_problem_file, load_qubo
from .metrics import qubo_summary
from .solvers import solve_qubo, solved_payload
from .variables import BinaryVar, ContinuousVar, IntegerVar, SlackVar, Variable


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _cmd_encode_variable(args: argparse.Namespace) -> None:
    kind = args.type.lower()
    var: Variable
    if kind == "binary":
        var = BinaryVar(args.name)
    elif kind == "integer":
        var = IntegerVar(
            args.name,
            int(args.lower),
            int(args.upper),
            encoding=args.encoding,
            strict_bounds=not args.allow_out_of_range,
        )
    elif kind == "slack":
        var = SlackVar(args.name, float(args.lower), float(args.upper), args.digits, args.encoding)
    else:
        var = ContinuousVar(
            args.name, float(args.lower), float(args.upper), args.digits, args.encoding
        )
    _print_json(encode_variable(var).to_dict())


def _cmd_build_qubo(args: argparse.Namespace) -> None:
    data = load_problem_file(args.problem)
    result = build_qubo_from_mapping(data, rescale=args.rescale)
    export_qubo(result.qubo, args.output, args.format)
    _print_json({"output": args.output, "summary": qubo_summary(result.qubo)})


def _cmd_inspect_qubo(args: argparse.Namespace) -> None:
    _print_json(qubo_summary(load_qubo(args.path)))


def _solver_options(args: argparse.Namespace) -> dict[str, Any]:
    opts: dict[str, Any] = {}
    if args.num_reads is not None:
        opts["num_reads"] = args.num_reads
    if args.sweeps is not None:
        opts["sweeps"] = args.sweeps
    if args.seed is not None:
        opts["seed"] = args.seed
    return opts


def _cmd_solve(args: argparse.Namespace) -> None:
    path = Path(args.input)
    decoded = None
    if path.suffix.lower() in {".json", ".yaml", ".yml"}:
        try:
            data = load_problem_file(path)
            if "variables" in data:
                build = build_qubo_from_mapping(data, rescale=args.rescale)
                result = solve_qubo(build.qubo, args.solver, **_solver_options(args))
                decoded = build.decode_sample(result.sample)
                payload = solved_payload(build.qubo, result, decoded)
                payload["objective"] = build.objective_value(decoded)
                payload["feasibility"] = build.feasibility(decoded)
                _print_json(payload)
                return
        except Exception:
            if path.suffix.lower() != ".json":
                raise
    qubo = load_qubo(path)
    result = solve_qubo(qubo, args.solver, **_solver_options(args))
    _print_json(solved_payload(qubo, result, decoded))


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _cmd_reproduce_paper(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    selected = args.example
    written: list[str] = []

    def write(name: str, rows: list[dict[str, Any]]) -> None:
        path = out_dir / f"{name}.csv"
        _write_rows(path, rows)
        written.append(str(path))

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
    _print_json({"written": written})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rpqubo")
    sub = parser.add_subparsers(dest="command", required=True)

    enc = sub.add_parser("encode-variable")
    enc.add_argument("--name", required=True)
    enc.add_argument("--type", choices=["continuous", "integer", "binary", "slack"], required=True)
    enc.add_argument("--lower", default=0.0)
    enc.add_argument("--upper", default=1.0)
    enc.add_argument("--digits", type=int, default=1)
    enc.add_argument("--encoding", default="sbe")
    enc.add_argument("--allow-out-of-range", action="store_true")
    enc.set_defaults(func=_cmd_encode_variable)

    build = sub.add_parser("build-qubo")
    build.add_argument("problem")
    build.add_argument("--output", required=True)
    build.add_argument("--format", choices=["json", "csv", "coo", "npz", "dimod"], default=None)
    build.add_argument("--rescale", type=float, default=None)
    build.set_defaults(func=_cmd_build_qubo)

    inspect = sub.add_parser("inspect-qubo")
    inspect.add_argument("path")
    inspect.set_defaults(func=_cmd_inspect_qubo)

    solve = sub.add_parser("solve")
    solve.add_argument("input")
    solve.add_argument("--solver", default="exact")
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
