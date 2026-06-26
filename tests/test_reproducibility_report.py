from __future__ import annotations

import argparse
import json

import pytest

from rpqubo import cli
from rpqubo.paper_expectations import NONLINEAR, TABLE3, TABLE9_BY_LAMBDA


def test_paper_checks_accept_table3_tied_optimum() -> None:
    rows = [
        {
            "Jx": 1,
            "Js": 1,
            "x": 0.3,
            "y": 0,
            "s": 0.7,
            "objective": 0.0025,
            "feasibility": 0.0,
            "n_vars": 11,
            "n_quad": 55,
        },
        {
            **TABLE3[1],
            "x": 0.35,
            "y": 0,
            "s": 0.65,
        },
        {
            **TABLE3[2],
            "x": 0.35,
            "y": 0,
            "s": 0.65,
        },
    ]
    check = cli._paper_checks({"ex2_miqp_bit_growth": rows})["table3"]
    assert check["passed"] is True
    assert check["status"] == "accepted_tied_optimum"
    assert check["exact_reference_match"] is False


def test_paper_checks_report_table9_stochastic_status() -> None:
    rows = []
    for expected in TABLE9_BY_LAMBDA.values():
        rows.append(
            {
                "lambda_input": expected.lambda_input,
                "objective": expected.objective,
                "feasibility": expected.feasibility,
                "e1_abs": expected.e1_abs,
                "e2_abs": expected.e2_abs,
                "n_vars": expected.n_vars,
                "n_quad": expected.n_quad,
                "qubo_max_abs_unscaled": expected.qubo_max_abs_unscaled,
                "qubo_dyn_range_unscaled": expected.qubo_dyn_range_unscaled,
            }
        )
    rows[0]["objective"] = float(rows[0]["objective"]) + 0.01
    check = cli._paper_checks({"alan_penalty_sensitivity": rows})["table9"]
    assert check["passed"] is True
    assert check["status"] == "within_documented_stochastic_tolerance"
    assert check["exact_reference_match"] is False


def test_reproduce_paper_strict_exits_after_failed_report(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        cli,
        "nonlinear_error_table",
        lambda *_args, **_kwargs: [{"a": 0.6, "J": 1.0, "max_abs_error": 99.0, "rmse": 99.0}],
    )
    args = argparse.Namespace(example="nonlinear", output_dir=str(tmp_path), strict=True)
    with pytest.raises(SystemExit):
        cli._cmd_reproduce_paper(args)
    report = json.loads((tmp_path / "reproducibility_report.json").read_text())
    assert report["overall_passed"] is False
    assert report["all_exact_reference_matches"] is False


def test_reproduce_paper_report_contains_release_metadata(tmp_path) -> None:
    args = argparse.Namespace(example="nonlinear", output_dir=str(tmp_path), strict=False)
    cli._cmd_reproduce_paper(args)
    report = json.loads((tmp_path / "reproducibility_report.json").read_text())
    assert report["overall_passed"] is True
    assert report["all_exact_reference_matches"] is True
    assert report["python_requires"] == ">=3.10"
    assert "platform" in report
    assert "accepted_table9_reference" in report
    assert "neal is a heuristic" in report["neal_heuristic_notice"]


def test_nonlinear_expectations_are_numerical_not_row_count_only() -> None:
    rows = [dict(row) for row in NONLINEAR]
    rows[0]["rmse"] = 1.0
    check = cli._paper_checks({"nonlinear_encoding_error": rows})["nonlinear"]
    assert check["passed"] is False
    assert check["status"] == "failed"
