from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _run(args: list[str], cwd: Path, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(cwd / "src")
    return subprocess.run(
        [sys.executable, "-m", "rpqubo.cli", *args],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=check,
    )


def test_cli_errors_and_model_bundle(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]

    encoded = _run(
        ["encode-variable", "--name", "i", "--type", "integer", "--lower", "0", "--upper", "3"],
        root,
    )
    assert '"integer_weights"' in encoded.stdout

    bad_problem = tmp_path / "bad.json"
    bad_problem.write_text(json.dumps({"variables": [], "objective": {"linear": {"missing": 1}}}))
    failed = _run(["solve", str(bad_problem)], root, check=False)
    assert failed.returncode != 0

    problem = tmp_path / "problem.json"
    problem.write_text(
        json.dumps(
            {
                "variables": [
                    {"name": "x", "type": "continuous", "lower": 0, "upper": 1, "digits": 1}
                ],
                "objective": {
                    "constant": 0.1225,
                    "linear": {"x": -0.7},
                    "quadratic": [{"vars": ["x", "x"], "coef": 1.0}],
                },
            }
        )
    )

    invalid_reads = _run(
        ["solve", str(problem), "--solver", "random", "--num-reads", "0"],
        root,
        check=False,
    )
    assert invalid_reads.returncode != 0
    invalid_sweeps = _run(
        ["solve", str(problem), "--solver", "neal", "--sweeps", "0"],
        root,
        check=False,
    )
    assert invalid_sweeps.returncode != 0
    invalid_rescale = _run(
        [
            "build-qubo",
            str(problem),
            "--output",
            str(tmp_path / "q.json"),
            "--rescale",
            "0",
        ],
        root,
        check=False,
    )
    assert invalid_rescale.returncode != 0

    bundle = tmp_path / "model.model"
    _run(["build-qubo", str(problem), "--output", str(bundle), "--format", "model"], root)
    solved = _run(["solve", str(bundle), "--solver", "exact"], root)
    assert '"decoded"' in solved.stdout


def test_reproduce_paper_all_cli(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = tmp_path / "paper"
    result = _run(["reproduce-paper", "--example", "all", "--output-dir", str(out_dir)], root)
    assert "reproducibility_report.json" in result.stdout
    expected = {
        "ex1_unconstrained_bit_growth.csv",
        "ex1_unconstrained_zoom.csv",
        "ex2_miqp_bit_growth.csv",
        "ex2_miqp_zoom.csv",
        "ex3_alan_bit_growth.csv",
        "ex3_alan_zoom.csv",
        "nonlinear_encoding_error.csv",
        "alan_penalty_sensitivity.csv",
        "reproducibility_report.json",
    }
    assert expected <= {path.name for path in out_dir.iterdir()}
