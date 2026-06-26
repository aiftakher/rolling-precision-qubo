from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def run_cli(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(cwd / "src")
    return subprocess.run(
        [sys.executable, "-m", "rpqubo.cli", *args],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )


def test_cli_commands_round_trip(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    enc = run_cli(
        ["encode-variable", "--name", "x", "--type", "continuous", "--digits", "1"],
        root,
    )
    assert "binary_variables" in enc.stdout
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
    qubo = tmp_path / "qubo.json"
    run_cli(["build-qubo", str(problem), "--output", str(qubo)], root)
    inspect = run_cli(["inspect-qubo", str(qubo)], root)
    assert "n_variables" in inspect.stdout
    solved = run_cli(["solve", str(qubo), "--solver", "exact"], root)
    assert "sample" in solved.stdout
