# Repair Audit

Date: 2026-06-24

## Environment

- Working directory: `/Users/ashfaqiftakher/Downloads/2025-QUBO/Nov30-code`
- Local shell: `zsh`
- `python`: not on PATH in this shell.
- `python3`: Python 3.9.6, Clang 21.0.0.
- Installed reference packages:
  - dimod 0.12.21
  - dwave-neal 0.6.0
  - dwave-samplers 1.6.0
  - numpy 1.26.4
  - PyYAML 6.0.2
  - pytest 8.4.2
  - ruff 0.15.19
  - mypy 1.19.1
  - build 1.4.4

## Baseline Command Results

Exact requested baseline commands:

```text
$ python -m pip install -e ".[dev]"
zsh:1: command not found: python

$ pytest -q
zsh:1: command not found: pytest

$ ruff check .
zsh:1: command not found: ruff

$ mypy src/rpqubo
zsh:1: command not found: mypy

$ python -m build
zsh:1: command not found: python
```

Practical `python3 -m ...` baseline after allowing dependency installation:

```text
$ python3 -m pytest -q
FAILED tests/test_cli.py::test_cli_commands_round_trip
ImportError: cannot import name 'reproduce_alan_penalty_sensitivity' from 'rpqubo.examples'

$ python3 -m ruff check .
3 import-order errors in builders.py, examples.py, and qubo.py

$ python3 -m mypy src/rpqubo
5 errors:
- qubo.py tuple key type inference
- examples.py unexpected penalties keyword for solve_alan_at_box
- cli.py missing reproduce_alan_penalty_sensitivity

$ python3 -m build
No module named build
```

## Implementation-To-Paper Map

- Tables 1 and 2: `rpqubo.examples.reproduce_example1_bit_growth()` and
  `reproduce_example1_zoom()`.
- Tables 3 and 4: generic Example 2 bit growth plus
  `paper_reference.rolling_example2_zoom_reference()` for the legacy constant-size
  zoom trajectory.
- Table 5: `paper_reference.solve_alan_bit_growth_reference_at_precision()` using
  notebook labels `b6..b9`, `s6..s9`, and SBE cardinality slack.
- Table 6: `paper_reference.rolling_alan_zoom_reference()` using integer
  cardinality slack `sc0 + 2*sc1` and legacy dynamic penalties.
- Table 9: `paper_reference.alan_penalty_sensitivity_reference()`, which runs the
  constant-size Alan zoom workflow for each lambda.
- Tables 7 and 8: `encodings.nonlinear_error_table()` plus cumulative-unary order
  penalty helpers.

## Manuscript/Code Discrepancies

- Alan Table 6 prose describes uniform penalties of 500, but the legacy settings
  that reproduce the checked-in Table 6 trajectory are
  `{"e1": 200, "e2": 200, "link": 300, "card": 200}` with dynamic width scaling.
- Alan Table 5 uses SBE cardinality slack; Alan Tables 6 and 9 use two-bit integer
  cardinality slack.
- The local Python interpreter is 3.9.6, while repaired package metadata declares
  Python `>=3.10` as requested. Exact install/build gates requiring `python` or
  Python 3.10 cannot pass in this local shell.
- Table 9 is regenerated from the legacy workflow. The local dependency set
  reproduces the checked-in legacy script behavior; a few rounded values differ
  slightly from the prompt text for high penalties because seeded `neal` output is
  version/order sensitive.

## Final Results After Repair

Commands run successfully with local `python3`:

```text
$ python3 -m ruff check .
All checks passed!

$ python3 -m mypy src/rpqubo
Success: no issues found in 15 source files

$ python3 -m pytest -q
26 passed

$ python3 -m rpqubo.cli reproduce-paper --example all --output-dir outputs/paper
wrote all requested CSV files and reproducibility_report.json

$ python3 -m build
Successfully built rolling_precision_qubo-0.1.0.tar.gz and
rolling_precision_qubo-0.1.0-py3-none-any.whl
```

Exact final gates blocked by the local shell/environment:

```text
$ python -m pip install -e ".[dev]"
zsh:1: command not found: python

$ ruff check .
zsh:1: command not found: ruff

$ mypy src/rpqubo
zsh:1: command not found: mypy

$ pytest -q
zsh:1: command not found: pytest

$ rpqubo reproduce-paper --example all --output-dir outputs/paper
zsh:1: command not found: rpqubo

$ python -m build
zsh:1: command not found: python
```

The practical Python 3.9 install gate also fails after the repair because
`pyproject.toml` now declares `requires-python = ">=3.10"`:

```text
$ python3 -m pip install -e ".[dev]"
ERROR: Package 'rolling-precision-qubo' requires a different Python: 3.9.6 not in '>=3.10'
```
