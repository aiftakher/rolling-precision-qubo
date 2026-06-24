# rolling-precision-qubo

`rolling-precision-qubo` turns bounded continuous, integer, binary, and slack
variables into binary/QUBO form. It is the package version of the research code
for:

> Solving Mixed-Integer Problems as QUBO: Encodings, Reformulations, and
> Rolling-precision Algorithm

QUBO models are binary quadratic objectives. This package supports exact
enumeration for tiny verification cases, a random debugging baseline, and
simulated annealing through D-Wave `neal`. It does not implement or advertise
D-Wave hardware or hybrid execution. Heuristic annealing does not certify global
optimality.

## Install

```bash
python -m pip install -e ".[dev]"
```

Python 3.10 or newer is the supported package target.

## Quickstart

Encode a continuous variable:

```bash
rpqubo encode-variable --name x --type continuous --lower 0 --upper 1 --digits 2 --encoding sbe
```

Build and solve a tiny QUBO in Python:

```python
from rpqubo import ContinuousVar, build_qubo, solve_qubo
from rpqubo.variables import Problem, QuadraticObjective

problem = Problem(
    variables=[ContinuousVar("x", 0.0, 1.0, digits=1)],
    objective=QuadraticObjective(
        constant=0.35**2,
        linear={"x": -2 * 0.35},
        quadratic={("x", "x"): 1.0},
    ),
)

build = build_qubo(problem)
result = solve_qubo(build.qubo, solver="exact")
print(build.decode_sample(result.sample))
```

Reproduce paper outputs:

```bash
rpqubo reproduce-paper --example ex1 --output-dir outputs/paper
```

## Features

- SBE decimal encoding with weights `(1, 2, 3, 3)` plus tail bit.
- Digit-sum unary and cumulative-unary encodings.
- Safe bounded integer encoding for non-power-of-two ranges.
- Explicit slack-variable reformulation for inequalities.
- Sparse QUBO object with linear terms, quadratic terms, offset, and groups.
- JSON, CSV/COO, NumPy NPZ, and dimod BQM export.
- Reloadable model bundles with QUBO, encodings, constraints, and metadata.
- Exact, random, and `neal` solver adapters.
- Reproducible paper examples through `rpqubo reproduce-paper`.

## Repository Layout

- `src/rpqubo/`: package implementation.
- `tests/`: pytest suite.
- `legacy/` and `notebooks/`: original research artifacts retained for traceability.
- `outputs/paper/`: generated reference CSV/JSON outputs.

## Citation

Use `CITATION.cff` and cite the paper above when publishing results derived from
this package.
