# rolling-precision-qubo

`rolling-precision-qubo` turns bounded continuous, integer, binary, and slack
variables into binary/QUBO form. It is the package version of the research code
for:

> Solving Mixed-Integer Problems as QUBO: Encodings, Reformulations, and
> Rolling-precision Algorithm

QUBO models are binary quadratic objectives. They can be solved by classical
exact enumeration for tiny cases, simulated annealing through D-Wave `neal`, or
optional D-Wave hardware/hybrid backends when credentials are configured.
Heuristic annealing does not certify global optimality.

## Install

```bash
python -m pip install -e .
```

Optional D-Wave hardware support:

```bash
python -m pip install -e ".[dwave]"
```

D-Wave credentials are never required for normal installation. Hardware access
uses standard D-Wave environment variables or config files.

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
- Exact, random, and `neal` solver adapters.
- Reproducible paper example scripts under `examples/paper/`.

## Repository Layout

- `src/rpqubo/`: package implementation.
- `examples/paper/`: scripts that reproduce paper examples through the package.
- `tests/`: pytest suite.
- `legacy/` and `notebooks/`: original research artifacts retained for traceability.
- `papers/`: manuscript/proof PDFs.

## Citation

Use `CITATION.cff` and cite the paper above when publishing results derived from
this package.
