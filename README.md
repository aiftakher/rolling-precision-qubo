# rolling-precision-qubo

`rolling-precision-qubo` is the package version of the research code
for:

Solving Mixed-Integer Problems as QUBO: Encodings, Reformulations, and Rolling-precision Algorithm

📄 **Paper:** [Solving Mixed-Integer Problems as QUBO: Encodings, Reformulations, and Rolling-precision Algorithm](https://doi.org/10.1016/j.compchemeng.2026.109757)


It QUBO models are binary quadratic objectives. This package turns bounded continuous, integer, binary, and slack
variables into binary/QUBO form, supports exact enumeration for small problem instances, and simulated annealing through D-Wave `neal`. It does not implement or claim global optimality.

## Install

```bash
python -m pip install -e ".[dev]"
```

Python >=3.10.

## Quickstart

Encode a continuous variable:

```bash
rpqubo encode-variable --name x --type continuous --lower 0 --upper 1 --digits 2 --encoding sbe
```

Build and solve an example mixed-integer quadratic problem from a JSON file:

```bash
rpqubo build-qubo examples/tiny_miqp.json --output tiny_miqp.model --format model
rpqubo solve tiny_miqp.model --solver exact
```

The `.model` bundle preserves the original problem, the generated QUBO, the
binary encodings, and the metadata needed to decode the binary solution back to
the original variables. See [QUICKSTART.md](QUICKSTART.md) for the full
problem-file format and the step-by-step build/solve flow.

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
rpqubo reproduce-paper --example all --output-dir outputs/paper
```

`outputs/paper/` is generated output. CI regenerates it and uploads it as an
artifact; accepted reference data lives under `data/`.

## Features

- SBE decimal encoding with weights `(1, 2, 3, 3)` plus tail bit.
- Digit-sum unary and cumulative-unary encodings. Cumulative-unary variables
  require an explicit `ordering_penalty`.
- Bounded integer encoding for non-power-of-two ranges.
- Explicit slack-variable reformulation for inequalities.
- Sparse QUBO object with linear terms, quadratic terms, offset, and groups.
- JSON, CSV/COO, NumPy NPZ, and dimod BQM export.
- Reloadable model bundles with QUBO, encodings, constraints, and metadata.
- Exact, random, and `neal` solver adapters.
- Reproducible paper examples through `rpqubo reproduce-paper`.

## Repository Layout

- `src/rpqubo/`: package implementation.
- `tests/`: pytest suite.
- `legacy/` and `notebooks/`: preliminary research codes.
- `outputs/paper/`: generated CSV/JSON outputs.
- `examples/tiny_miqp.json`: minimal user-defined MIQP example for the CLI.

## Reproducibility Notes

- `neal` is the supported heuristic annealing backend; exact enumeration is for
  small problem instances.
- Heuristic annealing does not certify global optimality.
- The one-digit nonlinear cumulative-unary surrogate is exact on its grid.
  Multi-digit nonlinear surrogates are approximate and reported with error
  metrics.

## Citation

Please cite our work if you use this code in your research.

```bibtex
@article{iftakher2026quantum,
  title={Solving Mixed-Integer Problems as QUBO: Encodings, Reformulations, and Rolling-precision Algorithm},
  author={Iftakher, Ashfaq and Turkay, Metin and Hasan, MM Faruque},
  journal={Computers \& Chemical Engineering},
  year={2026},
  publisher={Elsevier},
  DOI = {\url{https://doi.org/10.1016/j.compchemeng.2026.109757}},
}
