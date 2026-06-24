# API Reference

Main imports:

```python
from rpqubo import (
    BinaryVar,
    ContinuousVar,
    IntegerVar,
    SlackVar,
    build_qubo,
    encode_variable,
    export_qubo,
    load_qubo,
    solve_qubo,
)
```

Variable specs:

- `ContinuousVar(name, lower, upper, digits, encoding="sbe")`
- `IntegerVar(name, lower, upper, encoding="binary", strict_bounds=True)`
- `BinaryVar(name)`
- `SlackVar(name, lower, upper, digits, encoding="sbe")`

QUBO build:

- `build_qubo(problem, rescale=None)` returns a `BuildResult`.
- `BuildResult.qubo` contains `linear`, `quadratic`, `offset`, and
  `variable_groups`.
- `BuildResult.decode_sample(sample)` maps binary samples back to original
  variables.
- `BuildResult.objective_value(decoded)` evaluates the original objective.
- `BuildResult.feasibility(decoded)` returns max residual/violation.

Solvers:

- `solve_qubo(qubo, solver="exact")`
- `solve_qubo(qubo, solver="neal", num_reads=200, sweeps=2500, seed=11)`
- `solve_qubo(qubo, solver="random", num_reads=1000, seed=1)`

Exports:

- `export_qubo(qubo, "model.json")`
- `export_qubo(qubo, "model.csv", fmt="csv")`
- `export_qubo(qubo, "model.npz", fmt="npz")`
