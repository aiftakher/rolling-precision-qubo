# API Reference

Implemented scope for v0.1:

- bounded binary, integer, continuous, and slack variables;
- bounded quadratic objectives;
- linear equality and inequality constraints through quadratic penalties;
- SBE, unary, cumulative-unary, and bounded integer encodings;
- rolling precision and anchored zoom helpers;
- exact, random, and `neal` solvers;
- JSON/CSV/NPZ/dimod QUBO export plus reloadable model bundles.

Key entry points:

- `rpqubo.variables.Problem.validate()` validates structured inputs before build.
- `rpqubo.builders.build_qubo()` returns `BuildResult` with `qubo`,
  `unscaled_qubo`, `rescale_factor`, encodings, and constraint diagnostics.
- `rpqubo.solvers.solve_qubo()` supports `exact`, `random`, and `neal`.
- `rpqubo.io.export_model()` and `rpqubo.io.load_model()` preserve encodings so
  reloaded models can be solved and decoded without rebuilding the source
  problem.
- `rpqubo.examples.reproduce_*()` regenerates paper-reference examples by
  constructing and solving QUBOs.

The package does not claim arbitrary MIQCQP or geometric-program support.
