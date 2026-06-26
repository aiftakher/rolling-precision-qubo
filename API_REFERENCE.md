# API Reference

Implemented scope for v0.1:

- bounded binary, integer, continuous, and slack variables;
- bounded quadratic objectives;
- linear equality and inequality constraints through quadratic penalties;
- SBE, unary, cumulative-unary, and bounded integer encodings;
- rolling precision and anchored zoom helpers;
- exact, random, and `neal` solvers;
- JSON/CSV/NPZ/dimod/BQM QUBO export plus reloadable model bundles.

Key entry points:

- `rpqubo.variables.Problem.validate()` validates structured inputs before build.
  `ContinuousVar` and `SlackVar` accept `ordering_penalty` only for
  `encoding="cumulative_unary"`; `LinearConstraint` accepts
  `slack_ordering_penalty` for cumulative-unary generated slacks. The builder
  rejects cumulative-unary encodings without a finite, positive strength.
- `rpqubo.builders.build_qubo()` returns `BuildResult` with `qubo`,
  `unscaled_qubo`, `rescale_factor`, encodings, and constraint diagnostics.
  Cumulative-unary ordering penalties are inserted exactly once, before
  objective and constraint penalties.
- `rpqubo.solvers.solve_qubo()` supports `exact`, `random`, and `neal`.
- `rpqubo.io.export_model()` and `rpqubo.io.load_model()` preserve encodings so
  reloaded models can be solved and decoded without rebuilding the source
  problem.
- `rpqubo.examples.reproduce_*()` regenerates paper-reference examples by
  constructing and solving QUBOs.
- `rpqubo.paper_expectations.compare_table9_row()` classifies generated Table 9
  rows as exact reference matches, documented stochastic alternatives, or
  failures against `data/alan_penalty_sensitivity.csv`.

The package does not claim arbitrary MIQCQP or geometric-program support.
The multi-digit nonlinear cumulative-unary surrogate is approximate; use
`rpqubo.encodings.nonlinear_error_table()` to inspect its error envelope.
