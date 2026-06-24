# Reproducibility

The original paper examples use D-Wave `neal` simulated annealing. These runs
are stochastic and depend on package versions, seed handling, and hardware
runtime. They should be treated as regression checks with tolerances, not
certificates of global optimality.

Searchable audit anchors came from `papers/clean-manuscript-R3.pdf`.
`papers/galley-proof.pdf` did not produce searchable text with `pdftotext` in
this environment.

## Solver Settings

The manuscript numerical section states:

- solver: D-Wave `neal`
- `num_reads = 200`
- `num_sweeps = 2500`
- `seed = 11`

Legacy artifacts are not fully consistent with that global statement:

- `notebooks/ex1.ipynb`: 200 reads, 2500 sweeps, seed 11.
- `notebooks/ex3.ipynb`: 200 reads, 2500 sweeps, seed 13.
- `notebooks/ex4.ipynb`: 300 reads, 3000 sweeps, seed 7.
- zoom scripts: commonly 300 reads, 4000 sweeps, seed 13.

These mismatches are documented in `AUDIT.md`.

## Deterministic Checks

Use exact enumeration for:

- SBE grid representability and bounds.
- Integer encoding coverage.
- QUBO algebra expansion on small bit sets.
- Rosenberg quadratization on small pseudo-Boolean polynomials.
- Example 1 at small `J`.
- Example 2 at `J=2`, where the exact optimum is representable.

## Stochastic Checks

Use fixed seeds and tolerances for:

- high-precision Example 1 with `neal`;
- Alan bit-growth and zoom;
- Alan penalty sensitivity.

Runtime values are hardware-dependent and should not be strict regression
targets.

## Known Difference

Example 2 at `J=1` has multiple grid solutions with the same objective under
exact enumeration. The paper table reports `x=0.4, s=0.6`; exact enumeration in
this package may return `x=0.3, s=0.7` unless a tie-breaker or stochastic solver
selects the other representative. Both have objective `2.5e-3` and zero
constraint residual.
