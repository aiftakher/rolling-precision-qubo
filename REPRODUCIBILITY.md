# Reproducibility

Paper outputs are regenerated with:

```bash
rpqubo reproduce-paper --example all --output-dir outputs/paper --strict
```

The command writes CSV files plus `reproducibility_report.json`, including git
state, Python and dependency versions, and the configuration used by each table.
`--strict` still writes all outputs, then exits nonzero if the report's
`overall_passed` field is false.

`outputs/paper/` is generated output and is ignored by Git except for
`outputs/paper/.gitkeep`. CI regenerates the directory in the Python 3.10
reference job and uploads it as a workflow artifact. Accepted reference data
remains under `data/`, including the immutable Table 9 file
`data/alan_penalty_sensitivity.csv`.

Reference solver settings:

- Example 1 bit growth: `neal`, `num_reads=200`, `sweeps=2500`, `seed=11`.
- Example 1 zoom: `neal`, `num_reads=300`, `sweeps=4000`, `seed=11`.
- Example 2 bit growth: `neal`, `num_reads=200`, `sweeps=2500`, `seed=13`.
- Example 2 zoom: `neal`, `num_reads=300`, `sweeps=4000`, `seed=13`, dynamic
  penalties, target max coefficient 10.
- Alan bit growth: `neal`, `num_reads=300`, `sweeps=3000`, `seed=7`, SBE
  cardinality slack.
- Alan Table 6 zoom: `neal`, `num_reads=300`, `sweeps=4000`, `seed=13`, integer
  cardinality slack, dynamic penalties, target max coefficient 10.
- Alan Table 9 sensitivity: `neal`, `num_reads=300`, `sweeps=4000`, `seed=13`,
  fixed penalty per run, integer cardinality slack, zoom enabled.

Important notes:

- Seeded `neal` output is sensitive to BQM variable and interaction insertion
  order. The `paper_reference` layer preserves notebook/legacy construction
  order separately from the public builder.
- Reproducibility reports separate `exact_reference_match` from
  `within_documented_stochastic_tolerance`. A scientifically acceptable
  stochastic Table 9 row is not reported as an exact match unless it is
  numerically identical to the accepted CSV within strict tolerances.
- The manuscript prose says Alan rolling penalties are all 500. The Table 6
  legacy code uses `{"e1": 200, "e2": 200, "link": 300, "card": 200}` with
  dynamic width scaling. Both modes are exposed; `paper_table6_reference`
  reproduces the table and `manuscript_uniform_500` follows the prose.
- Alan Table 5 uses SBE cardinality slack. Tables 6 and 9 use the two-bit
  integer cardinality slack `sc = sc0 + 2*sc1`.
- Nonlinear cumulative-unary encoding is exact only for `J=1` on the one-digit
  grid. For `J>1`, the power representation is an approximation with recorded
  error.
- Cumulative-unary ordering is enforced only when the model supplies a finite,
  positive ordering penalty. The builder rejects cumulative-unary variables or
  generated slacks that omit it.
- Heuristic annealing does not certify global optimality, and runtime is not a
  regression-tested quantity.
- The normal Python matrix runs fast API and mathematical checks with slow tests
  excluded. The Python 3.10 reference job installs with
  `requirements-lock.txt`, runs slow seeded-neal checks, regenerates
  `outputs/paper`, and runs `rpqubo reproduce-paper --strict`.
