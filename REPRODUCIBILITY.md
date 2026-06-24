# Reproducibility

Paper outputs are regenerated with:

```bash
rpqubo reproduce-paper --example all --output-dir outputs/paper
```

The command writes CSV files plus `reproducibility_report.json`, including git
state, Python and dependency versions, and the configuration used by each table.

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
- The manuscript prose says Alan rolling penalties are all 500. The Table 6
  legacy code uses `{"e1": 200, "e2": 200, "link": 300, "card": 200}` with
  dynamic width scaling. Both modes are exposed; `paper_table6_reference`
  reproduces the table and `manuscript_uniform_500` follows the prose.
- Alan Table 5 uses SBE cardinality slack. Tables 6 and 9 use the two-bit
  integer cardinality slack `sc = sc0 + 2*sc1`.
- Nonlinear cumulative-unary encoding is exact only for `J=1` on the one-digit
  grid. For `J>1`, the power representation is an approximation with recorded
  error.
- Heuristic annealing does not certify global optimality, and runtime is not a
  regression-tested quantity.
- In this checkout, the current local interpreter is Python 3.9.6 while package
  metadata targets Python 3.10+. See `AUDIT.md` for the exact gate limitation.
