# Audit and Repository Plan

This audit is based on `clean-manuscript-R3.pdf`, the legacy Python scripts,
the notebooks, and the shipped CSV files. `galley-proof.pdf` appears to be a
print/image PDF in this workspace: `pdftotext` produced no searchable text, so
the searchable audit anchors come from `clean-manuscript-R3.pdf`.

## Paper Workflow Summary

The paper proposes a mixed-integer-to-QUBO workflow:

1. Start from a bounded mixed-integer quadratic or quadratically constrained
   problem. All variables must have finite lower and upper bounds.
2. Encode integer variables with binary expansion. If an integer variable has
   range `R = U - L`, the paper chooses `B = ceil(log2(R + 1))` bits and writes
   `x = L + sum_b 2^b z_b`.
3. Scale bounded continuous variables to `[0, 1]`, then approximate them on a
   decimal grid.
4. Support two decimal encodings:
   - digit-sum unary encoding with `9J + 1` bits, including an endpoint bit;
   - digit-wise SBE encoding with weights `(1, 2, 3, 3)` plus an endpoint/tail
     bit, using `4J + 1` bits.
5. Convert each inequality to an equality with an explicit nonnegative slack:
   `a^T x <= b` becomes `a^T x + s = b`, with a finite slack upper bound.
6. Add squared residual penalties for equality constraints and slack
   reformulated inequalities. Penalty weights are heuristic and must be checked
   for feasibility and coefficient dynamic range.
7. For quadratic constraints or higher-order pseudo-Boolean terms, introduce
   auxiliary variables for products and enforce `w = z_p z_q` using the
   Rosenberg penalty `3w - 2z_p w - 2z_q w + z_p z_q`.
8. Build the binary quadratic model and solve it with a QUBO solver. The paper
   uses D-Wave `neal` simulated annealing for numerical examples, but notes that
   heuristic annealing does not certify global optimality.
9. Use rolling precision to avoid one large high-resolution QUBO:
   - sequential bit-growth increases selected decimal depths `J_i`;
   - constant-size zoom keeps `J` fixed and shrinks search boxes around
     incumbents.

For constrained zoom examples, the manuscript explicitly states a practical
lexicographic acceptance rule: improve feasibility first, then original
objective once both incumbent and candidate are feasible within tolerance.

## Artifact Map

| Paper item | Current artifact(s) | Notes |
| --- | --- | --- |
| Encoding definitions, slack reformulation, rolling algorithms | `clean-manuscript-R3.pdf`; generic prototype in `rolling_precision_QUBO.py` | The prototype only partially implements the paper workflow. |
| Table 1, Example 1 bit-growth | `ex1.ipynb` | Output matches manuscript table values and QUBO sizes. |
| Table 2, Example 1 zoom | `zoom-in.ipynb`, `zoom_alan_sensitivity.py` | Uses fixed `J=1`, zoom factor `rho=0.2`, rescaled BQM coefficients. |
| Table 3, Example 2 MIQP bit-growth | `ex3.ipynb` | Despite the file name, this is the manuscript Example 2. |
| Table 4, Example 2 MIQP zoom | `zoom-in.ipynb`, `zoom_alan_sensitivity.py` | Uses seed 13 and 300/4000 reads/sweeps in the zoom script. |
| Table 5, Alan bit-growth | `ex4.ipynb` | Uses all penalties 500, but solver settings differ from the manuscript global statement. |
| Table 6, Alan zoom | `ex4-zoom.ipynb`, `zoom-in.ipynb`, `zoom_alan_sensitivity.py` | Current zoom code uses dynamic penalties with base values 200/200/300/200, which conflicts with the manuscript text saying all Alan rolling experiments use 500. |
| Tables 7-8, nonlinear coefficients and surrogate errors | `nonlinear-encoding.ipynb` | Deterministic computations match the appendix values to rounding. |
| Table 9, Alan penalty sensitivity | `zoom_alan_sensitivity.py`, `alan_penalty_sensitivity.csv`, `alan_penalty_sensitivity_summary.csv` | Uses fixed common penalties, seed 13, 300 reads, 4000 sweeps. |
| Legacy constrained toy with two continuous variables | `ex2.ipynb` | Not the manuscript Example 2. It solves `min (x1-0.3)^2 + (x2-0.6)^2` with `x1+x2<=1`. Keep only as legacy/demo material. |
| Generic solver prototype | `rolling_precision_QUBO.py` | Contains symbolic parsing, SBE, Rosenberg reduction, and rolling bit-growth, but is not production-ready. |

## Duplicated Code

The following logic is copied across notebooks and scripts:

- SBE constants and affine encoding: `DEC_WEIGHTS = (1, 2, 3, 3)`,
  `sbe_affine_bits`/`sbe_affine`.
- Decoding affine encodings from binary samples.
- QUBO algebra helpers: `add_linear`, `add_quad`,
  `add_square_of_affine`, `add_product_of_affines`.
- BQM construction and QUBO size statistics.
- `SAOptions` dataclasses or equivalent solver option bundles.
- Rolling/backtracking loops for bit-growth and zoom.
- Alan model construction, residual computation, and reporting.
- Nonlinear surrogate coefficient and error computations.

The package should centralize these into reusable modules and make the paper
scripts thin wrappers around the public API.

## Implementation Risks and Bugs

### Generic inequality handling is not polynomial

`rolling_precision_QUBO.py` handles `<=` and `>=` constraints with
`sympy.Max(g, 0)**2` and `sympy.Min(g, 0)**2`. This is not a polynomial QUBO
formulation. A direct runtime check fails during term extraction with an
unexpected `Max(...)**2` factor. The paper uses explicit slack-variable
reformulation, so the package should make slack reformulation the default and
only allow non-polynomial penalties through an explicit opt-in path.

### Mutable/default dataclass instances

Several functions/classes use instantiated dataclasses as defaults, for example
`SAOptions()`, `PenaltyConfig()`, and `ZoomConfig()`. This shares mutable state
on older Python versions and is rejected by newer dataclass checks in common
setups. Use `default_factory` for dataclass fields and `None` plus local
construction for function parameters.

### Integer encoding can silently exceed upper bounds

The paper's binary integer encoding is exact when `R + 1` is a power of two,
but for ranges such as `[0, 5]` three bits represent values 0 through 7. Current
code has no safe integer encoding layer. The package must either:

- use a bounded encoding that removes invalid states;
- add a validity penalty for out-of-domain states;
- or decode with domain filtering/reporting and never silently accept invalid
  integer assignments.

The public API should default to `strict_bounds=True`.

### Rosenberg quadratization needs adaptive strength and tests

The current Rosenberg implementation uses a fixed default strength of `1e3`.
That is not generally sufficient. For example, a polynomial containing
`10000*a*b*c - 4000*a - 4000*b - 4000*c` can produce a lower invalid-ancilla
QUBO energy than the original pseudo-Boolean minimum when the strength is 1000.
The package should choose or require a strength tied to coefficient magnitudes
and should brute-force validate small reductions in tests.

### Solver acceptance criteria are inconsistent

Current code uses multiple acceptance notions:

- `rolling_precision_QUBO.py` accepts bit-growth moves on penalized QUBO energy.
- `ex3.ipynb` and `ex4.ipynb` use penalized energy/merit for bit-growth.
- `zoom-in.ipynb` and `zoom_alan_sensitivity.py` use feasibility-first
  lexicographic acceptance.
- Some reporting evaluates original objective and feasibility residuals after
  selecting by penalized energy.

The package should expose the acceptance criterion explicitly, record it in
solver metadata, and make paper examples use the criterion described in the
paper for that experiment.

### QUBO rescaling is undocumented

Zoom scripts rescale QUBO coefficients to a target max absolute value. Positive
global scaling preserves minimizers, but it changes reported BQM energies. This
must be documented and tracked with a `scale_factor`. Coefficient summaries
should distinguish unscaled model coefficients from scaled solver coefficients.

### Slack bounds are ad hoc

The manuscript gives a bound computation for linear inequality slacks. Legacy
examples manually use `[0, 1]` or `[0, 3]`. The package builder should compute
conservative slack bounds from variable bounds when possible, while allowing
users to override them.

### SBE is grid-complete but nonunique

Brute force confirms the SBE `(1, 2, 3, 3)` plus tail bit represents exactly
the decimal grid `{k/10^J : k = 0, ..., 10^J}` for small `J`, with no decoded
values outside `[0, 1]`. However, many binary patterns decode to the same digit
value. Tests should validate representability, bounds, and decoding behavior,
but users should not assume a unique binary representation.

### Nonlinear cumulative-unary constraints are missing from general code

The nonlinear notebook computes coefficients and errors, but there is no reusable
encoding object and no monotonicity penalty implementation for cumulative unary
digits. The package should implement both one-digit exact cumulative-unary
coefficients and multi-digit surrogate error reporting.

### Reproducibility gaps

- The workspace is not currently a Git repository.
- There is no package metadata, CLI, test suite, CI, license, citation file, or
  reproducibility document.
- Legacy notebooks contain hidden output and duplicated logic.
- `galley-proof.pdf` is not searchable by normal PDF text extraction here.
- Runtime measurements in tables are hardware-dependent and should not be used
  as strict regression checks.

## Manuscript and Code Mismatches

| Topic | Manuscript statement | Current code/output |
| --- | --- | --- |
| Global solver settings | Numerical section states `neal` with `num_reads=200`, `num_sweeps=2500`, `seed=11`. | `ex1.ipynb` matches. `ex3.ipynb` uses seed 13. `ex4.ipynb` uses 300 reads, 3000 sweeps, seed 7. Zoom scripts often use 300 reads, 4000 sweeps, seed 13. |
| Alan rolling penalties | Manuscript text says `lambda1=lambda2=lambda3=lambda4=500`. | Alan bit-growth uses 500. Alan zoom code that reproduces Table 6 uses dynamic penalties with base 200/200/300/200. Penalty sensitivity uses common fixed lambdas. |
| Generic inequality formulation | Manuscript converts inequalities to slack equalities. | `rolling_precision_QUBO.py` uses `Max/Min`, which is not QUBO and fails extraction. |
| Example numbering | Paper Example 2 is MIQP with one continuous and one binary variable. | `ex2.ipynb` is a different two-continuous-variable constrained toy; `ex3.ipynb` is the paper Example 2. |
| Acceptance rule | Algorithms use observed objective, and constrained zoom note uses feasibility-first then objective. | Bit-growth prototypes often accept by penalized energy; zoom prototypes use lexicographic feasibility/objective. |
| Coefficient scaling | Paper discusses dynamic range and hardware scaling qualitatively. | Zoom scripts rescale BQM coefficients to target max abs 10, including offset; this is not consistently documented in outputs. |

## Deterministic vs Stochastic Results

Should be deterministic or exactly enumerable:

- Variable encoding grids and decoded values.
- QUBO algebra expansions.
- Nonlinear coefficient tables and surrogate error tables.
- Tiny Example 1 QUBOs at small `J`.
- Rosenberg reductions on small pseudo-Boolean polynomials.
- Small MIQP QUBO construction when solved by exact enumeration.

Stochastic and seed-dependent:

- All `neal` simulated annealing outputs.
- Alan bit-growth and zoom results.
- Penalty sensitivity rows when generated by `neal`.
- Runtime columns.

Regression tests should use exact enumeration for small cases and seeded
`neal` checks with tolerances for larger examples. If seeded outputs differ
under pinned dependency versions, `REPRODUCIBILITY.md` should record the
closest deterministic configuration and explain the difference.

## Proposed Repository Structure

```text
rolling-precision-qubo/
  pyproject.toml
  README.md
  INSTALL.md
  QUICKSTART.md
  API_REFERENCE.md
  REPRODUCIBILITY.md
  AUDIT.md
  CITATION.cff
  LICENSE
  requirements-lock.txt
  .pre-commit-config.yaml
  .github/workflows/ci.yml
  src/
    rpqubo/
      __init__.py
      variables.py
      encodings.py
      expressions.py
      qubo.py
      builders.py
      quadratization.py
      solvers.py
      rolling.py
      io.py
      metrics.py
      examples.py
      cli.py
  examples/
    README.md
    paper/
      ex1_unconstrained_bit_growth.py
      ex1_unconstrained_zoom.py
      ex2_miqp_bit_growth.py
      ex2_miqp_zoom.py
      ex3_alan_bit_growth.py
      ex3_alan_zoom.py
      nonlinear_encoding_error.py
      alan_penalty_sensitivity.py
  tests/
    test_encodings.py
    test_integer_encoding.py
    test_qubo_algebra.py
    test_quadratization.py
    test_solvers.py
    test_cli.py
    test_reproducibility.py
  legacy/
    rolling_precision_QUBO.py
    zoom_alan_sensitivity.py
  notebooks/
    ex1.ipynb
    ex2.ipynb
    ex3.ipynb
    ex4.ipynb
    ex4-zoom.ipynb
    zoom-in.ipynb
    nonlinear-encoding.ipynb
  data/
    alan_penalty_sensitivity.csv
    alan_penalty_sensitivity_summary.csv
  papers/
    clean-manuscript-R3.pdf
    galley-proof.pdf
```

## Implementation Direction

The refactor should proceed by implementing reusable primitives first:

1. `variables.py` and `encodings.py`: variable specs, affine encodings,
   integer strict-bounds handling, SBE, unary, cumulative unary, slack vars.
2. `qubo.py`: low-level QUBO object, algebraic expansion helpers, coefficient
   stats, rescaling with metadata, dimod conversion.
3. `builders.py`: problem spec parsing, slack reformulation, penalty assembly.
4. `quadratization.py`: Rosenberg reduction with strength policy and ancilla
   validation helpers.
5. `solvers.py`: exact, random, neal, and optional D-Wave backends.
6. `rolling.py`: bit-growth and zoom algorithms with explicit acceptance
   criteria and complete metadata.
7. `io.py` and `cli.py`: JSON/CSV/NPZ/dimod export/import and command-line
   commands.
8. `examples.py` plus `examples/paper/*`: canonical reproducibility scripts
   that import the package and do not duplicate QUBO logic.
9. Tests and documentation.

Legacy files should be moved, not deleted, so the accepted-paper code remains
traceable while the package stops depending on notebook-only logic.
