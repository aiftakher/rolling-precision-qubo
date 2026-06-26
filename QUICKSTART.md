# Quickstart

This guide shows the normal user flow:

```text
problem file -> encoded QUBO -> solver -> decoded solution
```

The package accepts bounded binary, integer, continuous, and slack variables.
Continuous variables are placed on a finite grid; `digits` controls how fine
that grid is. For example, `digits: 1` means one decimal digit of precision.

## 1. Inspect One Variable Encoding

Integer variables default to bounded binary encoding:

```bash
rpqubo encode-variable --name i --type integer --lower 0 --upper 3
```

Continuous variables need bounds and a digit precision:

```bash
rpqubo encode-variable --name x --type continuous --lower 0 --upper 3 --digits 1 --encoding sbe
```

The output shows the binary variables and weights used to represent the original
variable.

## 2. Define A Small MIQP

The file `examples/tiny_miqp.json` describes this problem:

```text
minimize    (x - 1.5)^2 + 0.2 y
subject to  x + y >= 2

0 <= x <= 3, x continuous on a one-decimal grid
0 <= y <= 2, y integer
```

The expanded objective is:

```text
x^2 - 3x + 2.25 + 0.2y
```

The same problem file is:

```json
{
  "name": "tiny_miqp",
  "variables": [
    {
      "name": "x",
      "type": "continuous",
      "lower": 0.0,
      "upper": 3.0,
      "digits": 1,
      "encoding": "sbe"
    },
    {
      "name": "y",
      "type": "integer",
      "lower": 0,
      "upper": 2
    }
  ],
  "objective": {
    "constant": 2.25,
    "linear": {
      "x": -3.0,
      "y": 0.2
    },
    "quadratic": [
      {
        "vars": ["x", "x"],
        "coef": 1.0
      }
    ]
  },
  "constraints": [
    {
      "name": "demand",
      "linear": {
        "x": 1.0,
        "y": 1.0
      },
      "sense": ">=",
      "rhs": 2.0,
      "penalty": 50.0,
      "slack_digits": 1
    }
  ]
}
```

The important fields are:

- `variables`: original decision variables before binary encoding.
- `objective.constant`: constant term in the objective.
- `objective.linear`: coefficients such as `-3.0 * x`.
- `objective.quadratic`: coefficients such as `1.0 * x * x`.
- `constraints`: linear equalities or inequalities.
- `penalty`: strength used to discourage constraint violation.
- `slack_digits`: grid precision for the generated slack variable on an
  inequality.

## 3. Build The QUBO

Build the structured problem:

```bash
rpqubo build-qubo examples/tiny_miqp.json --output tiny_miqp.model --format model
```

This command reads the JSON problem, validates it, creates binary encodings,
adds the quadratic objective, turns the inequality into a squared penalty, and
writes a reloadable model bundle.

The `.model` format is usually the best choice while learning because it stores
both the QUBO and the information needed to decode the solution.

## 4. Solve And Decode

For this tiny model, use exact enumeration:

```bash
rpqubo solve tiny_miqp.model --solver exact
```

The output contains:

- `sample`: the binary QUBO solution.
- `decoded`: the solution in original variables, such as `x`, `y`, and the
  generated slack variable.
- `energy`: the QUBO energy.
- `objective`: the original objective value after decoding.
- `feasibility`: the largest constraint residual after decoding.

For larger models, exact enumeration becomes too expensive. Use a heuristic
solver instead:

```bash
rpqubo solve tiny_miqp.model --solver neal --num-reads 200 --sweeps 2500 --seed 123
```

Heuristic solvers can be useful, but they do not certify global optimality.

## 5. What Happens Inside

The command-line entry point is configured in `pyproject.toml`:

```text
rpqubo = "rpqubo.cli:main"
```

When you run `rpqubo build-qubo ...`, the main execution order is:

```text
src/rpqubo/cli.py
  -> src/rpqubo/io.py
  -> src/rpqubo/variables.py
  -> src/rpqubo/builders.py
  -> src/rpqubo/encodings.py
  -> src/rpqubo/qubo.py
```

In plain language:

1. The CLI receives your command.
2. The JSON file is read.
3. The problem is converted into Python objects.
4. The problem is checked for valid bounds, names, coefficients, and references.
5. Each original variable is represented by binary bits.
6. The objective is rewritten in those binary bits.
7. Each constraint is added as a quadratic penalty.
8. The finished QUBO is saved.

When you run `rpqubo solve ...`, the main execution order is:

```text
src/rpqubo/cli.py
  -> src/rpqubo/io.py
  -> src/rpqubo/solvers.py
  -> src/rpqubo/builders.py (BuildResult.decode_sample)
```

The solver finds a binary sample, then the model bundle decodes that binary
sample back into the original variables.

## 6. Paper Outputs

Regenerate the paper outputs:

```bash
rpqubo reproduce-paper --example all --output-dir outputs/paper
```

The generated report records dependency versions, git state, solver settings,
reference-check statuses, and the files written during reproduction.
