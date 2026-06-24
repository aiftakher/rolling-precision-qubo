# Quickstart

Encode a variable:

```bash
rpqubo encode-variable --name x --type continuous --lower 2 --upper 5 --digits 3 --encoding sbe
```

Create `problem.json`:

```json
{
  "name": "tiny",
  "variables": [
    {"name": "x", "type": "continuous", "lower": 0, "upper": 1, "digits": 1}
  ],
  "objective": {
    "constant": 0.1225,
    "linear": {"x": -0.7},
    "quadratic": [{"vars": ["x", "x"], "coef": 1.0}]
  }
}
```

Build, inspect, and solve:

```bash
rpqubo build-qubo problem.json --output tiny_qubo.json
rpqubo inspect-qubo tiny_qubo.json
rpqubo solve tiny_qubo.json --solver exact
```

Reproduce paper Example 1:

```bash
rpqubo reproduce-paper --example ex1 --output-dir outputs/paper
```
