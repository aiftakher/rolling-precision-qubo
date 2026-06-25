# Quickstart

Encode an integer variable. Integer variables default to bounded binary
encoding.

```bash
rpqubo encode-variable --name i --type integer --lower 0 --upper 3
```

Build, solve, and decode a small structured problem:

```bash
rpqubo build-qubo problem.json --output model.model --format model
rpqubo solve model.model --solver exact
```

Regenerate the paper outputs:

```bash
rpqubo reproduce-paper --example all --output-dir outputs/paper
```

The generated report records dependency versions, git state, solver settings,
and the known Alan Table 6 manuscript/code penalty discrepancy.
