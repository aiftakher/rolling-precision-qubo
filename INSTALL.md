# Install

Supported Python: 3.10 or newer.

```bash
python -m pip install -e ".[dev]"
```

The only annealing backend supported by the package is D-Wave `neal` simulated
annealing. Hardware and hybrid D-Wave backends are not implemented.

For deterministic paper-reference checks, use the pinned versions in
`requirements-lock.txt` or a resolver-generated lock derived from those
constraints.
