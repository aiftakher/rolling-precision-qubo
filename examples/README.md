# Examples

The `examples/paper/` scripts rebuild QUBOs through `rpqubo` and write CSV
outputs under `outputs/paper/`.

Run one example from an editable checkout with:

```bash
PYTHONPATH=src python examples/paper/ex1_unconstrained_bit_growth.py
```

Or use the CLI:

```bash
rpqubo reproduce-paper --example ex1 --output-dir outputs/paper
```
