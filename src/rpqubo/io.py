"""Import and export helpers for QUBO files."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from .qubo import QUBO


def load_problem_file(path: str | Path) -> Mapping[str, Any]:
    path = Path(path)
    text = path.read_text()
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml  # type: ignore[import-untyped]

        return yaml.safe_load(text)
    return json.loads(text)


def export_qubo(qubo: QUBO, path: str | Path, fmt: str | None = None) -> None:
    path = Path(path)
    fmt = (fmt or path.suffix.lstrip(".") or "json").lower()
    if fmt == "json":
        path.write_text(json.dumps(qubo.to_dict(), indent=2, sort_keys=True) + "\n")
        return
    if fmt in {"csv", "coo"}:
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["u", "v", "bias"])
            writer.writeheader()
            for var, bias in sorted(qubo.linear.items()):
                writer.writerow({"u": var, "v": var, "bias": bias})
            for (u, v), bias in sorted(qubo.quadratic.items()):
                writer.writerow({"u": u, "v": v, "bias": bias})
            if qubo.offset:
                writer.writerow({"u": "__offset__", "v": "__offset__", "bias": qubo.offset})
        return
    if fmt == "npz":
        import numpy as np

        variables = sorted(qubo.variables)
        index = {v: i for i, v in enumerate(variables)}
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        for var, bias in qubo.linear.items():
            rows.append(index[var])
            cols.append(index[var])
            data.append(bias)
        for (u, v), bias in qubo.quadratic.items():
            rows.append(index[u])
            cols.append(index[v])
            data.append(bias)
        np.savez(
            path,
            variables=np.array(variables),
            rows=np.array(rows),
            cols=np.array(cols),
            data=np.array(data),
            offset=np.array([qubo.offset]),
        )
        return
    if fmt == "dimod":
        with path.open("wb") as f:
            f.write(qubo.to_dimod_bqm().to_file().read())
        return
    raise ValueError(f"Unsupported QUBO export format {fmt!r}")


def load_qubo(path: str | Path) -> QUBO:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        return QUBO.from_dict(json.loads(path.read_text()))
    if suffix in {".csv", ".coo"}:
        linear: dict[str, float] = {}
        quadratic: dict[tuple[str, str], float] = {}
        offset = 0.0
        with path.open() as f:
            for row in csv.DictReader(f):
                u = row["u"]
                v = row["v"]
                bias = float(row["bias"])
                if u == "__offset__" and v == "__offset__":
                    offset += bias
                elif u == v:
                    linear[u] = linear.get(u, 0.0) + bias
                else:
                    first, second = sorted((u, v))
                    key = (first, second)
                    quadratic[key] = quadratic.get(key, 0.0) + bias
        return QUBO(linear=linear, quadratic=quadratic, offset=offset)
    if suffix == ".npz":
        import numpy as np

        npz = np.load(path, allow_pickle=False)
        variables = [str(v) for v in npz["variables"]]
        linear_npz: dict[str, float] = {}
        quadratic_npz: dict[tuple[str, str], float] = {}
        for row, col, bias in zip(npz["rows"], npz["cols"], npz["data"]):
            u = variables[int(row)]
            v = variables[int(col)]
            if u == v:
                linear_npz[u] = linear_npz.get(u, 0.0) + float(bias)
            else:
                first, second = sorted((u, v))
                quadratic_npz[(first, second)] = float(bias)
        return QUBO(
            linear=linear_npz,
            quadratic=quadratic_npz,
            offset=float(cast(Any, npz["offset"])[0]),
        )
    raise ValueError(f"Unsupported QUBO file type: {path}")
