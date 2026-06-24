"""Generic rolling-precision helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class AcceptanceConfig:
    criterion: str = "penalized_energy"
    improvement_tol: float = 1e-9
    feasibility_tol: float = 5e-3
    feasibility_eps: float = 1e-6
    objective_eps: float = 1e-8


def is_better(
    candidate: dict[str, Any], incumbent: dict[str, Any] | None, cfg: AcceptanceConfig
) -> bool:
    if incumbent is None:
        return True
    if cfg.criterion == "penalized_energy":
        return candidate["energy"] < incumbent["energy"] - cfg.improvement_tol
    if cfg.criterion == "objective":
        return candidate["objective"] < incumbent["objective"] - cfg.improvement_tol
    if cfg.criterion == "feasibility_first":
        cf = candidate.get("feasibility", 0.0)
        inf = incumbent.get("feasibility", 0.0)
        if cf < inf - cfg.feasibility_eps:
            return True
        if cf <= cfg.feasibility_tol and inf <= cfg.feasibility_tol:
            return candidate["objective"] < incumbent["objective"] - cfg.objective_eps
        return False
    raise ValueError(f"Unknown acceptance criterion {cfg.criterion!r}")


def rolling_bit_growth(
    initial: tuple[int, ...],
    maximum: tuple[int, ...],
    solve_at: Callable[[tuple[int, ...]], dict[str, Any]],
    *,
    step: int = 1,
    acceptance: AcceptanceConfig | None = None,
    allow_backtrack: bool = True,
    max_iters: int = 100,
) -> dict[str, Any]:
    cfg = acceptance or AcceptanceConfig()
    current = initial
    visited = {current}
    incumbent = solve_at(current)
    history = [{**incumbent, "precision": current, "move": "init"}]
    for _ in range(max_iters):
        candidates: list[tuple[str, tuple[int, ...]]] = []
        for i in range(len(current)):
            if current[i] + step <= maximum[i]:
                nxt = list(current)
                nxt[i] += step
                candidates.append((f"refine_{i}", tuple(nxt)))
        if allow_backtrack:
            for i in range(len(current)):
                if current[i] - step >= initial[i]:
                    nxt = list(current)
                    nxt[i] -= step
                    candidates.append((f"backtrack_{i}", tuple(nxt)))
        accepted = False
        for move, precision in candidates:
            if precision in visited:
                continue
            visited.add(precision)
            result = solve_at(precision)
            record = {**result, "precision": precision, "move": move}
            if is_better(record, incumbent, cfg):
                incumbent = record
                current = precision
                history.append(record)
                accepted = True
                break
        if not accepted:
            break
    return {"incumbent": incumbent, "history": history}


def rolling_zoom_in(
    initial_box: dict[str, tuple[float, float]],
    solve_at: Callable[[dict[str, tuple[float, float]]], dict[str, Any]],
    *,
    rho: float = 0.2,
    min_width: float = 1e-4,
    acceptance: AcceptanceConfig | None = None,
    max_iters: int = 25,
) -> dict[str, Any]:
    cfg = acceptance or AcceptanceConfig(criterion="feasibility_first")
    original = dict(initial_box)
    box = dict(initial_box)
    stack: list[dict[str, tuple[float, float]]] = []
    incumbent: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []

    def shrink(current_box: dict[str, tuple[float, float]], point: dict[str, float]):
        new_box: dict[str, tuple[float, float]] = {}
        for name, (lower, upper) in current_box.items():
            width = max(min_width, (upper - lower) * rho)
            center = point[name]
            lo = center - 0.5 * width
            hi = center + 0.5 * width
            base_lo, base_hi = original[name]
            if lo < base_lo:
                hi += base_lo - lo
                lo = base_lo
            if hi > base_hi:
                lo -= hi - base_hi
                hi = base_hi
            new_box[name] = (max(base_lo, lo), min(base_hi, hi))
        return new_box

    for iteration in range(max_iters):
        current = solve_at(box)
        current.update({"iter": iteration, "move": "solve", "box": dict(box)})
        history.append(current)
        if is_better(current, incumbent, cfg):
            incumbent = current
        if incumbent is None:
            break
        candidate_box = shrink(box, incumbent["point"])
        candidate = solve_at(candidate_box)
        candidate.update({"iter": iteration, "move": "zoom", "box": dict(candidate_box)})
        history.append(candidate)
        if is_better(candidate, incumbent, cfg):
            stack.append(box)
            box = candidate_box
            incumbent = candidate
            continue
        if stack:
            box = stack.pop()
            continue
        break
    return {"incumbent": incumbent, "history": history}
