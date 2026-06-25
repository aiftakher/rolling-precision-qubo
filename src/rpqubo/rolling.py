"""Generic rolling-precision helpers with backtracking."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any, Literal, cast

Bounds = tuple[float, float]
Box = dict[str, Bounds]
Precision = tuple[int, ...]
Solution = dict[str, Any]


@dataclass(frozen=True)
class AcceptanceConfig:
    """Rules for comparing a trial solution with the incumbent."""

    criterion: str = "penalized_energy"
    improvement_tol: float = 1e-9
    feasibility_tol: float = 5e-3
    feasibility_eps: float = 1e-6
    objective_eps: float = 1e-8
    energy_key: str = "energy"
    objective_key: str = "objective"
    feasibility_key: str = "feasibility"
    backtrack_mode: Literal["restore_box_keep_global_incumbent", "restore_full_state"] = (
        "restore_box_keep_global_incumbent"
    )
    no_improve_stop: int = 2

    def __post_init__(self) -> None:
        allowed = {"penalized_energy", "objective", "feasibility_first"}
        if self.criterion not in allowed:
            raise ValueError(
                f"Unknown criterion {self.criterion!r}; expected one of {sorted(allowed)}"
            )
        if self.backtrack_mode not in {
            "restore_box_keep_global_incumbent",
            "restore_full_state",
        }:
            raise ValueError(f"Unknown backtrack_mode {self.backtrack_mode!r}")
        if self.no_improve_stop < 1:
            raise ValueError("no_improve_stop must be positive")
        for name, value in (
            ("improvement_tol", self.improvement_tol),
            ("feasibility_tol", self.feasibility_tol),
            ("feasibility_eps", self.feasibility_eps),
            ("objective_eps", self.objective_eps),
        ):
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class ZoomMove:
    """A named subset of variables to shrink in one candidate move."""

    name: str
    variables: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ZoomMove.name must be nonempty")
        if not self.variables:
            raise ValueError(f"Zoom move {self.name!r} contains no variables")
        if len(self.variables) != len(set(self.variables)):
            raise ValueError(f"Zoom move {self.name!r} contains duplicate variables")


@dataclass(frozen=True)
class PrecisionMove:
    """A named delta in precision-vector space."""

    name: str
    delta: Precision

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("PrecisionMove.name must be nonempty")
        if not self.delta:
            raise ValueError("PrecisionMove.delta must be nonempty")


def _metric(solution: Mapping[str, Any], key: str) -> float:
    if key not in solution:
        raise KeyError(f"Solution is missing required metric {key!r}")
    value = float(solution[key])
    if not isfinite(value):
        raise ValueError(f"Solution metric {key!r} must be finite")
    return value


def is_better(
    candidate: Mapping[str, Any],
    incumbent: Mapping[str, Any] | None,
    cfg: AcceptanceConfig,
) -> bool:
    """Return whether a candidate should replace the incumbent."""

    if incumbent is None:
        return True

    if cfg.criterion == "penalized_energy":
        return _metric(candidate, cfg.energy_key) < (
            _metric(incumbent, cfg.energy_key) - cfg.improvement_tol
        )

    if cfg.criterion == "objective":
        return _metric(candidate, cfg.objective_key) < (
            _metric(incumbent, cfg.objective_key) - cfg.improvement_tol
        )

    candidate_feas = _metric(candidate, cfg.feasibility_key)
    incumbent_feas = _metric(incumbent, cfg.feasibility_key)

    # First priority: improve feasibility.
    if candidate_feas < incumbent_feas - cfg.feasibility_eps:
        return True

    # Second priority: once both are feasible, improve the original objective.
    if candidate_feas <= cfg.feasibility_tol and incumbent_feas <= cfg.feasibility_tol:
        return _metric(candidate, cfg.objective_key) < (
            _metric(incumbent, cfg.objective_key) - cfg.objective_eps
        )

    return False


def normalized_coordinate(value: float, lower: float, upper: float) -> float:
    """Return the clipped normalized coordinate of a value in an interval."""

    if upper < lower:
        raise ValueError("upper must be at least lower")
    width = upper - lower
    if width <= 0.0:
        return 0.0
    return min(1.0, max(0.0, (value - lower) / width))


def anchored_shrink_bounds(
    original_lower: float,
    original_upper: float,
    current_lower: float,
    current_upper: float,
    value: float,
    rho: float,
    min_width: float,
) -> Bounds:
    """Shrink bounds while keeping the incumbent at the same grid coordinate."""

    values = (
        original_lower,
        original_upper,
        current_lower,
        current_upper,
        value,
        rho,
        min_width,
    )
    if not all(isfinite(v) for v in values):
        raise ValueError("Bounds, value, rho, and min_width must be finite")
    if original_upper < original_lower:
        raise ValueError("Bad original bounds")
    if current_upper < current_lower:
        raise ValueError("Bad current bounds")
    if current_lower < original_lower or current_upper > original_upper:
        raise ValueError("Current bounds must lie inside the original bounds")
    if not 0.0 < rho < 1.0:
        raise ValueError("rho must satisfy 0 < rho < 1")
    if min_width < 0.0:
        raise ValueError("min_width must be nonnegative")

    original_width = original_upper - original_lower
    current_width = current_upper - current_lower
    if original_width == 0.0 or current_width == 0.0:
        return current_lower, current_upper

    new_width = max(min_width, rho * current_width)
    new_width = min(new_width, current_width, original_width)
    if new_width >= current_width:
        return current_lower, current_upper

    xhat = normalized_coordinate(value, current_lower, current_upper)
    new_lower = value - new_width * xhat
    new_upper = new_lower + new_width

    if new_lower < original_lower:
        new_lower = original_lower
        new_upper = original_lower + new_width
    if new_upper > original_upper:
        new_upper = original_upper
        new_lower = original_upper - new_width

    new_lower = max(original_lower, new_lower)
    new_upper = min(original_upper, new_upper)
    if new_upper < new_lower:  # Floating-point guard.
        new_upper = new_lower

    return float(new_lower), float(new_upper)


def _copy_box(box: Mapping[str, Bounds]) -> Box:
    return {
        str(name): (float(bounds[0]), float(bounds[1]))
        for name, bounds in box.items()
    }


def _validate_box(box: Mapping[str, Bounds]) -> None:
    if not box:
        raise ValueError("initial_box must not be empty")
    for name, (lower, upper) in box.items():
        if not name:
            raise ValueError("Box variable names must be nonempty")
        if not isfinite(lower) or not isfinite(upper) or upper < lower:
            raise ValueError(f"Invalid bounds for {name!r}: {(lower, upper)}")


def _box_key(
    box: Mapping[str, Bounds],
    digits: int,
) -> tuple[tuple[str, float, float], ...]:
    return tuple(
        (name, round(lower, digits), round(upper, digits))
        for name, (lower, upper) in sorted(box.items())
    )


def _min_width_for(
    name: str,
    min_width: float | Mapping[str, float],
) -> float:
    value = float(min_width[name]) if isinstance(min_width, Mapping) else float(min_width)
    if not isfinite(value) or value < 0.0:
        raise ValueError(f"Invalid minimum width for {name!r}: {value}")
    return value


def make_zoom_box(
    original_box: Mapping[str, Bounds],
    current_box: Mapping[str, Bounds],
    point: Mapping[str, float],
    variables: Sequence[str],
    *,
    rho: float,
    min_width: float | Mapping[str, float],
) -> Box:
    """Create a candidate box by shrinking only selected variables."""

    candidate = _copy_box(current_box)
    for name in variables:
        if name not in original_box or name not in current_box:
            raise KeyError(f"Unknown zoom variable {name!r}")
        if name not in point:
            raise KeyError(f"Incumbent point is missing {name!r}")

        original_lower, original_upper = original_box[name]
        current_lower, current_upper = current_box[name]
        candidate[name] = anchored_shrink_bounds(
            original_lower,
            original_upper,
            current_lower,
            current_upper,
            float(point[name]),
            rho,
            _min_width_for(name, min_width),
        )
    return candidate


def _default_point_getter(solution: Mapping[str, Any]) -> Mapping[str, float]:
    point = solution.get("point")
    if not isinstance(point, Mapping):
        raise KeyError(
            "solve_at() must return a 'point' mapping, or provide point_getter=..."
        )
    return cast(Mapping[str, float], point)


def _extract_point(
    solution: Mapping[str, Any],
    point_getter: Callable[[Mapping[str, Any]], Mapping[str, float]],
    names: Sequence[str],
) -> dict[str, float]:
    raw = point_getter(solution)
    point: dict[str, float] = {}
    for name in names:
        if name not in raw:
            raise KeyError(f"Point getter did not provide {name!r}")
        value = float(raw[name])
        if not isfinite(value):
            raise ValueError(f"Point value for {name!r} must be finite")
        point[name] = value
    return point


def _zoom_record(
    result: Mapping[str, Any],
    *,
    iteration: int,
    action: str,
    move: str,
    accepted: bool,
    box: Mapping[str, Bounds],
    stack_depth: int,
) -> Solution:
    row = dict(result)
    row.update(
        {
            "iter": iteration,
            "iteration": iteration,
            "action": action,
            "move": move,
            "accepted": accepted,
            "box": _copy_box(box),
            "stack_depth": stack_depth,
        }
    )
    return row


def rolling_zoom_in(
    initial_box: Mapping[str, Bounds],
    solve_at: Callable[[Box], Mapping[str, Any]],
    *,
    candidate_moves: Sequence[ZoomMove] | None = None,
    point_getter: Callable[[Mapping[str, Any]], Mapping[str, float]] | None = None,
    rho: float = 0.2,
    min_width: float | Mapping[str, float] = 1e-4,
    acceptance: AcceptanceConfig | None = None,
    max_iters: int = 25,
    box_key_digits: int = 15,
) -> dict[str, Any]:
    """Run constant-size zoom-in with visited boxes and backtracking.

    Candidate moves are tried in order. The first candidate improving the global
    incumbent is accepted. If all candidates fail, the previous accepted box is
    restored, while the best incumbent found so far is retained.
    """

    if max_iters < 1:
        raise ValueError("max_iters must be at least 1")
    if box_key_digits < 6:
        raise ValueError("box_key_digits must be at least 6")

    original_box = _copy_box(initial_box)
    _validate_box(original_box)

    moves = (
        [ZoomMove("zoom_all", tuple(original_box))]
        if candidate_moves is None
        else list(candidate_moves)
    )
    if not moves:
        raise ValueError("candidate_moves must not be empty")

    move_names: set[str] = set()
    for move in moves:
        if move.name in move_names:
            raise ValueError(f"Duplicate zoom move name {move.name!r}")
        move_names.add(move.name)
        unknown = set(move.variables) - set(original_box)
        if unknown:
            raise ValueError(
                f"Zoom move {move.name!r} references unknown variables: {sorted(unknown)}"
            )
        for name in move.variables:
            _min_width_for(name, min_width)

    cfg = acceptance or AcceptanceConfig(criterion="feasibility_first")
    get_point = point_getter or _default_point_getter

    current_box = _copy_box(original_box)
    baseline_raw = dict(solve_at(_copy_box(current_box)))
    incumbent = _zoom_record(
        baseline_raw,
        iteration=0,
        action="baseline",
        move="init",
        accepted=True,
        box=current_box,
        stack_depth=0,
    )

    history: list[Solution] = [incumbent]
    stack: list[tuple[Box, Solution]] = []
    visited = {_box_key(current_box, box_key_digits)}
    trial_count = 0
    accepted_count = 0
    rejected_count = 0
    backtrack_count = 0
    no_improve_rounds = 0
    termination_reason = "max_iters"

    for iteration in range(1, max_iters + 1):
        point = _extract_point(incumbent, get_point, tuple(original_box))
        current_key = _box_key(current_box, box_key_digits)
        accepted = False
        generated_candidate = False

        for move in moves:
            candidate_box = make_zoom_box(
                original_box,
                current_box,
                point,
                move.variables,
                rho=rho,
                min_width=min_width,
            )
            candidate_key = _box_key(candidate_box, box_key_digits)

            if candidate_key == current_key or candidate_key in visited:
                continue

            generated_candidate = True
            visited.add(candidate_key)
            trial_count += 1

            candidate_raw = dict(solve_at(_copy_box(candidate_box)))
            better = is_better(candidate_raw, incumbent, cfg)
            candidate = _zoom_record(
                candidate_raw,
                iteration=iteration,
                action="accepted_zoom" if better else "rejected_zoom",
                move=move.name,
                accepted=better,
                box=candidate_box,
                stack_depth=len(stack) + (1 if better else 0),
            )
            history.append(candidate)

            if not better:
                rejected_count += 1
                continue

            stack.append((_copy_box(current_box), incumbent))
            current_box = _copy_box(candidate_box)
            incumbent = candidate
            accepted_count += 1
            accepted = True
            no_improve_rounds = 0
            break

        if accepted:
            continue

        if stack:
            departed_box = _copy_box(current_box)
            current_box, previous_incumbent = stack.pop()
            if cfg.backtrack_mode == "restore_full_state":
                incumbent = previous_incumbent
            backtrack_count += 1
            no_improve_rounds += 1
            row = _zoom_record(
                incumbent,
                iteration=iteration,
                action="backtrack",
                move="backtrack",
                accepted=False,
                box=current_box,
                stack_depth=len(stack),
            )
            row["from_box"] = departed_box
            history.append(row)
            if no_improve_rounds >= cfg.no_improve_stop:
                termination_reason = "no_improvement_after_backtracking"
                break
            continue

        termination_reason = (
            "no_unvisited_candidate_boxes"
            if not generated_candidate
            else "no_improving_candidate"
        )
        break
    else:
        termination_reason = "max_iters"

    return {
        "baseline": history[0],
        "baseline_no_zoom": history[0],
        "incumbent": incumbent,
        "current_box": _copy_box(current_box),
        "history": history,
        "visited_boxes": len(visited),
        "trial_count": trial_count,
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "backtrack_count": backtrack_count,
        "termination_reason": termination_reason,
    }


def rolling_bit_growth(
    initial: Precision,
    maximum: Precision,
    solve_at: Callable[[Precision], Mapping[str, Any]],
    *,
    step: int = 1,
    moves: Sequence[PrecisionMove] | None = None,
    acceptance: AcceptanceConfig | None = None,
    allow_backtrack: bool = True,
    max_iters: int = 100,
) -> dict[str, Any]:
    """Run sequential bit growth over a finite precision-vector lattice."""

    if not initial or len(initial) != len(maximum):
        raise ValueError("initial and maximum must be nonempty and have equal length")
    if step < 1 or max_iters < 1:
        raise ValueError("step and max_iters must be positive")
    for index, (start, stop) in enumerate(zip(initial, maximum)):
        if start < 0 or stop < 0 or start > stop:
            raise ValueError(
                f"Invalid precision bounds at index {index}: {(start, stop)}"
            )

    cfg = acceptance or AcceptanceConfig()
    if moves is not None:
        for move in moves:
            if len(move.delta) != len(initial):
                raise ValueError(f"Precision move {move.name!r} has wrong dimension")
    current = initial
    incumbent = dict(solve_at(current))
    incumbent.update(
        {
            "iter": 0,
            "iteration": 0,
            "precision": current,
            "move": "init",
            "action": "baseline",
            "accepted": True,
        }
    )
    history: list[Solution] = [incumbent]
    visited = {current}
    accepted_precisions = [current]
    trial_count = 0
    termination_reason = "max_iters"

    for iteration in range(1, max_iters + 1):
        candidates: list[tuple[str, Precision]] = []
        if moves is None:
            for index in range(len(current)):
                delta = [0] * len(current)
                delta[index] = step
                trial = tuple(current[i] + delta[i] for i in range(len(current)))
                if all(initial[i] <= trial[i] <= maximum[i] for i in range(len(current))):
                    candidates.append((f"refine_{index}", trial))
            if allow_backtrack:
                for index in range(len(current)):
                    delta = [0] * len(current)
                    delta[index] = -step
                    trial = tuple(current[i] + delta[i] for i in range(len(current)))
                    if all(initial[i] <= trial[i] <= maximum[i] for i in range(len(current))):
                        candidates.append((f"lower_{index}", trial))
        else:
            for move in moves:
                trial = tuple(current[i] + move.delta[i] for i in range(len(current)))
                if all(initial[i] <= trial[i] <= maximum[i] for i in range(len(current))):
                    candidates.append((move.name, trial))

        accepted = False
        generated_candidate = False
        for move_name, precision in candidates:
            if precision in visited:
                continue
            generated_candidate = True
            visited.add(precision)
            trial_count += 1

            candidate_raw = dict(solve_at(precision))
            better = is_better(candidate_raw, incumbent, cfg)
            candidate = dict(candidate_raw)
            candidate.update(
                {
                    "iter": iteration,
                    "iteration": iteration,
                    "precision": precision,
                    "move": move_name,
                    "action": "accepted_precision" if better else "rejected_precision",
                    "accepted": better,
                }
            )
            history.append(candidate)

            if better:
                current = precision
                incumbent = candidate
                accepted_precisions.append(precision)
                accepted = True
                break

        if accepted:
            continue

        termination_reason = (
            "no_unvisited_precision_vectors"
            if not generated_candidate
            else "no_improving_precision_move"
        )
        break
    else:
        termination_reason = "max_iters"

    return {
        "baseline": history[0],
        "incumbent": incumbent,
        "current_precision": current,
        "history": history,
        "accepted_precisions": accepted_precisions,
        "visited_precisions": len(visited),
        "trial_count": trial_count,
        "termination_reason": termination_reason,
    }


__all__ = [
    "AcceptanceConfig",
    "Bounds",
    "Box",
    "Precision",
    "PrecisionMove",
    "ZoomMove",
    "anchored_shrink_bounds",
    "is_better",
    "make_zoom_box",
    "normalized_coordinate",
    "rolling_bit_growth",
    "rolling_zoom_in",
]
