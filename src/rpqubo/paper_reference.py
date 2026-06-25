"""Notebook-compatible reference builders for paper reproduction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from time import perf_counter
from typing import Any

from .paper_config import (
    ALAN_BIT_GROWTH,
    ALAN_MANUSCRIPT_UNIFORM_500,
    ALAN_PENALTY_SENSITIVITY,
    ALAN_TABLE6_REFERENCE,
    EXAMPLE2_ZOOM,
    AnnealConfig,
    PaperConfig,
    package_versions,
)
from .qubo import (
    QUBO,
    coefficient_stats,
    rescaled_qubo,
)
from .rolling import anchored_shrink_bounds
from .solvers import SolveResult

DEC_WEIGHTS = (1, 2, 3, 3)
Bounds = tuple[float, float]


class _LegacyQUBO(QUBO):
    """QUBO accumulator matching the notebook/legacy floating-point order."""

    def add_linear(self, var: str, coeff: float) -> None:
        self.linear[var] = self.linear.get(var, 0.0) + float(coeff)

    def add_quadratic(self, u: str, v: str, coeff: float) -> None:
        coeff = float(coeff)
        if u == v:
            self.add_linear(u, coeff)
            return
        key = (u, v) if u < v else (v, u)
        self.quadratic[key] = self.quadratic.get(key, 0.0) + coeff

    def add_offset(self, coeff: float) -> None:
        self.offset += float(coeff)


def add_square_of_affine(
    qubo: QUBO,
    constant: float,
    coeffs: Mapping[str, float],
    weight: float = 1.0,
) -> None:
    """Legacy arithmetic order for weight * (constant + coeffs)^2."""

    qubo.add_offset(weight * (constant * constant))
    items = list(coeffs.items())
    for bit, coeff in items:
        qubo.add_linear(bit, weight * (coeff * coeff + 2.0 * constant * coeff))
    for i, (left_bit, left_coeff) in enumerate(items):
        for right_bit, right_coeff in items[i + 1 :]:
            qubo.add_quadratic(
                left_bit,
                right_bit,
                weight * (2.0 * left_coeff * right_coeff),
            )


def add_product_of_affines(
    qubo: QUBO,
    c1: float,
    a1: Mapping[str, float],
    c2: float,
    a2: Mapping[str, float],
    weight: float = 1.0,
) -> None:
    """Legacy arithmetic order for weight * affine1 * affine2."""

    qubo.add_offset(weight * (c1 * c2))
    for bit, coeff in a2.items():
        qubo.add_linear(bit, weight * (c1 * coeff))
    for bit, coeff in a1.items():
        qubo.add_linear(bit, weight * (c2 * coeff))
    for left_bit, left_coeff in a1.items():
        for right_bit, right_coeff in a2.items():
            value = weight * (left_coeff * right_coeff)
            if left_bit == right_bit:
                qubo.add_linear(left_bit, value)
            else:
                qubo.add_quadratic(left_bit, right_bit, value)


@dataclass(frozen=True)
class PenaltyRuntime:
    dynamic: bool
    lambda_cap: float = 1e6
    eps: float = 1e-12


@dataclass(frozen=True)
class ZoomRuntime:
    rho: float = 0.2
    max_iters: int = 15
    no_improve_stop: int = 2
    feas_tol: float = 5e-3
    feas_eps: float = 1e-6
    obj_eps: float = 1e-8


EXAMPLE2_ZOOM_RUNTIME = ZoomRuntime(max_iters=12)
ALAN_ZOOM_RUNTIME = ZoomRuntime(max_iters=15)


def _anneal_options(config: PaperConfig | AnnealConfig) -> dict[str, Any]:
    anneal = config if isinstance(config, AnnealConfig) else config.anneal
    return {"num_reads": anneal.num_reads, "sweeps": anneal.sweeps, "seed": anneal.seed}


def _sbe_affine(
    var: str,
    digits: int,
    lower: float,
    upper: float,
) -> tuple[float, dict[str, float]]:
    if digits < 1:
        raise ValueError("SBE requires digits >= 1")
    if not all(isfinite(v) for v in (lower, upper)) or upper < lower:
        raise ValueError(f"Bad bounds for {var!r}: {(lower, upper)}")
    width = upper - lower
    coeffs: dict[str, float] = {}
    for digit in range(1, digits + 1):
        place = 10.0 ** (-digit)
        for index, weight in enumerate(DEC_WEIGHTS, start=1):
            coeffs[f"z_{var}_{digit}_{index}"] = width * place * float(weight)
    coeffs[f"z_{var}_tail_J{digits}"] = width * (10.0 ** (-digits))
    return lower, coeffs


def _decode(sample: Mapping[str, int], offset: float, coeffs: Mapping[str, float]) -> float:
    return offset + sum(coeff * float(sample.get(bit, 0)) for bit, coeff in coeffs.items())


def _add_linear(qubo: QUBO, var: str, bias: float) -> None:
    qubo.add_linear(var, bias)


def _add_canonical_product(
    qubo: QUBO,
    c1: float,
    a1: Mapping[str, float],
    c2: float,
    a2: Mapping[str, float],
    weight: float,
) -> None:
    add_product_of_affines(qubo, c1, a1, c2, a2, weight)


def _set_reference_order(qubo: QUBO) -> QUBO:
    ordered: list[str] = []
    seen: set[str] = set()
    for var in list(qubo.linear):
        if var not in seen:
            ordered.append(var)
            seen.add(var)
    for u, v in qubo.quadratic:
        for var in (u, v):
            if var not in seen:
                ordered.append(var)
                seen.add(var)
    for var in qubo.variable_order:
        if var not in seen:
            ordered.append(var)
            seen.add(var)
    qubo.variable_order = ordered
    return qubo


def _solve_reference_qubo(
    qubo: QUBO,
    config: PaperConfig | AnnealConfig,
    *,
    rescale_target: float | None = None,
    include_offset: bool = False,
) -> tuple[QUBO, QUBO, float, Any, dict[str, float], dict[str, float], float]:
    unscaled = _set_reference_order(qubo.copy())
    unscaled_stats = coefficient_stats(unscaled, include_offset=False)
    if rescale_target is None:
        scaled = unscaled.copy()
        factor = 1.0
    else:
        scaled, factor = rescaled_qubo(
            unscaled,
            target_max_abs=rescale_target,
            include_offset=include_offset,
        )
    scaled_stats = coefficient_stats(scaled, include_offset=False)
    anneal = config if isinstance(config, AnnealConfig) else config.anneal
    try:
        import dimod
        from neal import SimulatedAnnealingSampler
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("dimod and dwave-neal are required for paper reproduction") from exc
    bqm = dimod.BinaryQuadraticModel(
        scaled.linear,
        scaled.quadratic,
        scaled.offset,
        vartype="BINARY",
    )
    t0 = perf_counter()
    sampleset = SimulatedAnnealingSampler().sample(
        bqm,
        num_reads=anneal.num_reads,
        sweeps=anneal.sweeps,
        seed=anneal.seed,
    )
    elapsed = perf_counter() - t0
    best = sampleset.first
    result = SolveResult(
        sample={str(k): int(v) for k, v in dict(best.sample).items()},
        energy=float(best.energy),
        solver="neal",
        metadata={
            "num_reads": anneal.num_reads,
            "sweeps": anneal.sweeps,
            "seed": anneal.seed,
            "bqm_variable_order": [str(v) for v in bqm.variables],
            "package_versions": package_versions(),
        },
    )
    return unscaled, scaled, factor, result, unscaled_stats, scaled_stats, elapsed


def _example2_objective(x_value: float, y_value: int) -> float:
    return (x_value - 0.35) ** 2 + 0.2 * y_value


def solve_example2_reference_at_box(
    bounds_x: Bounds,
    bounds_s: Bounds,
    *,
    digits_x: int = 1,
    digits_s: int = 1,
    lambda0: float = 100.0,
    penalty_runtime: PenaltyRuntime | None = None,
    config: PaperConfig = EXAMPLE2_ZOOM,
) -> dict[str, Any]:
    penalty_runtime = penalty_runtime or PenaltyRuntime(dynamic=config.dynamic_penalty)
    cx, ax = _sbe_affine("x", digits_x, *bounds_x)
    cs, a_s = _sbe_affine("s", digits_s, *bounds_s)

    width_x = max(penalty_runtime.eps, bounds_x[1] - bounds_x[0])
    width_s = max(penalty_runtime.eps, bounds_s[1] - bounds_s[0])
    width_ref = max(width_x, width_s, penalty_runtime.eps)
    scale = 1.0 / (width_ref * width_ref + penalty_runtime.eps) if penalty_runtime.dynamic else 1.0
    penalty = min(penalty_runtime.lambda_cap, lambda0 * scale)

    qubo = _LegacyQUBO()
    add_square_of_affine(qubo, cx - 0.35, ax, 1.0)
    _add_linear(qubo, "y", 0.2)
    residual = dict(ax)
    for bit, coeff in a_s.items():
        residual[bit] = residual.get(bit, 0.0) + coeff
    residual["y"] = residual.get("y", 0.0) + 0.8
    add_square_of_affine(qubo, cx + cs - 1.0, residual, penalty)
    unscaled, scaled, factor, result, stats_u, stats_s, elapsed = _solve_reference_qubo(
        qubo,
        config,
        rescale_target=config.rescale_target,
        include_offset=config.include_offset_in_rescale,
    )
    sample = result.sample
    x_value = _decode(sample, cx, ax)
    s_value = _decode(sample, cs, a_s)
    y_value = int(sample.get("y", 0))
    violation = max(0.0, x_value + 0.8 * y_value - 1.0)
    residual_value = x_value + 0.8 * y_value + s_value - 1.0
    return {
        "x": x_value,
        "y": y_value,
        "s": s_value,
        "objective": _example2_objective(x_value, y_value),
        "feasibility": max(violation, abs(residual_value)),
        "violation": violation,
        "residual": residual_value,
        "energy": result.energy,
        "n_vars": len(scaled.variables),
        "n_quad": len(scaled.quadratic),
        "qubo_max_abs_unscaled": stats_u["max_abs"],
        "qubo_dyn_range_unscaled": stats_u["dynamic_range"],
        "qubo_max_abs_scaled": stats_s["max_abs"],
        "qubo_dyn_range_scaled": stats_s["dynamic_range"],
        "rescale_factor": factor,
        "lambda_build": penalty,
        "solver": result.solver,
        "solver_metadata": result.metadata,
        "time_s": elapsed,
        "point": {"x": x_value, "s": s_value},
        "bounds_x": bounds_x,
        "bounds_s": bounds_s,
        "config_name": config.name,
    }


def _better(
    candidate: Mapping[str, Any],
    incumbent: Mapping[str, Any] | None,
    cfg: ZoomRuntime,
) -> bool:
    if incumbent is None:
        return True
    cf = float(candidate["feasibility"])
    inf = float(incumbent["feasibility"])
    co = float(candidate["objective"])
    io = float(incumbent["objective"])
    if cf < inf - cfg.feas_eps:
        return True
    return cf <= cfg.feas_tol and inf <= cfg.feas_tol and co < io - cfg.obj_eps


def rolling_example2_zoom_reference(
    *,
    config: PaperConfig = EXAMPLE2_ZOOM,
    zoom: ZoomRuntime = EXAMPLE2_ZOOM_RUNTIME,
) -> dict[str, Any]:
    x0: Bounds = (0.0, 1.0)
    s0: Bounds = (0.0, 1.0)
    bx, bs = x0, s0
    stack: list[tuple[Bounds, Bounds]] = []
    history: list[dict[str, Any]] = []
    incumbent: dict[str, Any] | None = None
    fail_rounds = 0

    for iteration in range(zoom.max_iters):
        sol = solve_example2_reference_at_box(bx, bs, config=config)
        sol.update({"iter": iteration, "action": "baseline" if iteration == 0 else "solve"})
        history.append(sol)
        if _better(sol, incumbent, zoom):
            incumbent = sol
        assert incumbent is not None

        candidate_boxes = [
            (
                "zoom_xs",
                anchored_shrink_bounds(
                    x0[0], x0[1], bx[0], bx[1], incumbent["x"], zoom.rho, 1e-4
                ),
                anchored_shrink_bounds(
                    s0[0], s0[1], bs[0], bs[1], incumbent["s"], zoom.rho, 1e-4
                ),
            ),
            (
                "zoom_x",
                anchored_shrink_bounds(
                    x0[0], x0[1], bx[0], bx[1], incumbent["x"], zoom.rho, 1e-4
                ),
                bs,
            ),
            (
                "zoom_s",
                bx,
                anchored_shrink_bounds(
                    s0[0], s0[1], bs[0], bs[1], incumbent["s"], zoom.rho, 1e-4
                ),
            ),
        ]
        accepted = False
        for move, bx_c, bs_c in candidate_boxes:
            cand = solve_example2_reference_at_box(bx_c, bs_c, config=config)
            better = _better(cand, incumbent, zoom)
            cand.update(
                {
                    "iter": iteration,
                    "action": "accepted_zoom" if better else "rejected_zoom",
                    "move": move,
                }
            )
            history.append(cand)
            if better:
                stack.append((bx, bs))
                bx, bs = bx_c, bs_c
                incumbent = cand
                accepted = True
                fail_rounds = 0
                break

        if accepted:
            continue
        fail_rounds += 1
        if stack:
            from_box = {"x": bx, "s": bs}
            bx, bs = stack.pop()
            history.append(
                {
                    **incumbent,
                    "iter": iteration,
                    "action": "backtrack",
                    "move": "backtrack",
                    "from_box": from_box,
                    "box": {"x": bx, "s": bs},
                }
            )
            continue
        if fail_rounds >= zoom.no_improve_stop:
            break

    return {
        "history": history,
        "incumbent": incumbent,
        "backtrack_count": sum(1 for r in history if r.get("action") == "backtrack"),
    }


def _alan_objective(x: Mapping[str, float]) -> float:
    return (
        4.0 * x["x1"] ** 2
        + 6.0 * x["x1"] * x["x2"]
        - 2.0 * x["x1"] * x["x3"]
        + 6.0 * x["x2"] ** 2
        + 2.0 * x["x2"] * x["x3"]
        + 10.0 * x["x3"] ** 2
    )


def _alan_public_row(
    *,
    x: Mapping[str, float],
    b: Mapping[str, int],
    s: Mapping[str, float],
    sc: float,
    energy: float,
    n_vars: int,
    n_quad: int,
    solver: str,
    solver_metadata: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    y = {f"y{i}": b[f"b{i + 5}"] for i in range(1, 5)}
    public_s = {f"s{i}": s[f"s{i + 5}"] for i in range(1, 5)}
    e1 = sum(x[f"x{i}"] for i in range(1, 5)) - 1.0
    e2 = 8.0 * x["x1"] + 9.0 * x["x2"] + 12.0 * x["x3"] + 7.0 * x["x4"] - 10.0
    link_violation = max(max(0.0, x[f"x{i}"] - y[f"y{i}"]) for i in range(1, 5))
    link_residual = max(abs(x[f"x{i}"] - y[f"y{i}"] + public_s[f"s{i}"]) for i in range(1, 5))
    card_violation = max(0.0, sum(y.values()) - 3.0)
    card_residual = sum(y.values()) + sc - 3.0
    row = {
        "x": dict(x),
        "y": y,
        "b": dict(b),
        "s": public_s,
        "legacy_s": dict(s),
        "sc": sc,
        "objective": _alan_objective(x),
        "obj": _alan_objective(x),
        "feasibility": max(abs(e1), abs(e2), link_violation, card_violation),
        "feas": max(abs(e1), abs(e2), link_violation, card_violation),
        "e1": e1,
        "e2": e2,
        "e1_abs": abs(e1),
        "e2_abs": abs(e2),
        "link_violation": link_violation,
        "linkV": link_violation,
        "link_residual": link_residual,
        "linkR": link_residual,
        "card_violation": card_violation,
        "cardV": card_violation,
        "card_residual": card_residual,
        "card_residual_abs": abs(card_residual),
        "cardR": card_residual,
        "cardR_abs": abs(card_residual),
        "energy": energy,
        "pen_energy": energy,
        "n_vars": n_vars,
        "n_quad": n_quad,
        "solver": solver,
        "solver_metadata": dict(solver_metadata),
        "package_versions": package_versions(),
    }
    if extra:
        row.update(extra)
    return row


def solve_alan_bit_growth_reference_at_precision(
    digits_x: int,
    digits_s: int,
    *,
    config: PaperConfig = ALAN_BIT_GROWTH,
) -> dict[str, Any]:
    cx: dict[str, float] = {}
    ax: dict[str, dict[str, float]] = {}
    for name in ("x1", "x2", "x3", "x4"):
        cx[name], ax[name] = _sbe_affine(name, digits_x, 0.0, 1.0)
    cs: dict[str, float] = {}
    a_s: dict[str, dict[str, float]] = {}
    for name in ("s6", "s7", "s8", "s9"):
        cs[name], a_s[name] = _sbe_affine(name, digits_s, 0.0, 1.0)
    csc, asc = _sbe_affine("sc", digits_s, 0.0, 3.0)
    penalties = config.penalties

    qubo = _LegacyQUBO()
    add_square_of_affine(qubo, cx["x1"], ax["x1"], 4.0)
    add_square_of_affine(qubo, cx["x2"], ax["x2"], 6.0)
    add_square_of_affine(qubo, cx["x3"], ax["x3"], 10.0)
    _add_canonical_product(qubo, cx["x1"], ax["x1"], cx["x2"], ax["x2"], 6.0)
    _add_canonical_product(qubo, cx["x1"], ax["x1"], cx["x3"], ax["x3"], -2.0)
    _add_canonical_product(qubo, cx["x2"], ax["x2"], cx["x3"], ax["x3"], 2.0)

    g: dict[str, float] = {}
    for name in ("x1", "x2", "x3", "x4"):
        for bit, coeff in ax[name].items():
            g[bit] = g.get(bit, 0.0) + coeff
    add_square_of_affine(qubo, -1.0, g, penalties["e1"])

    h: dict[str, float] = {}
    for name, weight in {"x1": 8.0, "x2": 9.0, "x3": 12.0, "x4": 7.0}.items():
        for bit, coeff in ax[name].items():
            h[bit] = h.get(bit, 0.0) + weight * coeff
    add_square_of_affine(qubo, -10.0, h, penalties["e2"])

    for x_name, b_name, s_name in (
        ("x1", "b6", "s6"),
        ("x2", "b7", "s7"),
        ("x3", "b8", "s8"),
        ("x4", "b9", "s9"),
    ):
        r: dict[str, float] = {}
        for bit, coeff in ax[x_name].items():
            r[bit] = r.get(bit, 0.0) + coeff
        for bit, coeff in a_s[s_name].items():
            r[bit] = r.get(bit, 0.0) + coeff
        r[b_name] = r.get(b_name, 0.0) - 1.0
        add_square_of_affine(qubo, 0.0, r, penalties["link"])

    card = dict(asc)
    for b_name in ("b6", "b7", "b8", "b9"):
        card[b_name] = card.get(b_name, 0.0) + 1.0
    add_square_of_affine(qubo, csc - 3.0, card, penalties["card"])

    unscaled, scaled, factor, result, stats_u, stats_s, elapsed = _solve_reference_qubo(
        qubo, config
    )
    sample = result.sample
    x = {name: _decode(sample, cx[name], ax[name]) for name in ("x1", "x2", "x3", "x4")}
    b = {name: int(sample.get(name, 0)) for name in ("b6", "b7", "b8", "b9")}
    s = {name: _decode(sample, cs[name], a_s[name]) for name in ("s6", "s7", "s8", "s9")}
    sc = _decode(sample, csc, asc)
    return _alan_public_row(
        x=x,
        b=b,
        s=s,
        sc=sc,
        energy=result.energy,
        n_vars=len(scaled.variables),
        n_quad=len(scaled.quadratic),
        solver=result.solver,
        solver_metadata=result.metadata,
        extra={
            "Jx": digits_x,
            "Js": digits_s,
            "config_name": config.name,
            "rescale_factor": factor,
            "qubo_max_abs_unscaled": stats_u["max_abs"],
            "qubo_dyn_range_unscaled": stats_u["dynamic_range"],
            "qubo_max_abs_scaled": stats_s["max_abs"],
            "qubo_dyn_range_scaled": stats_s["dynamic_range"],
            "time_s": elapsed,
        },
    )


def solve_alan_reference_at_box(
    x_bounds: Mapping[str, Bounds],
    slack_bounds: Mapping[str, Bounds],
    *,
    digits_x: int = 1,
    digits_s: int = 1,
    config: PaperConfig = ALAN_TABLE6_REFERENCE,
    penalty_runtime: PenaltyRuntime | None = None,
) -> dict[str, Any]:
    penalty_runtime = penalty_runtime or PenaltyRuntime(dynamic=config.dynamic_penalty)
    width_x = max(upper - lower for lower, upper in x_bounds.values())
    width_s = max(upper - lower for lower, upper in slack_bounds.values())
    width_ref = max(width_x, width_s, penalty_runtime.eps)
    scale = 1.0 / (width_ref * width_ref + penalty_runtime.eps) if penalty_runtime.dynamic else 1.0
    penalties = {
        name: min(penalty_runtime.lambda_cap, value * scale)
        for name, value in config.penalties.items()
    }

    cx: dict[str, float] = {}
    ax: dict[str, dict[str, float]] = {}
    for name in ("x1", "x2", "x3", "x4"):
        cx[name], ax[name] = _sbe_affine(name, digits_x, *x_bounds[name])
    cs: dict[str, float] = {}
    a_s: dict[str, dict[str, float]] = {}
    for public_name, legacy_name in (("s1", "s1"), ("s2", "s2"), ("s3", "s3"), ("s4", "s4")):
        cs[legacy_name], a_s[legacy_name] = _sbe_affine(
            legacy_name, digits_s, *slack_bounds[public_name]
        )

    qubo = _LegacyQUBO()
    add_square_of_affine(qubo, cx["x1"], ax["x1"], 4.0)
    _add_canonical_product(qubo, cx["x1"], ax["x1"], cx["x2"], ax["x2"], 6.0)
    _add_canonical_product(qubo, cx["x1"], ax["x1"], cx["x3"], ax["x3"], -2.0)
    add_square_of_affine(qubo, cx["x2"], ax["x2"], 6.0)
    _add_canonical_product(qubo, cx["x2"], ax["x2"], cx["x3"], ax["x3"], 2.0)
    add_square_of_affine(qubo, cx["x3"], ax["x3"], 10.0)

    g: dict[str, float] = {}
    for name in ("x1", "x2", "x3", "x4"):
        for bit, coeff in ax[name].items():
            g[bit] = g.get(bit, 0.0) + coeff
    add_square_of_affine(qubo, sum(cx.values()) - 1.0, g, penalties["e1"])

    h: dict[str, float] = {}
    h0 = 8.0 * cx["x1"] + 9.0 * cx["x2"] + 12.0 * cx["x3"] + 7.0 * cx["x4"] - 10.0
    for name, weight in {"x1": 8.0, "x2": 9.0, "x3": 12.0, "x4": 7.0}.items():
        for bit, coeff in ax[name].items():
            h[bit] = h.get(bit, 0.0) + weight * coeff
    add_square_of_affine(qubo, h0, h, penalties["e2"])

    for x_name, b_name, s_name in (
        ("x1", "b6", "s1"),
        ("x2", "b7", "s2"),
        ("x3", "b8", "s3"),
        ("x4", "b9", "s4"),
    ):
        r: dict[str, float] = {}
        for bit, coeff in ax[x_name].items():
            r[bit] = r.get(bit, 0.0) + coeff
        for bit, coeff in a_s[s_name].items():
            r[bit] = r.get(bit, 0.0) + coeff
        r[b_name] = r.get(b_name, 0.0) - 1.0
        add_square_of_affine(qubo, cx[x_name] + cs[s_name], r, penalties["link"])

    card = {"sc0": 1.0, "sc1": 2.0, "b6": 1.0, "b7": 1.0, "b8": 1.0, "b9": 1.0}
    add_square_of_affine(qubo, -3.0, card, penalties["card"])

    unscaled, scaled, factor, result, stats_u, stats_s, elapsed = _solve_reference_qubo(
        qubo,
        config,
        rescale_target=config.rescale_target,
        include_offset=config.include_offset_in_rescale,
    )
    sample = result.sample
    x = {name: _decode(sample, cx[name], ax[name]) for name in ("x1", "x2", "x3", "x4")}
    b = {name: int(sample.get(name, 0)) for name in ("b6", "b7", "b8", "b9")}
    legacy_s = {f"s{i + 5}": _decode(sample, cs[f"s{i}"], a_s[f"s{i}"]) for i in range(1, 5)}
    sc = float(int(sample.get("sc0", 0)) + 2 * int(sample.get("sc1", 0)))
    return _alan_public_row(
        x=x,
        b=b,
        s=legacy_s,
        sc=sc,
        energy=result.energy,
        n_vars=len(scaled.variables),
        n_quad=len(scaled.quadratic),
        solver=result.solver,
        solver_metadata=result.metadata,
        extra={
            "Jx": digits_x,
            "Js": digits_s,
            "config_name": config.name,
            "lambda_build": penalties,
            "lam_build": penalties,
            "rescale_factor": factor,
            "qubo_max_abs_unscaled": stats_u["max_abs"],
            "qubo_dyn_range_unscaled": stats_u["dynamic_range"],
            "qubo_max_abs_scaled": stats_s["max_abs"],
            "qubo_dyn_range_scaled": stats_s["dynamic_range"],
            "time_s": elapsed,
            "point": {**x, **{f"s{i}": legacy_s[f"s{i + 5}"] for i in range(1, 5)}},
            "box": {"x": dict(x_bounds), "s": dict(slack_bounds)},
            "sample": result.sample,
        },
    )


def rolling_alan_zoom_reference(
    *,
    config: PaperConfig = ALAN_TABLE6_REFERENCE,
    zoom: ZoomRuntime = ALAN_ZOOM_RUNTIME,
) -> dict[str, Any]:
    x0 = {f"x{i}": (0.0, 1.0) for i in range(1, 5)}
    s0 = {f"s{i}": (0.0, 1.0) for i in range(1, 5)}
    x_bounds = dict(x0)
    s_bounds = dict(s0)
    stack: list[tuple[dict[str, Bounds], dict[str, Bounds]]] = []
    history: list[dict[str, Any]] = []
    incumbent: dict[str, Any] | None = None
    fail_rounds = 0

    for iteration in range(zoom.max_iters):
        sol = solve_alan_reference_at_box(x_bounds, s_bounds, config=config)
        sol.update(
            {
                "iter": iteration,
                "action": "baseline" if iteration == 0 else "solve",
                "move": "solve",
            }
        )
        history.append(sol)
        if _better(sol, incumbent, zoom):
            incumbent = sol
        assert incumbent is not None

        point = incumbent["point"]

        current_x = dict(x_bounds)
        current_s = dict(s_bounds)
        point_now = dict(point)

        def zoomed(
            do_x: bool,
            do_s: bool,
            *,
            current_x: Mapping[str, Bounds] = current_x,
            current_s: Mapping[str, Bounds] = current_s,
            point_now: Mapping[str, float] = point_now,
        ) -> tuple[dict[str, Bounds], dict[str, Bounds]]:
            next_x = dict(current_x)
            next_s = dict(current_s)
            if do_x:
                for name in x0:
                    next_x[name] = anchored_shrink_bounds(
                        x0[name][0],
                        x0[name][1],
                        current_x[name][0],
                        current_x[name][1],
                        point_now[name],
                        zoom.rho,
                        1e-4,
                    )
            if do_s:
                for name in s0:
                    next_s[name] = anchored_shrink_bounds(
                        s0[name][0],
                        s0[name][1],
                        current_s[name][0],
                        current_s[name][1],
                        point_now[name],
                        zoom.rho,
                        1e-4,
                    )
            return next_x, next_s

        candidates = [
            ("zoom_xs", *zoomed(True, True)),
            ("zoom_x", *zoomed(True, False)),
            ("zoom_s", *zoomed(False, True)),
        ]
        accepted = False
        for move, cand_x, cand_s in candidates:
            cand = solve_alan_reference_at_box(cand_x, cand_s, config=config)
            better = _better(cand, incumbent, zoom)
            cand.update(
                {
                    "iter": iteration,
                    "action": "accepted_zoom" if better else "rejected_zoom",
                    "move": move,
                }
            )
            history.append(cand)
            if better:
                stack.append((dict(x_bounds), dict(s_bounds)))
                x_bounds, s_bounds = dict(cand_x), dict(cand_s)
                incumbent = cand
                accepted = True
                fail_rounds = 0
                break
        if accepted:
            continue
        fail_rounds += 1
        if stack:
            from_box = {"x": dict(x_bounds), "s": dict(s_bounds)}
            x_bounds, s_bounds = stack.pop()
            history.append(
                {
                    **incumbent,
                    "iter": iteration,
                    "action": "backtrack",
                    "move": "backtrack",
                    "from_box": from_box,
                    "box": {"x": dict(x_bounds), "s": dict(s_bounds)},
                }
            )
            continue
        if fail_rounds >= zoom.no_improve_stop:
            break

    return {
        "history": history,
        "incumbent": incumbent,
        "backtrack_count": sum(1 for r in history if r.get("action") == "backtrack"),
    }


def alan_penalty_sensitivity_reference(
    lambda_values: list[float] | None = None,
    *,
    config: PaperConfig = ALAN_PENALTY_SENSITIVITY,
) -> list[dict[str, Any]]:
    values = lambda_values or [50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0]
    rows: list[dict[str, Any]] = []
    for value in values:
        run_config = PaperConfig(
            name=config.name,
            anneal=config.anneal,
            penalties={"e1": value, "e2": value, "link": value, "card": value},
            dynamic_penalty=False,
            rescale_target=config.rescale_target,
            include_offset_in_rescale=config.include_offset_in_rescale,
            cardinality_slack_encoding=config.cardinality_slack_encoding,
        )
        report = rolling_alan_zoom_reference(config=run_config)
        row = dict(report["incumbent"])
        row.update(
            {
                "label": "zoom_best",
                "lambda_input": value,
                "seed": config.anneal.seed,
                "num_reads": config.anneal.num_reads,
                "sweeps": config.anneal.sweeps,
                "dynamic_penalty": False,
                "run_zoom": True,
                "x": _format_x(row["x"]),
                "y": _format_y(row["y"]),
                "b": _format_b(row["b"]),
                "total_time_s": sum(float(h.get("time_s", 0.0)) for h in report["history"]),
                "backtrack_count": report["backtrack_count"],
            }
        )
        rows.append(row)
    return rows


def _format_x(x: Mapping[str, float]) -> str:
    return "[" + ",".join(f"{x[f'x{i}']:.6f}" for i in range(1, 5)) + "]"


def _format_y(y: Mapping[str, int]) -> str:
    return "[" + ",".join(str(y[f"y{i}"]) for i in range(1, 5)) + "]"


def _format_b(b: Mapping[str, int]) -> str:
    return "[" + ",".join(str(b[f"b{i}"]) for i in range(6, 10)) + "]"


def alan_table6_config(mode: str) -> PaperConfig:
    if mode == "paper_table6_reference":
        return ALAN_TABLE6_REFERENCE
    if mode == "manuscript_uniform_500":
        return ALAN_MANUSCRIPT_UNIFORM_500
    raise ValueError("mode must be 'paper_table6_reference' or 'manuscript_uniform_500'")
