"""
Constant-size “zoom-in” Rolling Precision QUBO (with backtracking + dynamic penalties)
====================================================================================

This single script contains *zoom-in / constant-size* versions of ALL the examples we
previously did with “bit growth” (J increasing):

  - Example 1 (unconstrained): min (x1-a)^2 + (x2-b)^2, x in [0,1]^2
  - Example 2 (MIQP): min (x-0.35)^2 + 0.2 y  s.t.  x + 0.8 y <= 1
  - Large instance (alan, MINLPLib): the 4x + 4b problem you pasted

Key idea (constant size):
  - FIX number of bits J for continuous/slack vars -> QUBO size stays constant
  - “precision” comes from shrinking bounds (zoom in), not from adding bits
  - backtracking reverts bounds if zooming doesn’t improve

Dependencies:
  pip install dimod dwave-neal
"""

import time
import csv
from dataclasses import dataclass
from typing import Dict, Tuple, List, Any, Optional

import dimod
from neal import SimulatedAnnealingSampler

# ----------------------------
# SBE encoding (constant size)
# ----------------------------

DEC_WEIGHTS = (1, 2, 3, 3)  # SBE digit weights


def sbe_affine(var: str, J: int, L: float, U: float) -> Tuple[float, Dict[str, float]]:
    """x = c + sum a_i z_i using SBE decimal encoding on [L,U]."""
    if J < 1:
        raise ValueError("SBE requires J >= 1")
    if U < L:
        raise ValueError("Bad bounds: U < L")

    c = L
    width = (U - L)
    coeffs: Dict[str, float] = {}

    for j in range(1, J + 1):
        place = 10 ** (-j)
        for k, w in enumerate(DEC_WEIGHTS, start=1):
            b = f"z_{var}_{j}_{k}"
            coeffs[b] = coeffs.get(b, 0.0) + width * place * w

    tail = f"z_{var}_tail_J{J}"
    coeffs[tail] = coeffs.get(tail, 0.0) + width * (10 ** (-J))
    return c, coeffs


def decode_affine(sample: Dict[str, int], c: float, coeffs: Dict[str, float]) -> float:
    val = c
    for b, a in coeffs.items():
        val += a * float(sample.get(b, 0))
    return val


def normalized_coord(x: float, L: float, U: float) -> float:
    """xhat in [0,1] such that x = L + (U-L)*xhat."""
    w = (U - L)
    if w <= 0:
        return 0.0
    xhat = (x - L) / w
    if xhat < 0:
        return 0.0
    if xhat > 1:
        return 1.0
    return xhat


# ----------------------------
# QUBO helpers
# ----------------------------

def add_linear(Qlin: Dict[str, float], v: str, w: float):
    Qlin[v] = Qlin.get(v, 0.0) + w


def add_quad(Qquad: Dict[Tuple[str, str], float], u: str, v: str, w: float):
    if u == v:
        raise ValueError("Use add_linear for diagonal terms.")
    a, b = (u, v) if u < v else (v, u)
    Qquad[(a, b)] = Qquad.get((a, b), 0.0) + w


def add_square_of_affine(
    Qlin: Dict[str, float],
    Qquad: Dict[Tuple[str, str], float],
    offset_ref: List[float],
    c0: float,
    coeffs: Dict[str, float],
    weight: float,
):
    """Add weight*(c0 + sum a_i z_i)^2 using z^2=z."""
    offset_ref[0] += weight * (c0 * c0)
    items = list(coeffs.items())

    for bi, ai in items:
        add_linear(Qlin, bi, weight * (ai * ai + 2.0 * c0 * ai))

    for i in range(len(items)):
        bi, ai = items[i]
        for j in range(i + 1, len(items)):
            bj, aj = items[j]
            add_quad(Qquad, bi, bj, weight * (2.0 * ai * aj))


def add_product_of_affines(
    Qlin: Dict[str, float],
    Qquad: Dict[Tuple[str, str], float],
    offset_ref: List[float],
    c1: float,
    a1: Dict[str, float],
    c2: float,
    a2: Dict[str, float],
    weight: float,
):
    """Add weight*(c1+sum a1 z)*(c2+sum a2 z)."""
    offset_ref[0] += weight * (c1 * c2)

    for b, a in a2.items():
        add_linear(Qlin, b, weight * (c1 * a))
    for b, a in a1.items():
        add_linear(Qlin, b, weight * (c2 * a))

    for bi, ai in a1.items():
        for bj, aj in a2.items():
            if bi == bj:
                add_linear(Qlin, bi, weight * (ai * aj))  # z^2=z
            else:
                add_quad(Qquad, bi, bj, weight * (ai * aj))


def bqm_stats(bqm: dimod.BinaryQuadraticModel) -> Dict[str, int]:
    return {"n_vars": len(bqm.variables), "n_lin": len(bqm.linear), "n_quad": len(bqm.quadratic)}


def qubo_coeff_stats(
    Qlin: Dict[str, float],
    Qquad: Dict[Tuple[str, str], float],
    offset: float = 0.0,
    include_offset: bool = False,
) -> Dict[str, float]:
    """Basic coefficient diagnostics for penalty-sensitivity reporting.

    We report both the maximum magnitude and the ratio max/min nonzero magnitude.
    The latter is a simple proxy for coefficient dynamic range. The offset is
    excluded by default because it does not affect the optimizer.
    """
    vals = list(Qlin.values()) + list(Qquad.values())
    if include_offset:
        vals.append(offset)
    nz = [abs(v) for v in vals if abs(v) > 0.0]
    if not nz:
        return {
            "max_abs": 0.0,
            "min_nonzero_abs": 0.0,
            "dynamic_range": 0.0,
            "n_nonzero_coeffs": 0,
        }
    max_abs = max(nz)
    min_abs = min(nz)
    return {
        "max_abs": max_abs,
        "min_nonzero_abs": min_abs,
        "dynamic_range": max_abs / min_abs if min_abs > 0 else float("inf"),
        "n_nonzero_coeffs": len(nz),
    }


def rescale_bqm_inplace(
    Qlin: Dict[str, float],
    Qquad: Dict[Tuple[str, str], float],
    offset_ref: List[float],
    target_max_abs: float = 10.0,
) -> float:
    """Scale biases to keep magnitudes moderate (numerical stability)."""
    max_abs = 0.0
    for v in Qlin.values():
        max_abs = max(max_abs, abs(v))
    for v in Qquad.values():
        max_abs = max(max_abs, abs(v))
    max_abs = max(max_abs, abs(offset_ref[0]))

    if max_abs <= target_max_abs or max_abs == 0.0:
        return 1.0

    s = max_abs / target_max_abs
    for k in list(Qlin.keys()):
        Qlin[k] /= s
    for k in list(Qquad.keys()):
        Qquad[k] /= s
    offset_ref[0] /= s
    return s


# ----------------------------
# Zoom + backtracking core
# ----------------------------

@dataclass
class SAOptions:
    num_reads: int = 300
    sweeps: int = 4000
    seed: Optional[int] = 13


@dataclass
class PenaltyConfig:
    dynamic: bool = True
    lam_cap: float = 1e6
    eps: float = 1e-12


@dataclass
class ZoomConfig:
    rho: float = 0.2
    max_iters: int = 25
    no_improve_stop: int = 2

    # acceptance
    feas_tol: float = 5e-3
    feas_eps: float = 1e-6
    obj_eps: float = 1e-8


def anchored_shrink_bounds(
    L0: float,
    U0: float,
    L: float,
    U: float,
    x_val: float,
    rho: float,
    min_width: float,
) -> Tuple[float, float]:
    """
    Shrink [L,U] by rho while anchoring xhat so incumbent stays representable.
    """
    width = U - L
    width_new = max(min_width, rho * width)
    xhat = normalized_coord(x_val, L, U)

    L_new = x_val - width_new * xhat
    U_new = L_new + width_new

    # clip into [L0,U0] preserving width_new
    if L_new < L0:
        L_new = L0
        U_new = L0 + width_new
    if U_new > U0:
        U_new = U0
        L_new = U0 - width_new

    # final safety
    L_new = max(L0, L_new)
    U_new = min(U0, U_new)
    if U_new < L_new:
        U_new = L_new
    return (L_new, U_new)


def better_solution(sol: Dict[str, Any], inc: Optional[Dict[str, Any]], cfg: ZoomConfig) -> bool:
    """Lexicographic: improve feasibility, then (if feasible enough) objective."""
    if inc is None:
        return True

    fs, fi = sol["feas"], inc["feas"]
    os, oi = sol["obj"], inc["obj"]

    if fs < fi - cfg.feas_eps:
        return True

    if fs <= cfg.feas_tol and fi <= cfg.feas_tol:
        if os < oi - cfg.obj_eps:
            return True

    return False


# =====================================================================================
# Example 1 (zoom-in, constant size)
#   min (x1-a)^2 + (x2-b)^2, x in [0,1]^2
# =====================================================================================

def solve_example1_at_box(
    a: float,
    b: float,
    bounds: Dict[str, Tuple[float, float]],
    J: int,
    sa: SAOptions,
) -> Dict[str, Any]:
    sampler = SimulatedAnnealingSampler()

    c1, A1 = sbe_affine("x1", J, *bounds["x1"])
    c2, A2 = sbe_affine("x2", J, *bounds["x2"])

    Qlin: Dict[str, float] = {}
    Qquad: Dict[Tuple[str, str], float] = {}
    offset = [0.0]

    add_square_of_affine(Qlin, Qquad, offset, c1 - a, A1, weight=1.0)
    add_square_of_affine(Qlin, Qquad, offset, c2 - b, A2, weight=1.0)

    rescale_bqm_inplace(Qlin, Qquad, offset, target_max_abs=10.0)
    bqm = dimod.BinaryQuadraticModel(Qlin, Qquad, offset[0], vartype=dimod.BINARY)

    t0 = time.perf_counter()
    ss = sampler.sample(bqm, num_reads=sa.num_reads, sweeps=sa.sweeps, seed=sa.seed)
    t1 = time.perf_counter()

    sample = ss.first.sample
    x1 = decode_affine(sample, c1, A1)
    x2 = decode_affine(sample, c2, A2)

    obj = (x1 - a) ** 2 + (x2 - b) ** 2
    feas = 0.0  # unconstrained

    return {
        "x": {"x1": x1, "x2": x2},
        "obj": obj,
        "feas": feas,
        "penE": float(ss.first.energy),
        "stats": bqm_stats(bqm),
        "time_s": (t1 - t0),
        "bounds": dict(bounds),
    }


def rolling_zoom_example1(
    a: float = 0.1234567,
    b: float = 0.7654321,
    J: int = 1,
    sa: SAOptions = SAOptions(),
    zoom: ZoomConfig = ZoomConfig(rho=0.2, max_iters=15),
    min_width: float = 1e-4,
    verbose: bool = True,
) -> Dict[str, Any]:
    # original bounds
    b0 = {"x1": (0.0, 1.0), "x2": (0.0, 1.0)}
    bounds = dict(b0)
    stack: List[Dict[str, Tuple[float, float]]] = []
    hist: List[Dict[str, Any]] = []

    inc: Optional[Dict[str, Any]] = None
    fail_rounds = 0

    for it in range(zoom.max_iters):
        sol = solve_example1_at_box(a, b, bounds, J, sa)
        sol.update({"iter": it, "move": "solve"})
        hist.append(sol)

        if better_solution(sol, inc, zoom):
            inc = sol

        if verbose:
            st = sol["stats"]
            x = sol["x"]
            print(
                f"it={it:02d} move=solve "
                f"x=[{x['x1']:.6f},{x['x2']:.6f}] obj={sol['obj']:.3e} "
                f"nvars={st['n_vars']} nquad={st['n_quad']} time={sol['time_s']:.3f}s"
            )

        # propose zooms
        assert inc is not None
        cx = inc["x"]

        def make_zoom_box(zoom_both: bool) -> Dict[str, Tuple[float, float]]:
            nb = dict(bounds)
            if zoom_both:
                for xi in ["x1", "x2"]:
                    L0, U0 = b0[xi]
                    L, U = bounds[xi]
                    nb[xi] = anchored_shrink_bounds(L0, U0, L, U, cx[xi], zoom.rho, min_width)
            return nb

        candidates = [("zoom", make_zoom_box(True))]

        accepted = False
        for mv, cand_bounds in candidates:
            cand = solve_example1_at_box(a, b, cand_bounds, J, sa)
            cand.update({"iter": it, "move": mv})
            hist.append(cand)

            if better_solution(cand, inc, zoom):
                stack.append(bounds)
                bounds = cand_bounds
                inc = cand
                accepted = True
                fail_rounds = 0

                if verbose:
                    st = cand["stats"]
                    x = cand["x"]
                    print(
                        f"    ACCEPT {mv:<5} "
                        f"x=[{x['x1']:.6f},{x['x2']:.6f}] obj={cand['obj']:.3e} "
                        f"nvars={st['n_vars']} nquad={st['n_quad']} time={cand['time_s']:.3f}s"
                    )
                break

        if accepted:
            continue

        # backtrack
        fail_rounds += 1
        if stack:
            bounds = stack.pop()
            if verbose:
                print("    BACKTRACK")
            continue

        if fail_rounds >= zoom.no_improve_stop:
            break

    baseline = solve_example1_at_box(a, b, b0, J, sa)
    return {"history": hist, "incumbent": inc, "baseline_no_zoom": baseline}


# =====================================================================================
# Example 2 (zoom-in, constant size)
#   min (x-0.35)^2 + 0.2 y
#   s.t. x + 0.8 y <= 1
# Using slack: x + 0.8y + s = 1, s in [0,1]
# =====================================================================================

def solve_example2_at_box(
    bounds_x: Tuple[float, float],
    bounds_s: Tuple[float, float],
    Jx: int,
    Js: int,
    lam0: float,
    pen_cfg: PenaltyConfig,
    sa: SAOptions,
) -> Dict[str, Any]:
    sampler = SimulatedAnnealingSampler()

    cx, Ax = sbe_affine("x", Jx, *bounds_x)
    cs, As = sbe_affine("s", Js, *bounds_s)
    yname = "y"

    # dynamic scaling based on current widths
    wx = max(pen_cfg.eps, bounds_x[1] - bounds_x[0])
    ws = max(pen_cfg.eps, bounds_s[1] - bounds_s[0])
    wref = max(wx, ws, pen_cfg.eps)
    scale = (1.0 / (wref * wref + pen_cfg.eps)) if pen_cfg.dynamic else 1.0
    lam = min(pen_cfg.lam_cap, lam0 * scale)

    Qlin: Dict[str, float] = {}
    Qquad: Dict[Tuple[str, str], float] = {}
    offset = [0.0]

    # objective: (x-0.35)^2 + 0.2 y
    add_square_of_affine(Qlin, Qquad, offset, cx - 0.35, Ax, weight=1.0)
    add_linear(Qlin, yname, 0.2)

    # constraint equality: x + 0.8 y + s - 1 = 0
    g0 = (cx + cs - 1.0)
    g: Dict[str, float] = {}

    for b, a in Ax.items():
        g[b] = g.get(b, 0.0) + a
    for b, a in As.items():
        g[b] = g.get(b, 0.0) + a
    g[yname] = g.get(yname, 0.0) + 0.8

    add_square_of_affine(Qlin, Qquad, offset, g0, g, weight=lam)

    rescale_bqm_inplace(Qlin, Qquad, offset, target_max_abs=10.0)
    bqm = dimod.BinaryQuadraticModel(Qlin, Qquad, offset[0], vartype=dimod.BINARY)

    t0 = time.perf_counter()
    ss = sampler.sample(bqm, num_reads=sa.num_reads, sweeps=sa.sweeps, seed=sa.seed)
    t1 = time.perf_counter()

    sample = ss.first.sample
    x = decode_affine(sample, cx, Ax)
    s = decode_affine(sample, cs, As)
    y = int(sample.get(yname, 0))

    obj = (x - 0.35) ** 2 + 0.2 * y
    viol = max(0.0, x + 0.8 * y - 1.0)      # original inequality violation
    resid = (x + 0.8 * y + s - 1.0)         # penalty equality residual
    feas = max(viol, abs(resid))

    return {
        "x": x, "y": y, "s": s,
        "obj": obj, "feas": feas,
        "viol": viol, "resid": resid,
        "penE": float(ss.first.energy),
        "stats": bqm_stats(bqm),
        "time_s": (t1 - t0),
        "bounds_x": bounds_x,
        "bounds_s": bounds_s,
        "lam_build": lam,
    }


def rolling_zoom_example2(
    Jx: int = 1,
    Js: int = 1,
    lam0: float = 100.0,
    pen_cfg: PenaltyConfig = PenaltyConfig(dynamic=True, lam_cap=1e6),
    sa: SAOptions = SAOptions(),
    zoom: ZoomConfig = ZoomConfig(rho=0.2, max_iters=15, feas_tol=5e-3),
    min_width_x: float = 1e-4,
    min_width_s: float = 1e-4,
    verbose: bool = True,
) -> Dict[str, Any]:
    x0 = (0.0, 1.0)
    s0 = (0.0, 1.0)
    bx, bs = x0, s0

    stack: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    hist: List[Dict[str, Any]] = []
    inc: Optional[Dict[str, Any]] = None
    fail_rounds = 0

    for it in range(zoom.max_iters):
        sol = solve_example2_at_box(bx, bs, Jx, Js, lam0, pen_cfg, sa)
        sol.update({"iter": it, "move": "solve"})
        hist.append(sol)

        if better_solution(sol, inc, zoom):
            inc = sol

        if verbose:
            st = sol["stats"]
            print(
                f"it={it:02d} move=solve "
                f"x={sol['x']:.6f} y={sol['y']} s={sol['s']:.6f} "
                f"obj={sol['obj']:.3e} feas={sol['feas']:.2e} "
                f"viol={sol['viol']:.1e} resid={sol['resid']:+.1e} "
                f"nvars={st['n_vars']} nquad={st['n_quad']} time={sol['time_s']:.3f}s"
            )

        assert inc is not None

        cx, cs = inc["x"], inc["s"]

        def zoom_box(zoom_x: bool, zoom_s: bool) -> Tuple[Tuple[float, float], Tuple[float, float]]:
            nbx, nbs = bx, bs
            if zoom_x:
                nbx = anchored_shrink_bounds(x0[0], x0[1], bx[0], bx[1], cx, zoom.rho, min_width_x)
            if zoom_s:
                nbs = anchored_shrink_bounds(s0[0], s0[1], bs[0], bs[1], cs, zoom.rho, min_width_s)
            return nbx, nbs

        candidates = [
            ("zoom_xs", zoom_box(True, True)),
            ("zoom_x",  zoom_box(True, False)),
            ("zoom_s",  zoom_box(False, True)),
        ]

        accepted = False
        for mv, (bx_c, bs_c) in candidates:
            cand = solve_example2_at_box(bx_c, bs_c, Jx, Js, lam0, pen_cfg, sa)
            cand.update({"iter": it, "move": mv})
            hist.append(cand)

            if better_solution(cand, inc, zoom):
                stack.append((bx, bs))
                bx, bs = bx_c, bs_c
                inc = cand
                accepted = True
                fail_rounds = 0

                if verbose:
                    st = cand["stats"]
                    print(
                        f"    ACCEPT {mv:<7} "
                        f"x={cand['x']:.6f} y={cand['y']} s={cand['s']:.6f} "
                        f"obj={cand['obj']:.3e} feas={cand['feas']:.2e} "
                        f"nvars={st['n_vars']} nquad={st['n_quad']} time={cand['time_s']:.3f}s"
                    )
                break

        if accepted:
            continue

        fail_rounds += 1
        if stack:
            bx, bs = stack.pop()
            if verbose:
                print("    BACKTRACK")
            continue

        if fail_rounds >= zoom.no_improve_stop:
            break

    baseline = solve_example2_at_box(x0, s0, Jx, Js, lam0, pen_cfg, sa)
    return {"history": hist, "incumbent": inc, "baseline_no_zoom": baseline}


# =====================================================================================
# Large instance (alan) zoom-in constant size
#   (same model you ran earlier, now with zoom-only not bit growth)
# =====================================================================================

@dataclass
class BoxAlan:
    x_bounds: Dict[str, Tuple[float, float]]
    s_bounds: Dict[str, Tuple[float, float]]  # s1..s4 (link slacks)


def solve_alan_at_box(
    box: BoxAlan,
    Jx: int,
    Js: int,
    lam0: Dict[str, float],      # base penalties
    pen_cfg: PenaltyConfig,
    sa: SAOptions,
) -> Dict[str, Any]:
    sampler = SimulatedAnnealingSampler()

    # dynamic scaling
    wx = max((U - L) for (L, U) in box.x_bounds.values())
    ws = max((U - L) for (L, U) in box.s_bounds.values())
    wref = max(wx, ws, pen_cfg.eps)
    scale = (1.0 / (wref * wref + pen_cfg.eps)) if pen_cfg.dynamic else 1.0

    lam_e1   = min(pen_cfg.lam_cap, lam0["e1"]   * scale)
    lam_e2   = min(pen_cfg.lam_cap, lam0["e2"]   * scale)
    lam_link = min(pen_cfg.lam_cap, lam0["link"] * scale)
    lam_card = min(pen_cfg.lam_cap, lam0["card"] * scale)

    # encode x
    cx, ax = {}, {}
    for xi, (L, U) in box.x_bounds.items():
        c, a = sbe_affine(xi, Jx, L, U)
        cx[xi] = c
        ax[xi] = a

    # encode link slacks s1..s4
    cs, as_ = {}, {}
    for si, (L, U) in box.s_bounds.items():
        c, a = sbe_affine(si, Js, L, U)
        cs[si] = c
        as_[si] = a

    # binaries
    bnames = ["b6", "b7", "b8", "b9"]

    # cardinality slack sc in {0,1,2,3}: sc = sc0 + 2 sc1
    sc0, sc1 = "sc0", "sc1"

    Qlin: Dict[str, float] = {}
    Qquad: Dict[Tuple[str, str], float] = {}
    offset = [0.0]

    # objective:
    # 4 x1^2 + 6 x1 x2 - 2 x1 x3 + 6 x2^2 + 2 x2 x3 + 10 x3^2
    add_square_of_affine(Qlin, Qquad, offset, cx["x1"], ax["x1"], weight=4.0)
    add_product_of_affines(Qlin, Qquad, offset, cx["x1"], ax["x1"], cx["x2"], ax["x2"], weight=6.0)
    add_product_of_affines(Qlin, Qquad, offset, cx["x1"], ax["x1"], cx["x3"], ax["x3"], weight=-2.0)
    add_square_of_affine(Qlin, Qquad, offset, cx["x2"], ax["x2"], weight=6.0)
    add_product_of_affines(Qlin, Qquad, offset, cx["x2"], ax["x2"], cx["x3"], ax["x3"], weight=2.0)
    add_square_of_affine(Qlin, Qquad, offset, cx["x3"], ax["x3"], weight=10.0)

    # e1: x1+x2+x3+x4 = 1
    g1_0 = (cx["x1"] + cx["x2"] + cx["x3"] + cx["x4"] - 1.0)
    g1: Dict[str, float] = {}
    for xi in ["x1", "x2", "x3", "x4"]:
        for b, a in ax[xi].items():
            g1[b] = g1.get(b, 0.0) + a
    add_square_of_affine(Qlin, Qquad, offset, g1_0, g1, weight=lam_e1)

    # e2: 8x1+9x2+12x3+7x4 = 10
    g2_0 = (8.0*cx["x1"] + 9.0*cx["x2"] + 12.0*cx["x3"] + 7.0*cx["x4"] - 10.0)
    g2: Dict[str, float] = {}
    wts = {"x1": 8.0, "x2": 9.0, "x3": 12.0, "x4": 7.0}
    for xi in ["x1", "x2", "x3", "x4"]:
        for b, a in ax[xi].items():
            g2[b] = g2.get(b, 0.0) + wts[xi]*a
    add_square_of_affine(Qlin, Qquad, offset, g2_0, g2, weight=lam_e2)

    # link: x_i <= b_i  -> x_i - b_i + s_i = 0, s_i in [0,1]
    link_map = [("x1", "b6", "s1"), ("x2", "b7", "s2"), ("x3", "b8", "s3"), ("x4", "b9", "s4")]
    for xi, bi, si in link_map:
        r0 = cx[xi] + cs[si]
        r: Dict[str, float] = {}
        for b, a in ax[xi].items():
            r[b] = r.get(b, 0.0) + a
        for b, a in as_[si].items():
            r[b] = r.get(b, 0.0) + a
        r[bi] = r.get(bi, 0.0) - 1.0
        add_square_of_affine(Qlin, Qquad, offset, r0, r, weight=lam_link)

    # card: b6+b7+b8+b9 <= 3 -> bsum + sc - 3 = 0
    c0 = -3.0
    gc: Dict[str, float] = {sc0: 1.0, sc1: 2.0}
    for bi in bnames:
        gc[bi] = gc.get(bi, 0.0) + 1.0
    add_square_of_affine(Qlin, Qquad, offset, c0, gc, weight=lam_card)

    coeff_stats_unscaled = qubo_coeff_stats(Qlin, Qquad, offset[0])
    rescale_factor = rescale_bqm_inplace(Qlin, Qquad, offset, target_max_abs=10.0)
    coeff_stats_scaled = qubo_coeff_stats(Qlin, Qquad, offset[0])
    bqm = dimod.BinaryQuadraticModel(Qlin, Qquad, offset[0], vartype=dimod.BINARY)

    t0 = time.perf_counter()
    ss = sampler.sample(bqm, num_reads=sa.num_reads, sweeps=sa.sweeps, seed=sa.seed)
    t1 = time.perf_counter()

    sample = ss.first.sample
    energy = float(ss.first.energy)

    x = {xi: decode_affine(sample, cx[xi], ax[xi]) for xi in ["x1", "x2", "x3", "x4"]}
    b = {bi: int(sample.get(bi, 0)) for bi in bnames}
    s = {si: decode_affine(sample, cs[si], as_[si]) for si in ["s1", "s2", "s3", "s4"]}
    sc_val = int(sample.get(sc0, 0)) + 2 * int(sample.get(sc1, 0))

    obj = (
        4.0 * x["x1"] ** 2
        + 6.0 * x["x1"] * x["x2"]
        - 2.0 * x["x1"] * x["x3"]
        + 6.0 * x["x2"] ** 2
        + 2.0 * x["x2"] * x["x3"]
        + 10.0 * x["x3"] ** 2
    )

    e1 = (x["x1"] + x["x2"] + x["x3"] + x["x4"] - 1.0)
    e2 = (8.0*x["x1"] + 9.0*x["x2"] + 12.0*x["x3"] + 7.0*x["x4"] - 10.0)

    linkV = max(0.0, x["x1"] - b["b6"], x["x2"] - b["b7"], x["x3"] - b["b8"], x["x4"] - b["b9"])
    cardV = max(0.0, (b["b6"] + b["b7"] + b["b8"] + b["b9"]) - 3)

    linkR = max(
        abs(x["x1"] - b["b6"] + s["s1"]),
        abs(x["x2"] - b["b7"] + s["s2"]),
        abs(x["x3"] - b["b8"] + s["s3"]),
        abs(x["x4"] - b["b9"] + s["s4"]),
    )
    cardR = (b["b6"] + b["b7"] + b["b8"] + b["b9"]) + sc_val - 3

    feas = max(abs(e1), abs(e2), linkV, cardV)

    return {
        "x": x, "b": b, "s": s, "sc": sc_val,
        "obj": obj, "feas": feas,
        "penE": energy,
        "e1": e1, "e2": e2,
        "linkV": linkV, "linkR": linkR,
        "cardV": cardV, "cardR": cardR,
        "stats": bqm_stats(bqm),
        "coeff_stats_unscaled": coeff_stats_unscaled,
        "coeff_stats_scaled": coeff_stats_scaled,
        "rescale_factor": rescale_factor,
        "lam_build": {"e1": lam_e1, "e2": lam_e2, "link": lam_link, "card": lam_card},
        "time_s": (t1 - t0),
        "box": {"x": dict(box.x_bounds), "s": dict(box.s_bounds)},
    }


def rolling_zoom_alan(
    Jx: int = 1,
    Js: int = 1,
    lam0: Dict[str, float] = None,
    pen_cfg: PenaltyConfig = PenaltyConfig(dynamic=True, lam_cap=1e6),
    sa: SAOptions = SAOptions(),
    zoom: ZoomConfig = ZoomConfig(rho=0.2, max_iters=20, feas_tol=5e-3),
    min_width_x: float = 1e-4,
    min_width_s: float = 1e-4,
    verbose: bool = True,
) -> Dict[str, Any]:
    if lam0 is None:
        lam0 = {"e1": 200.0, "e2": 200.0, "link": 300.0, "card": 200.0}

    x0 = {f"x{i}": (0.0, 1.0) for i in range(1, 5)}
    s0 = {f"s{i}": (0.0, 1.0) for i in range(1, 5)}

    box = BoxAlan(x_bounds=dict(x0), s_bounds=dict(s0))
    stack: List[BoxAlan] = []
    hist: List[Dict[str, Any]] = []

    inc: Optional[Dict[str, Any]] = None
    fail_rounds = 0

    for it in range(zoom.max_iters):
        sol = solve_alan_at_box(box, Jx, Js, lam0, pen_cfg, sa)
        sol.update({"iter": it, "move": "solve"})
        hist.append(sol)

        if better_solution(sol, inc, zoom):
            inc = sol

        if verbose:
            st = sol["stats"]
            x = sol["x"]
            b = sol["b"]
            print(
                f"it={it:02d} move=solve "
                f"x=[{x['x1']:.6f},{x['x2']:.6f},{x['x3']:.6f},{x['x4']:.6f}] "
                f"b=[{b['b6']},{b['b7']},{b['b8']},{b['b9']}] "
                f"obj={sol['obj']:.6e} feas={sol['feas']:.2e} "
                f"e1={sol['e1']:+.1e} e2={sol['e2']:+.1e} "
                f"linkV={sol['linkV']:.1e} cardV={sol['cardV']:.1e} "
                f"nvars={st['n_vars']} nquad={st['n_quad']} time={sol['time_s']:.3f}s"
            )

        assert inc is not None
        cx = inc["x"]
        cs = inc["s"]

        def make_zoom_box(zoom_x: bool, zoom_s: bool) -> BoxAlan:
            nx = dict(box.x_bounds)
            ns = dict(box.s_bounds)

            if zoom_x:
                for xi in ["x1", "x2", "x3", "x4"]:
                    L0, U0 = x0[xi]
                    L, U = box.x_bounds[xi]
                    nx[xi] = anchored_shrink_bounds(L0, U0, L, U, cx[xi], zoom.rho, min_width_x)

            if zoom_s:
                for si in ["s1", "s2", "s3", "s4"]:
                    L0, U0 = s0[si]
                    L, U = box.s_bounds[si]
                    ns[si] = anchored_shrink_bounds(L0, U0, L, U, cs[si], zoom.rho, min_width_s)

            return BoxAlan(x_bounds=nx, s_bounds=ns)

        candidates = [
            ("zoom_xs", make_zoom_box(True, True)),
            ("zoom_x",  make_zoom_box(True, False)),
            ("zoom_s",  make_zoom_box(False, True)),
        ]

        accepted = False
        for mv, cand_box in candidates:
            cand = solve_alan_at_box(cand_box, Jx, Js, lam0, pen_cfg, sa)
            cand.update({"iter": it, "move": mv})
            hist.append(cand)

            if better_solution(cand, inc, zoom):
                stack.append(box)
                box = cand_box
                inc = cand
                accepted = True
                fail_rounds = 0

                if verbose:
                    st = cand["stats"]
                    x = cand["x"]
                    b = cand["b"]
                    print(
                        f"    ACCEPT {mv:<7} "
                        f"x=[{x['x1']:.6f},{x['x2']:.6f},{x['x3']:.6f},{x['x4']:.6f}] "
                        f"b=[{b['b6']},{b['b7']},{b['b8']},{b['b9']}] "
                        f"obj={cand['obj']:.6e} feas={cand['feas']:.2e} "
                        f"nvars={st['n_vars']} nquad={st['n_quad']} time={cand['time_s']:.3f}s"
                    )
                break

        if accepted:
            continue

        fail_rounds += 1
        if stack:
            box = stack.pop()
            if verbose:
                print("    BACKTRACK")
            continue

        if fail_rounds >= zoom.no_improve_stop:
            break

    baseline = solve_alan_at_box(BoxAlan(x_bounds=dict(x0), s_bounds=dict(s0)), Jx, Js, lam0, pen_cfg, sa)
    return {"history": hist, "incumbent": inc, "baseline_no_zoom": baseline}


# =====================================================================================
# Alan penalty-weight sensitivity
# =====================================================================================

def format_alan_x(sol: Dict[str, Any]) -> str:
    x = sol["x"]
    return f"[{x['x1']:.6f},{x['x2']:.6f},{x['x3']:.6f},{x['x4']:.6f}]"


def format_alan_b(sol: Dict[str, Any]) -> str:
    b = sol["b"]
    return f"[{b['b6']},{b['b7']},{b['b8']},{b['b9']}]"


def scalar_lam_dict(lam: float) -> Dict[str, float]:
    """Use the same base penalty for all Alan constraints."""
    return {"e1": lam, "e2": lam, "link": lam, "card": lam}


def alan_solution_row(
    sol: Dict[str, Any],
    lam_input: float,
    seed: Optional[int],
    dynamic_penalty: bool,
    run_zoom: bool,
    label: str,
    total_time_s: Optional[float] = None,
) -> Dict[str, Any]:
    """Flatten an Alan solution into a CSV/table row."""
    coeff_u = sol.get("coeff_stats_unscaled", {})
    coeff_s = sol.get("coeff_stats_scaled", {})
    lam_build = sol.get("lam_build", {})
    stats = sol.get("stats", {})
    return {
        "label": label,
        "lambda_input": lam_input,
        "seed": seed,
        "dynamic_penalty": dynamic_penalty,
        "run_zoom": run_zoom,
        "obj": sol["obj"],
        "feas": sol["feas"],
        "e1_abs": abs(sol["e1"]),
        "e2_abs": abs(sol["e2"]),
        "linkV": sol["linkV"],
        "linkR": sol["linkR"],
        "cardV": sol["cardV"],
        "cardR_abs": abs(sol["cardR"]),
        "x": format_alan_x(sol),
        "b": format_alan_b(sol),
        "n_vars": stats.get("n_vars"),
        "n_quad": stats.get("n_quad"),
        "lam_e1_final": lam_build.get("e1"),
        "lam_e2_final": lam_build.get("e2"),
        "lam_link_final": lam_build.get("link"),
        "lam_card_final": lam_build.get("card"),
        "qubo_max_abs_unscaled": coeff_u.get("max_abs"),
        "qubo_dyn_range_unscaled": coeff_u.get("dynamic_range"),
        "qubo_max_abs_scaled": coeff_s.get("max_abs"),
        "qubo_dyn_range_scaled": coeff_s.get("dynamic_range"),
        "rescale_factor": sol.get("rescale_factor"),
        "solve_time_s": sol.get("time_s"),
        "total_time_s": total_time_s if total_time_s is not None else sol.get("time_s"),
    }


def choose_best_row(rows: List[Dict[str, Any]], feas_tol: float) -> Dict[str, Any]:
    """Choose a representative row: feasible rows by objective, otherwise lowest infeasibility."""
    feasible = [r for r in rows if r["feas"] <= feas_tol]
    if feasible:
        return min(feasible, key=lambda r: (r["obj"], r["feas"]))
    return min(rows, key=lambda r: (r["feas"], r["obj"]))


def run_alan_penalty_sensitivity(
    lambda_values: Optional[List[float]] = None,
    Jx: int = 1,
    Js: int = 1,
    dynamic_penalty: bool = False,
    run_zoom: bool = True,
    seeds: Optional[List[int]] = None,
    zoom: ZoomConfig = ZoomConfig(rho=0.2, max_iters=15, feas_tol=5e-3),
    sa_reads: int = 300,
    sa_sweeps: int = 4000,
    lam_cap: float = 1e6,
    csv_path: str = "alan_penalty_sensitivity.csv",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run a penalty-weight sensitivity study on the Alan instance.

    Recommended use for the manuscript/rebuttal:
      - dynamic_penalty=False gives a direct sensitivity around fixed lambda values
        such as 50, 100, 250, 500, 1000, ... and directly addresses the reviewer
        concern about a fixed lambda=500.
      - run_zoom=True evaluates the actual constant-size rolling-precision workflow.

    The CSV reports objective, feasibility residuals, selected solution, QUBO size,
    and coefficient dynamic-range diagnostics.
    """
    if lambda_values is None:
        # Sensitivity centered around the manuscript value lambda=500.
        lambda_values = [50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0]
    if seeds is None:
        seeds = [13]

    x0 = {f"x{i}": (0.0, 1.0) for i in range(1, 5)}
    s0 = {f"s{i}": (0.0, 1.0) for i in range(1, 5)}
    base_box = BoxAlan(x_bounds=dict(x0), s_bounds=dict(s0))

    raw_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for lam in lambda_values:
        rows_this_lam: List[Dict[str, Any]] = []
        for seed in seeds:
            sa = SAOptions(num_reads=sa_reads, sweeps=sa_sweeps, seed=seed)
            pen_cfg = PenaltyConfig(dynamic=dynamic_penalty, lam_cap=lam_cap)
            lam0 = scalar_lam_dict(lam)

            if run_zoom:
                rep = rolling_zoom_alan(
                    Jx=Jx,
                    Js=Js,
                    lam0=lam0,
                    pen_cfg=pen_cfg,
                    sa=sa,
                    zoom=zoom,
                    min_width_x=1e-4,
                    min_width_s=1e-4,
                    verbose=False,
                )
                sol = rep["incumbent"]
                total_time = sum(h.get("time_s", 0.0) for h in rep["history"])
                label = "zoom_best"
            else:
                sol = solve_alan_at_box(base_box, Jx, Js, lam0, pen_cfg, sa)
                total_time = sol.get("time_s", 0.0)
                label = "single_box"

            row = alan_solution_row(
                sol=sol,
                lam_input=lam,
                seed=seed,
                dynamic_penalty=dynamic_penalty,
                run_zoom=run_zoom,
                label=label,
                total_time_s=total_time,
            )
            raw_rows.append(row)
            rows_this_lam.append(row)

        best = choose_best_row(rows_this_lam, feas_tol=zoom.feas_tol)
        summary_rows.append(best)

        if verbose:
            print(
                f"lambda={lam:8.1f} | seed={best['seed']} | "
                f"obj={best['obj']:.6e} | feas={best['feas']:.2e} | "
                f"e1={best['e1_abs']:.1e} e2={best['e2_abs']:.1e} "
                f"linkV={best['linkV']:.1e} cardV={best['cardV']:.1e} | "
                f"Qmax(unscaled)={best['qubo_max_abs_unscaled']:.2e} "
                f"range(unscaled)={best['qubo_dyn_range_unscaled']:.2e} | "
                f"x={best['x']} b={best['b']}"
            )

    if csv_path:
        fieldnames = list(raw_rows[0].keys()) if raw_rows else []
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(raw_rows)

        summary_path = csv_path.replace(".csv", "_summary.csv")
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

        if verbose:
            print(f"\nWrote raw results to: {csv_path}")
            print(f"Wrote summary results to: {summary_path}")

    return {"raw_rows": raw_rows, "summary_rows": summary_rows}


# =====================================================================================
# MAIN: choose which example to run
# =====================================================================================

if __name__ == "__main__":
    MODE = "alan_sensitivity"   # choose: "ex1", "ex2", "alan", "alan_sensitivity"

    if MODE == "ex1":
        print("\n--- Example 1: constant-size zoom-in ---")
        rep = rolling_zoom_example1(
            a=0.1234567, b=0.7654321,
            J=1,
            sa=SAOptions(num_reads=300, sweeps=4000, seed=11),
            zoom=ZoomConfig(rho=0.2, max_iters=12, feas_tol=0.0),
            min_width=1e-4,
            verbose=True,
        )
        base = rep["baseline_no_zoom"]
        inc = rep["incumbent"]
        st = base["stats"]
        xb = base["x"]
        print("\nBaseline (no zoom):",
              f"x=[{xb['x1']:.6f},{xb['x2']:.6f}] obj={base['obj']:.3e} nvars={st['n_vars']} nquad={st['n_quad']} time={base['time_s']:.3f}s")
        if inc:
            st = inc["stats"]
            xi = inc["x"]
            print("Best (zoom):",
                  f"x=[{xi['x1']:.6f},{xi['x2']:.6f}] obj={inc['obj']:.3e} nvars={st['n_vars']} nquad={st['n_quad']} time={inc['time_s']:.3f}s")

    elif MODE in {"ex2", "ex3"}:
        print("\n--- Example 2: constant-size zoom-in ---")
        rep = rolling_zoom_example2(
            Jx=1, Js=1,
            lam0=100.0,
            pen_cfg=PenaltyConfig(dynamic=True, lam_cap=1e6),
            sa=SAOptions(num_reads=300, sweeps=4000, seed=13),
            zoom=ZoomConfig(rho=0.2, max_iters=12, feas_tol=5e-3),
            min_width_x=1e-4, min_width_s=1e-4,
            verbose=True,
        )
        base = rep["baseline_no_zoom"]
        inc = rep["incumbent"]
        st = base["stats"]
        print("\nBaseline (no zoom):",
              f"x={base['x']:.6f} y={base['y']} s={base['s']:.6f} obj={base['obj']:.3e} feas={base['feas']:.2e} "
              f"nvars={st['n_vars']} nquad={st['n_quad']} time={base['time_s']:.3f}s")
        if inc:
            st = inc["stats"]
            print("Best (zoom):",
                  f"x={inc['x']:.6f} y={inc['y']} s={inc['s']:.6f} obj={inc['obj']:.3e} feas={inc['feas']:.2e} "
                  f"nvars={st['n_vars']} nquad={st['n_quad']} time={inc['time_s']:.3f}s")

    elif MODE == "alan_sensitivity":
        print("\n--- Alan penalty-weight sensitivity ---")
        # Fixed-penalty sweep centered around lambda=500.
        # This directly supports the manuscript/rebuttal discussion of penalty choice.
        run_alan_penalty_sensitivity(
            lambda_values=[50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0],
            Jx=1, Js=1,
            dynamic_penalty=False,
            run_zoom=True,
            seeds=[13],              # use e.g. [13, 17, 23] for a small multi-seed check
            zoom=ZoomConfig(rho=0.2, max_iters=15, feas_tol=5e-3),
            sa_reads=300,
            sa_sweeps=4000,
            csv_path="alan_penalty_sensitivity.csv",
            verbose=True,
        )

    elif MODE == "alan":
        print("\n--- Large instance (alan): constant-size zoom-in ---")
        rep = rolling_zoom_alan(
            Jx=1, Js=1,
            lam0={"e1": 200.0, "e2": 200.0, "link": 300.0, "card": 200.0},
            pen_cfg=PenaltyConfig(dynamic=True, lam_cap=1e6),
            sa=SAOptions(num_reads=300, sweeps=4000, seed=13),
            zoom=ZoomConfig(rho=0.2, max_iters=15, feas_tol=5e-3),
            min_width_x=1e-4, min_width_s=1e-4,
            verbose=True,
        )
        base = rep["baseline_no_zoom"]
        inc = rep["incumbent"]

        def fmt_x(sol):
            x = sol["x"]
            return f"[{x['x1']:.6f},{x['x2']:.6f},{x['x3']:.6f},{x['x4']:.6f}]"

        def fmt_b(sol):
            b = sol["b"]
            return f"[{b['b6']},{b['b7']},{b['b8']},{b['b9']}]"

        st = base["stats"]
        print("\nBaseline (no zoom):",
              f"x={fmt_x(base)} b={fmt_b(base)} obj={base['obj']:.6e} feas={base['feas']:.2e} "
              f"nvars={st['n_vars']} nquad={st['n_quad']} time={base['time_s']:.3f}s")

        if inc:
            st = inc["stats"]
            print("Best (zoom):",
                  f"x={fmt_x(inc)} b={fmt_b(inc)} obj={inc['obj']:.6e} feas={inc['feas']:.2e} "
                  f"nvars={st['n_vars']} nquad={st['n_quad']} time={inc['time_s']:.3f}s")

    else:
        raise ValueError("Unknown MODE. Use 'ex1', 'ex2', or 'alan'.")
