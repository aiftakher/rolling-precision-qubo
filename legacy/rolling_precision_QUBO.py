"""
Rolling-Precision QUBO Solver (binary-decimal encoder) + D-Wave neal SA

- Decimal SBE encoding with weights (1,2,3,3) + tail bit
- General polynomial objective + (optionally) constraints with quadratic penalties
- Rolling precision over decimal digits J_i, with backtracking in J-space
- QUBO solved with neal.SimulatedAnnealingSampler

Dependencies: sympy, dimod, dwave-neal
"""

from __future__ import annotations

import math
import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional, Set

import sympy as sp
import dimod
from neal import SimulatedAnnealingSampler


# ---------------------------------------------------------------------------
# Encoding utilities
# ---------------------------------------------------------------------------

# (1,2,3,3) lets us represent any digit 0..9 using 4 binary bits without invalid patterns
DEC_WEIGHTS = (1, 2, 3, 3)


def dec_bit_name(var: str, j: int, k: int) -> str:
    """Name for digit-bit of variable `var` at decimal place j, weight index k."""
    return f"z_{var}_{j}_{k}"


def tail_bit_name(var: str, J: int) -> str:
    """Name for tail bit at the finest decimal place J."""
    return f"z_{var}_tail_J{J}"


# ---------------------------------------------------------------------------
# Problem specification dataclasses
# ---------------------------------------------------------------------------

@dataclass
class VarSpec:
    """Specification of a real variable x in [L,U] with decimal precision."""
    L: float = 0.0
    U: float = 1.0
    J0: int = 2        # initial decimal precision
    J_inc: int = 2     # step size for refinement/backtrack in J
    J_max: int = 8     # maximum decimal precision


@dataclass
class ConstraintSpec:
    """
    Constraint of the form:
      expr(x) (sense) rhs
    with a quadratic penalty P * g(x)^2, where g(x) = expr(x) - rhs,
    optionally with a slack variable for inequalities.
    """
    expr: str
    sense: str          # "==", "<=", ">="
    rhs: float
    penalty: float      # penalty weight P


@dataclass
class SAOptions:
    """Options for the neal simulated annealing sampler."""
    num_reads: int = 200
    sweeps: int = 2000
    seed: Optional[int] = None


# ---------------------------------------------------------------------------
# Symbolic parsing
# ---------------------------------------------------------------------------

@dataclass
class ParsedProblem:
    sym_vars: Dict[str, sp.Symbol]
    objective: sp.Expr
    constraints: List[Tuple[sp.Expr, str, float, float]]  # expr, sense, rhs, P


def parse_problem(var_names: List[str], objective_str: str,
                  constraints: List[ConstraintSpec]) -> ParsedProblem:
    """Parse objective and constraints into sympy expressions."""
    sym_vars = {v: sp.symbols(v, real=True) for v in var_names}
    objective = sp.sympify(objective_str, locals=sym_vars)

    parsed_cons: List[Tuple[sp.Expr, str, float, float]] = []
    for c in constraints:
        expr = sp.sympify(c.expr, locals=sym_vars)
        if c.sense not in {"==", "<=", ">="}:
            raise ValueError(f"Unsupported constraint sense: {c.sense}")
        parsed_cons.append((expr, c.sense, float(c.rhs), float(c.penalty)))

    return ParsedProblem(sym_vars=sym_vars, objective=objective, constraints=parsed_cons)


# ---------------------------------------------------------------------------
# Encoder: decimal SBE + tail bit for a single variable at precision J
# ---------------------------------------------------------------------------

@dataclass
class Encoder:
    name: str
    L: float
    U: float
    J: int  # decimal precision

    def build(self) -> Tuple[sp.Expr, List[sp.Symbol]]:
        """
        Build the sympy expression x(name) in terms of binary bits, plus the list of bits.

        Encoding (as in the paper):
          xhat = sum_{j=1}^J 10^{-j} sum_{k=1}^4 w_k z_{j,k} + 10^{-J} z_tail
          x    = L + (U - L) * xhat
        """
        if self.U < self.L:
            raise ValueError(f"For {self.name}, U({self.U}) < L({self.L})")
        if self.J <= 0:
            raise ValueError("J must be >= 1")

        bits: List[sp.Symbol] = []
        xhat = 0
        for j in range(1, self.J + 1):
            scale = 10 ** (-j)
            for k, w in enumerate(DEC_WEIGHTS, start=1):
                b = sp.symbols(dec_bit_name(self.name, j, k))
                bits.append(b)
                xhat += scale * w * b

        t = sp.symbols(tail_bit_name(self.name, self.J))
        bits.append(t)
        xhat += (10 ** (-self.J)) * t

        x_expr = self.L + (self.U - self.L) * xhat
        return x_expr, bits


# ---------------------------------------------------------------------------
# Build penalized polynomial in binary variables
# ---------------------------------------------------------------------------

@dataclass
class BuildContext:
    parsed: ParsedProblem
    var_specs: Dict[str, VarSpec]
    Jmap: Dict[str, int]       # current decimal precision J for each real variable

    def build_binary_polynomial(self) -> Tuple[Dict[Tuple[str, ...], float], Dict[str, Encoder]]:
        """
        Build penalized polynomial:
           F(x) = objective(x) + sum P * g(x)^2
        substitute each x with its decimal-binary encoding, expand,
        and return as a dict of monomials in binary vars.
        """
        parsed = self.parsed

        # 1) Encode each real variable according to its current precision Jmap
        encoders: Dict[str, Encoder] = {}
        var_subs: Dict[sp.Symbol, sp.Expr] = {}
        for v, sym in parsed.sym_vars.items():
            spec = self.var_specs[v]
            J = self.Jmap[v]
            enc = Encoder(v, spec.L, spec.U, J)
            x_expr, bits = enc.build()
            encoders[v] = enc
            var_subs[sym] = x_expr

        # 2) Build penalty terms for constraints
        penalties: List[sp.Expr] = []
        for (expr, sense, rhs, P) in parsed.constraints:
            g = expr.subs(var_subs) - rhs
            if sense == "==":
                penalties.append(P * g**2)
            elif sense == "<=":
                # Penalize only positive violation g(x) > 0
                penalties.append(P * sp.Max(g, 0)**2)
            elif sense == ">=":
                penalties.append(P * sp.Min(g, 0)**2)
            else:
                raise ValueError(f"Unsupported sense: {sense}")

        penalized = parsed.objective.subs(var_subs)
        for pen in penalties:
            # Note: sp.Max/Min leads to piecewise; for simplicity we can
            # approximate inequalities by just P * g^2 if you prefer.
            penalized += pen

        # If you want purely polynomial constraints (no Max/Min),
        # replace the above by:
        #   if sense in {"<=", ">="}: penalties.append(P * g**2)

        penalized = sp.expand(penalized)

        # 3) Extract polynomial in binary vars as dict: monomial -> coeff
        # monomial key: sorted tuple of bit names, with duplicates removed (z^2 = z)
        terms: Dict[Tuple[str, ...], float] = {}

        for t in penalized.as_ordered_terms():
            coeff, rest = t.as_coeff_Mul()
            coeff = float(coeff)

            if rest == 1:
                key = tuple()
            else:
                symbols: List[str] = []
                for factor in rest.as_ordered_factors():
                    if isinstance(factor, sp.Symbol):
                        symbols.append(str(factor))
                    else:
                        base, power = factor.as_base_exp()
                        if isinstance(base, sp.Symbol):
                            # b**2, b**3, ... -> b (since b in {0,1})
                            symbols.append(str(base))
                        else:
                            raise ValueError(f"Unexpected factor in polynomial: {factor}")

                key = tuple(sorted(set(symbols)))

            terms[key] = terms.get(key, 0.0) + coeff

        # Remove very small coefficients (numerical noise)
        terms = {k: v for k, v in terms.items() if abs(v) > 1e-15}
        return terms, encoders


# ---------------------------------------------------------------------------
# Rosenberg quadratization to QUBO
# ---------------------------------------------------------------------------

@dataclass
class QUBO:
    linear: Dict[str, float]
    quad: Dict[Tuple[str, str], float]
    offset: float


def qubo_add_linear(Q: QUBO, v: str, w: float):
    Q.linear[v] = Q.linear.get(v, 0.0) + w


def qubo_add_quad(Q: QUBO, u: str, v: str, w: float):
    if u == v:
        qubo_add_linear(Q, u, w)
    else:
        a, b = (u, v) if u < v else (v, u)
        Q.quad[(a, b)] = Q.quad.get((a, b), 0.0) + w


def qubo_add_offset(Q: QUBO, c: float):
    Q.offset += c


def rosenberg_reduce_to_qubo(
    terms: Dict[Tuple[str, ...], float],
    strength: float = 1e3,
) -> Tuple[QUBO, Set[str]]:
    """
    Reduce a pseudo-Boolean polynomial in binary variables to a QUBO
    using Rosenberg's quadratization. 'strength' controls the penalty
    enforcing y = a*b for ancilla variables.
    """
    Q = QUBO(linear={}, quad={}, offset=0.0)
    ancillas: Set[str] = set()
    anc_counter = 0

    def new_anc(a: str, b: str) -> str:
        nonlocal anc_counter
        anc_counter += 1
        name = f"anc_{a}__{b}__{anc_counter}"
        ancillas.add(name)
        return name

    work: List[Tuple[Tuple[str, ...], float]] = list(terms.items())

    while work:
        mon, c = work.pop()
        d = len(mon)

        if d == 0:
            qubo_add_offset(Q, c)
        elif d == 1:
            qubo_add_linear(Q, mon[0], c)
        elif d == 2:
            qubo_add_quad(Q, mon[0], mon[1], c)
        else:
            # pick two variables, introduce ancilla y ≈ a*b, and reduce degree
            a, b, *rest = mon
            y = new_anc(a, b)

            # Penalty enforcing y = a*b:
            # strength * (3y - 2ay - 2by + ab)
            qubo_add_quad(Q, a, b, strength)
            qubo_add_quad(Q, a, y, -2.0 * strength)
            qubo_add_quad(Q, b, y, -2.0 * strength)
            qubo_add_linear(Q, y, 3.0 * strength)

            # Replace ab with y in the high-order term
            new_mon = tuple(sorted((y, *rest)))
            work.append((new_mon, c))

    return Q, ancillas


def qubo_to_bqm(Q: QUBO) -> dimod.BinaryQuadraticModel:
    return dimod.BinaryQuadraticModel(Q.linear, Q.quad, Q.offset, vartype=dimod.BINARY)


# ---------------------------------------------------------------------------
# Rolling-precision annealer with backtracking
# ---------------------------------------------------------------------------

@dataclass
class RollingPrecisionAnnealer:
    var_specs: Dict[str, VarSpec]
    parsed: ParsedProblem
    sa_options: SAOptions = SAOptions()
    quadr_strength: float = 1e3
    max_iters: int = 100
    improvement_tol: float = 1e-9   # minimal decrease in QUBO energy

    Jmap: Dict[str, int] = field(default_factory=dict)

    def _init_Jmap(self):
        """Initialize the precision map Jmap with each variable's J0."""
        if not self.Jmap:
            for v, spec in self.var_specs.items():
                self.Jmap[v] = spec.J0

    def _build_and_sample(self, Jmap: Dict[str, int]) -> Tuple[float, Dict[str, int], Dict[str, Encoder], Dict[str, float]]:
        """
        Build QUBO for given precision Jmap, solve it with SA, and decode real variables.
        Returns: (QUBO_energy, bit_assignment, encoders, decoded_real_values)
        """
        ctx = BuildContext(parsed=self.parsed, var_specs=self.var_specs, Jmap=Jmap)
        terms, encoders = ctx.build_binary_polynomial()

        Q, ancillas = rosenberg_reduce_to_qubo(terms, strength=self.quadr_strength)
        bqm = qubo_to_bqm(Q)

        sampler = SimulatedAnnealingSampler()
        ss = sampler.sample(
            bqm,
            num_reads=self.sa_options.num_reads,
            sweeps=self.sa_options.sweeps,
            seed=self.sa_options.seed,
        )
        best = ss.first
        pen_obj = float(best.energy)
        assign = dict(best.sample)

        decoded = self.decode_reals(assign, encoders)
        return pen_obj, assign, encoders, decoded

    @staticmethod
    def decode_reals(bit_assign: Dict[str, int], encoders: Dict[str, Encoder]) -> Dict[str, float]:
        """
        Decode real-valued variables from bit_assign using the encoder definitions.
        """
        vals: Dict[str, float] = {}
        for name, enc in encoders.items():
            xhat = 0.0
            for j in range(1, enc.J + 1):
                scale = 10 ** (-j)
                for k, w in enumerate(DEC_WEIGHTS, start=1):
                    bname = dec_bit_name(name, j, k)
                    xhat += scale * w * float(bit_assign.get(bname, 0))
            tname = tail_bit_name(name, enc.J)
            xhat += (10 ** (-enc.J)) * float(bit_assign.get(tname, 0))
            vals[name] = enc.L + (enc.U - enc.L) * xhat
        return vals

    def run(self) -> Dict[str, Any]:
        """
        Rolling-precision with backtracking in J-space.

        - Start from J0 for each variable.
        - At each iteration, generate candidate moves:
            * refine: J_i <- J_i + J_inc (if <= J_max)
            * backtrack: J_i <- J_i - J_inc (if >= J0)
        - Evaluate each unvisited J; accept the first that improves
          QUBO energy by at least improvement_tol.
        - Stop when no improving candidate remains or max_iters reached.
        """
        self._init_Jmap()
        var_list = sorted(self.var_specs.keys())

        # Keep track of visited precision vectors to avoid cycles
        visited: Set[Tuple[int, ...]] = set()

        def J_key(J: Dict[str, int]) -> Tuple[int, ...]:
            return tuple(J[v] for v in var_list)

        # Initial solve at J0
        key0 = J_key(self.Jmap)
        visited.add(key0)
        best_pen, best_assign, best_enc, best_dec = self._build_and_sample(self.Jmap)
        best_J = self.Jmap.copy()

        history: List[Dict[str, Any]] = []
        history.append({
            "iter": 0,
            "move": "init",
            "var": None,
            "J": best_J.copy(),
            "pen_obj": best_pen,
            "decoded": {v: best_dec[v] for v in var_list},
        })

        # Rolling precision loop
        it = 0
        improved = True

        while improved and it < self.max_iters:
            it += 1
            improved = False

            # Generate candidate moves: refinements first, then backtracks
            candidates: List[Tuple[str, str, Dict[str, int]]] = []

            for v in var_list:
                spec = self.var_specs[v]
                # refine move
                if best_J[v] + spec.J_inc <= spec.J_max:
                    J_ref = best_J.copy()
                    J_ref[v] += spec.J_inc
                    candidates.append(("refine", v, J_ref))
                # backtrack move
                if best_J[v] - spec.J_inc >= spec.J0:
                    J_back = best_J.copy()
                    J_back[v] -= spec.J_inc
                    candidates.append(("backtrack", v, J_back))

            for move_type, var, Jcand in candidates:
                key = J_key(Jcand)
                if key in visited:
                    continue
                visited.add(key)

                cand_pen, cand_assign, cand_enc, cand_dec = self._build_and_sample(Jcand)

                if cand_pen + self.improvement_tol < best_pen:
                    # Accept this move
                    best_pen = cand_pen
                    best_assign = cand_assign
                    best_enc = cand_enc
                    best_dec = cand_dec
                    best_J = Jcand.copy()
                    improved = True

                    history.append({
                        "iter": it,
                        "move": move_type,
                        "var": var,
                        "J": best_J.copy(),
                        "pen_obj": best_pen,
                        "decoded": {v: best_dec[v] for v in var_list},
                    })
                    break  # go to next outer iteration

            # If no candidate improved, we terminate
        result_x = {v: best_dec[v] for v in var_list}
        return {
            "best_penalized": best_pen,
            "x": result_x,
            "J": best_J,
            "history": history,
        }


# ---------------------------------------------------------------------------
# Convenience + pretty printing
# ---------------------------------------------------------------------------

def solve_polynomial_qubo(
    var_specs: Dict[str, VarSpec],
    objective: str,
    constraints: List[ConstraintSpec],
    sa_options: SAOptions = SAOptions(),
    quadr_strength: float = 1e3,
    max_iters: int = 100,
    improvement_tol: float = 1e-9,
) -> Dict[str, Any]:
    parsed = parse_problem(list(var_specs.keys()), objective, constraints)
    annealer = RollingPrecisionAnnealer(
        var_specs=var_specs,
        parsed=parsed,
        sa_options=sa_options,
        quadr_strength=quadr_strength,
        max_iters=max_iters,
        improvement_tol=improvement_tol,
    )
    return annealer.run()


def print_iteration_log(history: List[Dict[str, Any]]):
    print("\n--- Iteration Log ---")
    for rec in history:
        J = rec["J"]
        x = rec["decoded"]
        J_str = ", ".join(f"{k}:{v}" for k, v in sorted(J.items()))
        x_str = ", ".join(f"{k}={x[k]:.10f}" for k in sorted(x))
        print(
            f"it={rec['iter']:03d}  move={rec['move']:>9}  var={str(rec['var']):>3}  "
            f"pen_obj={rec['pen_obj']:.12g}  "
            f"J={{{J_str}}}  "
            f"x={x_str}"
        )


if __name__ == "__main__":
    # ------------------------------------------------------------
    # Test 1: Unconstrained toy
    #   min (x1 - 0.1234567)^2 + (x2 - 0.7654321)^2,  x in [0,1]^2
    # ------------------------------------------------------------
    # vars_cfg = {
    #     "x1": VarSpec(L=0.0, U=1.0, J0=2, J_inc=2, J_max=8),
    #     "x2": VarSpec(L=0.0, U=1.0, J0=2, J_inc=2, J_max=8),
    # }
    # objective = "(x1 - 0.1234567)**2 + (x2 - 0.7654321)**2"
    # constraints: List[ConstraintSpec] = []  # unconstrained

    # result1 = solve_polynomial_qubo(
    #     var_specs=vars_cfg,
    #     objective=objective,
    #     constraints=constraints,
    #     sa_options=SAOptions(num_reads=200, sweeps=2500, seed=11),
    #     quadr_strength=1e3,
    #     max_iters=20,
    #     improvement_tol=1e-10,
    # )

    # print("=== Test 1: Unconstrained Quadratic ===")
    # print_iteration_log(result1["history"])
    # print("\nFinal:")
    # print("best_penalized:", result1["best_penalized"])
    # print("x:", {k: f"{v:.10f}" for k, v in result1["x"].items()})
    # print("J:", result1["J"])


    # ------------------------------------------------------------
    # Test 2: Simple constrained quadratic
    #   min (x1 - 0.3)^2 + (x2 - 0.6)^2
    #   s.t. x1 + x2 <= 1
    # ------------------------------------------------------------
    vars_cfg2 = {
        "x1": VarSpec(L=0.0, U=1.0, J0=2, J_inc=2, J_max=8),
        "x2": VarSpec(L=0.0, U=1.0, J0=2, J_inc=2, J_max=8),
    }
    objective2 = "(x1 - 0.3)**2 + (x2 - 0.6)**2"
    constraints2 = [
        ConstraintSpec(expr="x1 + x2", sense="<=", rhs=1.0, penalty=100.0)
    ]

    result2 = solve_polynomial_qubo(
        var_specs=vars_cfg2,
        objective=objective2,
        constraints=constraints2,
        sa_options=SAOptions(num_reads=200, sweeps=2500, seed=13),
        quadr_strength=1e3,
        max_iters=20,
        improvement_tol=1e-10,
    )

    print("\n\n=== Test 2: Constrained Quadratic ===")
    print_iteration_log(result2["history"])
    print("\nFinal:")
    print("best_penalized:", result2["best_penalized"])
    print("x:", {k: f"{v:.10f}" for k, v in result2["x"].items()})
    print("J:", result2["J"])
