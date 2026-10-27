"""Hard-margin trust-region QP solver for HFCST.

Uses CVXPY when available; otherwise SciPy SLSQP solves the same convex QP.
Failure is fail-closed: return a zero update, never an unconstrained direction.
"""
from __future__ import annotations
import numpy as np


def solve_margin_qp(gradient, G, margins, rho_m=0.05, lambda_d=0.01,
                    trust_radius=0.15, active_mask=None, solver_preference=None):
    g = np.asarray(gradient, dtype=np.float64).reshape(-1)
    G = np.asarray(G, dtype=np.float64).reshape(-1, len(g))
    margins = np.asarray(margins, dtype=np.float64).reshape(-1)
    zero = np.zeros_like(g)
    base = {"status": "rejected", "solver": "none", "d": zero.astype(np.float32),
            "directional_derivative": 0.0, "d_norm": 0.0,
            "constraint_residual_max": 0.0, "num_constraints": int(len(margins)),
            "reason": "not_solved"}
    if not np.isfinite(g).all() or not np.isfinite(G).all() or not np.isfinite(margins).all():
        base.update(status="nonfinite_rejected", reason="nonfinite_input")
        return base
    if len(margins) == 0:
        base.update(status="no_constraints_rejected", reason="no_valid_margin_constraints")
        return base
    if G.shape[0] != len(margins):
        raise ValueError("G row count and margins length differ")
    if active_mask is not None:
        active = np.asarray(active_mask, dtype=bool).reshape(-1)
        if len(active) != len(g):
            raise ValueError("active_mask dimension mismatch")
        g = g.copy(); g[~active] = 0.0
        G = G.copy(); G[:, ~active] = 0.0
    # The right hand side is non-positive, so d=0 is always feasible.
    lower = -float(rho_m) * np.maximum(margins, 0.0)
    radius = float(trust_radius)
    if radius <= 0:
        base.update(status="invalid_trust_radius", reason="trust_radius_must_be_positive")
        return base

    try:
        import cvxpy as cp
        d = cp.Variable(len(g))
        constraints = [G @ d >= lower, cp.norm(d, 2) <= radius]
        if active_mask is not None:
            inactive = np.flatnonzero(~np.asarray(active_mask, dtype=bool))
            if len(inactive): constraints.append(d[inactive] == 0)
        objective = cp.Minimize(g @ d + 0.5 * float(lambda_d) * cp.sum_squares(d))
        prob = cp.Problem(objective, constraints)
        installed = set(cp.installed_solvers())
        prefs = solver_preference or ["CLARABEL", "OSQP", "SCS"]
        chosen = next((s for s in prefs if s in installed), None)
        if chosen is None:
            raise RuntimeError("cvxpy installed but no supported solver available")
        prob.solve(solver=chosen, warm_start=False, verbose=False)
        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or d.value is None:
            base.update(status=str(prob.status), solver=chosen, reason="cvxpy_not_optimal")
            return base
        sol = np.asarray(d.value, dtype=np.float64).reshape(-1)
        solver_name = chosen
    except ImportError:
        solver_name = "scipy_slsqp"
        try:
            from scipy.optimize import minimize, LinearConstraint
            constraint = LinearConstraint(G, lower, np.full_like(lower, np.inf))
            constraints = [constraint, {"type": "ineq", "fun": lambda z: radius**2 - float(z @ z),
                                        "jac": lambda z: -2.0 * z}]
            bounds = None
            if active_mask is not None:
                mask = np.asarray(active_mask, dtype=bool)
                bounds = [(None, None) if a else (0.0, 0.0) for a in mask]
            result = minimize(lambda z: float(g @ z + 0.5 * lambda_d * (z @ z)),
                              np.zeros_like(g), jac=lambda z: g + lambda_d * z,
                              constraints=constraints, bounds=bounds, method="SLSQP",
                              options={"maxiter": 500, "ftol": 1e-10, "disp": False})
            sol = np.asarray(result.x, dtype=np.float64)
            if not result.success:
                base.update(status="scipy_failed", solver=solver_name, reason=str(result.message))
                return base
        except Exception as exc:
            base.update(status="solver_error", solver=solver_name, reason=f"{type(exc).__name__}: {exc}")
            return base
    except Exception as exc:
        base.update(status="solver_error", solver="cvxpy", reason=f"{type(exc).__name__}: {exc}")
        return base

    residual = lower - G @ sol
    violation = float(np.maximum(residual, 0.0).max()) if len(residual) else 0.0
    norm = float(np.linalg.norm(sol))
    if (not np.isfinite(sol).all() or norm > radius + 1e-5 or violation > 1e-4):
        base.update(status="postcheck_rejected", solver=solver_name, reason="post_solve_feasibility_check",
                    constraint_residual_max=violation)
        return base
    if active_mask is not None and np.any(np.abs(sol[~np.asarray(active_mask, dtype=bool)]) > 1e-8):
        base.update(status="postcheck_rejected", solver=solver_name, reason="inactive_parameter_moved")
        return base
    base.update(status="optimal", solver=solver_name, d=sol.astype(np.float32),
                directional_derivative=float(np.dot(g, sol)), d_norm=norm,
                constraint_residual_max=violation, reason="ok")
    return base
