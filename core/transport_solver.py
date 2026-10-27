"""Margin-constrained transport QP."""
from __future__ import annotations

import numpy as np


def solve_margin_qp(g_D, G, margins0, weights, rho_m=0.05,
                    lambda_d=1.0, lambda_xi=10.0, verbose=False):
    import cvxpy as cp

    g_D = np.asarray(g_D, dtype=np.float64)
    G = np.asarray(G, dtype=np.float64)
    margins0 = np.asarray(margins0, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    dim = g_D.shape[0]
    J = G.shape[0]

    if J == 0:
        # 严格安全策略：没有约束就没有方向，强制拒绝
        d_star = np.zeros(dim, dtype=np.float32)
        return {"status": "no_constraints_rejected", "d": d_star,
                "directional_derivative": float(g_D @ d_star),
                "d_norm": float(np.linalg.norm(d_star)),
                "slack_mean": 0.0, "slack_max": 0.0,
                "violation_max": 0.0, "num_constraints": 0}

    eps = rho_m * np.clip(margins0, 0.0, None)
    d = cp.Variable(dim); xi = cp.Variable(J, nonneg=True)
    W = cp.diag(np.clip(weights, 1e-8, None))
    objective = cp.Minimize(g_D @ d + (lambda_d/2) * cp.sum_squares(d)
                            + (lambda_xi/2) * cp.sum_squares(cp.multiply(np.sqrt(np.clip(weights, 1e-8, None)), xi)))
    constraints = [G @ d >= -eps - xi]
    prob = cp.Problem(objective, constraints)

    solved = False
    solver_used = None
    for solver in (cp.ECOS, cp.OSQP):
        try:
            prob.solve(solver=solver, verbose=verbose, warm_start=True)
            solved = True
            solver_used = str(solver)
            break
        except Exception:
            continue

    if not solved or d.value is None:
        return {"status": "solver_failure",
                "solver": solver_used,
                "constraint_residual_max": float("inf"),
                "d": np.zeros(dim, dtype=np.float32),
                "directional_derivative": 0.0, "d_norm": 0.0,
                "slack_mean": 0.0, "slack_max": 0.0,
                "violation_max": float("inf"), "num_constraints": int(J)}

    d_star = d.value.astype(np.float32)
    Gt = G @ d_star.astype(np.float64)
    # A10 fix: use solver's actual xi.value for diagnostics, not reconstructed slack.
    # Reconstructed slack satisfies the constraint by construction and hides violations.
    if xi.value is not None:
        slack = np.maximum(0.0, np.asarray(xi.value, dtype=np.float64).flatten())
    else:
        slack = np.maximum(0.0, -eps - Gt)
    # True residual: how much the constraint Gd + eps + xi >= 0 is violated by solver's d and xi
    residual = np.maximum(0.0, -(Gt + eps + slack))
    status_str = str(prob.status)
    # 映射 status
    if status_str in ("optimal", "optimal_inaccurate"):
        mapped_status = status_str
    elif status_str in ("infeasible", "infeasible_inaccurate"):
        mapped_status = "infeasible"
    else:
        mapped_status = status_str
    return {"status": mapped_status, "solver": solver_used,
            "objective": float(prob.value),
            "d": d_star, "directional_derivative": float(g_D @ d_star.astype(np.float64)),
            "d_norm": float(np.linalg.norm(d_star)),
            "slack_mean": float(slack.mean()), "slack_max": float(slack.max()),
            "constraint_residual_max": float(residual.max()),
            "violation_max": float(slack.max()), "num_constraints": int(J)}
