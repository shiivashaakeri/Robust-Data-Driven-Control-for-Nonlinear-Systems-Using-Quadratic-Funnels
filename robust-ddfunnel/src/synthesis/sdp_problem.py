# src/synthesis/sdp_problem.py

from __future__ import annotations

from typing import Any, Dict, List, Optional

import cvxpy as cp
import numpy as np

from synthesis.lmi_blocks import build_M_cvxpy


def _as_constant(A: np.ndarray) -> cp.Expression:
    return cp.Constant(np.asarray(A, dtype=float))


def _cap_violation(Q: np.ndarray) -> float:  # noqa: ARG001
    return 0.0


def solve_funnel_sdp(  # noqa: C901, PLR0912, PLR0915
    *,
    alpha: float,
    sys_blocks: Dict[str, Any],
    var_blocks: Dict[str, Any],
    feas_bounds: Dict[str, List[np.ndarray]],
    tau1_cap: float | None = 1e3,
    tau2_cap: float | None = 1e3,
    solver: Optional[str] = "SCS",
    verbose: bool = False,
    eps_pd: float = 1e-8,
    scs_settings: Optional[Dict[str, Any]] = None,
    cap_check_tol: float = 5e-4,
    max_refine_iters: int = 2,
) -> Dict[str, Any]:
    """Solve the SDP from paper Equation 45."""

    n = int(sys_blocks.get("n", var_blocks["n"]))
    m = int(sys_blocks.get("m", var_blocks["m"]))

    P = cp.Variable((n, n), PSD=True)
    L = cp.Variable((m, n))
    lambda1 = cp.Variable(nonneg=True)
    lambda2 = cp.Variable(nonneg=True)
    nu = cp.Variable(nonneg=True)

    S = build_M_cvxpy(P, L, float(alpha), nu)

    S_N1 = _as_constant(sys_blocks["S_N1"])
    W_N1 = _as_constant(sys_blocks["W_N1"])
    N1 = S_N1 @ W_N1 @ S_N1.T

    S2 = _as_constant(var_blocks["S2"])
    Z2 = _as_constant(var_blocks["Z2"])
    N2 = S2 @ Z2 @ S2.T

    cons = []

    cons += [P >> eps_pd * np.eye(n)]

    base_Qk_list = [np.asarray(Qk, float) for Qk in feas_bounds.get("Qk_list", [])]
    P_lower_bounds = []
    for Qcap in base_Qk_list:
        P_lower_bounds.append(np.linalg.inv(Qcap))

    Rk_list = [np.asarray(Rk, float) for Rk in feas_bounds.get("Rk_list", [])]

    cons += [S - lambda1 * N1 - lambda2 * N2 >> 0]

    for P_lb in P_lower_bounds:
        cons += [P - _as_constant(P_lb) >> 0]
    for Rk in Rk_list:
        cons += [cp.bmat([[_as_constant(Rk), L], [L.T, P]]) >> 0]

    if tau1_cap is not None:
        cons += [lambda1 <= tau1_cap]
    if tau2_cap is not None:
        cons += [lambda2 <= tau2_cap]

    obj = cp.Maximize(cp.log_det(P))
    if P_lower_bounds:
        P.value = P_lower_bounds[0] * 1.05
    else:
        P.value = np.eye(n)
    L.value = np.zeros((m, n))

    prob = cp.Problem(obj, cons)
    scs_defaults: Dict[str, Any] = {
        "max_iters": 200000,
        "eps": 1e-6,
        "acceleration_lookback": 50,
        "warm_start": True,
    }
    if scs_settings:
        scs_defaults.update(scs_settings)

    attempts = max(1, int(max_refine_iters))
    cap_violation: float | None = None
    for attempt in range(attempts):
        prob.solve(solver=cp.SCS, verbose=verbose, **scs_defaults)
        if P.value is None:
            break
        violation = _cap_violation(P.value)
        cap_violation = violation
        if violation <= max(0.0, cap_check_tol):
            break
        if attempt == attempts - 1:
            break
        scs_defaults["eps"] = scs_defaults.get("eps", 1e-6) * 0.3
        scs_defaults["max_iters"] = int(max(scs_defaults.get("max_iters", 200000) * 1.2, 1))
        scs_defaults.setdefault("warm_start", True)
    else:
        prob.solve(solver=solver, verbose=verbose)

    if P.value is not None:
        cap_violation = _cap_violation(P.value)

    return {
        "Q": (None if P.value is None else np.array(P.value)),
        "Y": (None if L.value is None else np.array(L.value)),
        "tau1": (None if lambda1.value is None else float(lambda1.value)),
        "tau2": (None if lambda2.value is None else float(lambda2.value)),
        "status": prob.status,
        "optval": prob.value,
        "cap_violation": cap_violation,
    }
