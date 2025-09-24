# src/nominal/max_elliposoids.py
"""
Max-volume (centered) ellipsoids for state/input boxes.

This module computes Q_max and R_max as *centered* ellipsoids that fit inside
(quasi-)polytopic constraint sets around a given center. For your box
constraints, the "quasi-polytope" is exact, so the problems reduce to

    maximize   logdet(Q)
    subject to sqrt(a_i^T Q a_i) + a_i^T c <= b_i,   i=1..m
               Q ⪰ 0,   (optional) Q ⪯ (cap)^2 I

with halfspaces A x ≤ b and center c (x_nominal or u_nominal). We solve these
as an SDP with a PSD matrix variable Q and constraints a_i^T Q a_i ≤ margin_i^2.

Key API
-------
- box_to_halfspaces(x_min, x_max) -> (A, b)
- max_vol_centered_ellipsoid(A, b, center, *, ball_cap=None, solver=None, verbose=False) -> Q
- qmax_for_state_box(center_x, x_min, x_max, ..., solver=None) -> Q
- rmax_for_input_box(center_u, u_min, u_max, ..., solver=None) -> R
- qmax_rmax_time_invariant(Xhat, Uhat, x_min, x_max, u_min, u_max, ..., solver=None)
    Builds *time-invariant* Q_max, R_max that are feasible for *all* centers
    along the nominal (uses the worst margin over k).

If CVXPY is unavailable, a conservative diagonal fallback is used *for box
constraints* only (it builds Q = diag(margins_min^2)).
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np

try:
    import cvxpy as cp

    _HAS_CVXPY = True
except Exception:
    cp = None
    _HAS_CVXPY = False

Array = np.ndarray

# ---------------------------
# Halfspaces from boxes
# ---------------------------


def box_to_halfspaces(x_min: Array, x_max: Array) -> Tuple[Array, Array]:
    """
    Convert a box {x_min} <= x <= {x_max} to halfspaces A x <= b with rows [I; -I].

    Returns
    -------
    A : (2n, n)
    b : (2n,)
        Stacked as:
        A = [I; -I], b = [x_max; -x_min]
    """

    x_min = np.asarray(x_min, float).reshape(-1)
    x_max = np.asarray(x_max, float).reshape(-1)

    if x_min.shape != x_max.shape:
        raise ValueError("x_min and x_max must have the same shape")
    if np.any(x_min >= x_max):
        raise ValueError("x_min must be less than x_max")

    n = x_min.size
    A = np.vstack([np.eye(n), -np.eye(n)])
    b = np.hstack([x_max, -x_min])

    return A, b


# ---------------------------
# Core solver
# ---------------------------


def _margins_for_centers(A: Array, b: Array, centers: Array) -> Array:
    """
    For a set of centers {c_k} compute the worst feasible margin per halfspace:
        margin_i = min_k (b_i - a_i^T c_k)

    Returns: margin shape (m,), clipped at a tiny positive epsilon
    """
    A = np.asarray(A, float)
    b = np.asarray(b, float).reshape(-1)
    C = np.atleast_2d(np.asarray(centers, float))
    vals = b[None, :] - C @ A.T
    margins = np.min(vals, axis=0)

    return np.maximum(margins, 1e-12)


def max_vol_centered_ellipsoid(  # noqa: C901
    A: Array,
    b: Array,
    center: Array | Iterable[Array],
    *,
    ball_cap: float | None = None,
    solver: Optional[str] = None,
    verbose: bool = False,
) -> Array:
    """
    Solve: maximize logdet(Q)  s.t. sqrt(a_i^T Q a_i) + a_i^T c <= b_i
                              Q ⪰ 0, (optional) Q ⪯ (ball_cap)^2 I

    Parameters
    ----------
    A, b : halfspaces A x ≤ b (A: (m,n), b: (m,))
    center : (n,) or (K,n)
        If multiple centers are provided, we enforce feasibility for *all* by
        using the worst margin per halfspace: min_k(b_i - a_i^T c_k).
    ball_cap : float | None
        If given, enforces Q ⪯ (ball_cap)^2 I (i.e., ellipsoid inside an l2-ball).
    solver : str | None
        CVXPY solver name (e.g., "ECOS", "SCS", "MOSEK"). If None, CVXPY picks.
    verbose : bool
        Pass-through to CVXPY's solve().

    Returns
    -------
    Q : (n,n) SPD/PSD matrix for the centered ellipsoid {c + z : z^T Q^{-1} z ≤ 1}.

    Fallback
    --------
    If CVXPY is not available, returns a conservative diagonal Q using only
    the axis-aligned rows if A is [I; -I]. Otherwise raises RuntimeError.
    """
    A = np.asarray(A, float)
    b = np.asarray(b, float).reshape(-1)

    n = A.shape[1]
    if A.shape[0] != b.size:
        raise ValueError("A and b must have consistent shapes.")

    C = np.asarray(center, float)
    if C.ndim == 1:
        C = C.reshape(1, -1)
    if C.shape[1] != n:
        raise ValueError("center must have shape (n,) or (K,n).")

    margins = _margins_for_centers(A, b, C)
    if np.any(margins <= 0):
        raise ValueError("Center(s) not strictly inside polytope.")

    # If we have CVXPY, solve the SDP with Q as a PSD variable

    if _HAS_CVXPY:
        Q = cp.Variable((n, n), PSD=True)
        cons = []
        for i in range(A.shape[0]):
            ai = A[i, :].reshape(-1, 1)
            cons.append(cp.quad_form(ai, Q) <= float(margins[i] ** 2))
        if ball_cap is not None and ball_cap > 0:
            cons.append(Q << float(ball_cap**2) * cp.eye(n))

        # Strict interior assumption => opt is bounded; add tiny ridge for stability is desired
        objective = cp.Maximize(cp.log_det(Q))
        prob = cp.Problem(objective, cons)
        prob.solve(solver=solver, verbose=verbose)
        if Q.value is None:
            raise RuntimeError(f"Ellipsoid SDP infeasible/failed (status={prob.status}).")

        Qval = 0.5 * (Q.value + Q.value.T)
        return Qval

    I = np.eye(n)
    if A.shape == (2 * n, n) and np.allclose(A[:n, :], I) and np.allclose(A[n:, :], -I):
        # margins are [x_max - c_i, c_i - x_min] per i => take the min of the pair
        m_up = margins[:n]
        m_lo = margins[n:]
        m = np.minimum(m_up, m_lo)
        if ball_cap is not None:
            m = np.minimum(m, float(ball_cap))
        return np.diag(m**2)

    raise RuntimeError("CVXPY not found and A is not axis aligned box; cannot compute max-volume ellipsoid.")


# ---------------------------
# Convenience wrappers
# ---------------------------


def qmax_for_state_box(
    center_x: Array | Iterable[Array],
    x_min: Array,
    x_max: Array,
    *,
    ball_cap: float | None = None,
    solver: Optional[str] = None,
    verbose: bool = False,
) -> Array:
    """
    Q_max centered at center_x for the state box [x_min, x_max].
    """
    A, b = box_to_halfspaces(x_min, x_max)
    return max_vol_centered_ellipsoid(A, b, center_x, ball_cap=ball_cap, solver=solver, verbose=verbose)


def rmax_for_input_box(
    center_u: Array | Iterable[Array],
    u_min: Array,
    u_max: Array,
    *,
    ball_cap: float | None = None,
    solver: Optional[str] = None,
    verbose: bool = False,
) -> Array:
    """
    R_max centered at center_u for the input box [u_min, u_max].
    """
    A, b = box_to_halfspaces(u_min, u_max)
    return max_vol_centered_ellipsoid(A, b, center_u, ball_cap=ball_cap, solver=solver, verbose=verbose)


def qmax_rmax_time_invariant(
    Xhat: Array,
    Uhat: Array,
    x_min: Array,
    x_max: Array,
    u_min: Array,
    u_max: Array,
    *,
    solver: Optional[str] = None,
    verbose: bool = False,
    x_ball_cap: float | None = None,
    u_ball_cap: float | None = None,
) -> Tuple[Array, Array]:
    """
    Time-invariant ellipsoids feasible for *every* nominal center along the trajectory.

    We enforce the worst margin over k:
        margin_i ← min_k (b_i - a_i^T c_k)
    which yields a single Q_max and R_max that satisfy A x ≤ b for all centers.

    Parameters
    ----------
    Xhat : (N+1,n) nominal states
    Uhat : (N,  m) nominal inputs
    x_min, x_max : (n,)
    u_min, u_max : (m,)
    solver : CVXPY solver name or None
    x_ball_cap, u_ball_cap : optional l2-ball caps on Q and R (radius in units of x/u)

    Returns
    -------
    Q_max : (n,n)
    R_max : (m,m)
    """
    Xhat = np.asarray(Xhat, float)
    Uhat = np.asarray(Uhat, float)
    if Xhat.ndim != 2 or Uhat.ndim != 2 or Xhat.shape[0] != Uhat.shape[0] + 1:
        raise ValueError("Shapes must be Xhat: (N+1,n), Uhat: (N,m).")

    Ax, bx = box_to_halfspaces(x_min, x_max)
    Au, bu = box_to_halfspaces(u_min, u_max)

    Q = max_vol_centered_ellipsoid(Ax, bx, Xhat[:-1, :], ball_cap=x_ball_cap, solver=solver, verbose=verbose)
    R = max_vol_centered_ellipsoid(Au, bu, Uhat, ball_cap=u_ball_cap, solver=solver, verbose=verbose)
    return Q, R


def qmax_rmax_per_time(
    Xhat: Array,
    Uhat: Array,
    x_min: Array,
    x_max: Array,
    u_min: Array,
    u_max: Array,
    *,
    solver: Optional[str] = None,
    verbose: bool = False,
    x_ball_cap: float | None = None,
    u_ball_cap: float | None = None,
) -> Tuple[list[Array], list[Array]]:
    """
    Per-time ellispoids (Q_max[k], R_max[k]) centered at (Xhat[k], Uhat[k]).
    Useful if you want the loosest possible shapes at each iteration.
    """
    Xhat = np.asarray(Xhat, float)
    Uhat = np.asarray(Uhat, float)
    if Xhat.ndim != 2 or Uhat.ndim != 2 or Xhat.shape[0] != Uhat.shape[0] + 1:
        raise ValueError("Shapes must be Xhat: (N+1,n), Uhat: (N,m).")

    Ax, bx = box_to_halfspaces(x_min, x_max)
    Au, bu = box_to_halfspaces(u_min, u_max)

    Qs, Rs = [], []
    for k in range(Uhat.shape[0]):
        Q = max_vol_centered_ellipsoid(Ax, bx, Xhat[k, :], ball_cap=x_ball_cap, solver=solver, verbose=verbose)
        R = max_vol_centered_ellipsoid(Au, bu, Uhat[k, :], ball_cap=u_ball_cap, solver=solver, verbose=verbose)
        Qs.append(Q)
        Rs.append(R)
    return Qs, Rs

def qmax_time_varying_box(
    Xhat: np.ndarray,
    x_low: np.ndarray,
    x_high: np.ndarray,
    *,
    x_ball_cap: float | None = None,
) -> np.ndarray:
    """
    Time-varying Q_k for box constraints.
    Returns Q_t with shape (N+1, n, n), diagonal per time step.
    """
    Xhat = np.asarray(Xhat, float)
    x_low = np.asarray(x_low, float).reshape(-1)
    x_high = np.asarray(x_high, float).reshape(-1)
    Np1, n = Xhat.shape
    Q_t = np.zeros((Np1, n, n), dtype=float)

    for k in range(Np1):
        x = Xhat[k]
        slack = np.minimum(x_high - x, x - x_low)
        if x_ball_cap is not None:
            slack = np.minimum(slack, x_ball_cap)
        slack = np.maximum(slack, 0.0)
        Q_t[k] = np.diag(slack**2)
    return Q_t


def rmax_time_varying_box(
    Uhat: np.ndarray,
    u_low: np.ndarray,
    u_high: np.ndarray,
    *,
    u_ball_cap: float | None = None,
) -> np.ndarray:
    """
    Time-varying R_k for box constraints.
    Returns R_t with shape (N, m, m), diagonal per time step.
    """
    Uhat = np.asarray(Uhat, float)
    u_low = np.asarray(u_low, float).reshape(-1)
    u_high = np.asarray(u_high, float).reshape(-1)
    N, m = Uhat.shape
    R_t = np.zeros((N, m, m), dtype=float)

    for k in range(N):
        u = Uhat[k]
        slack = np.minimum(u_high - u, u - u_low)
        if u_ball_cap is not None:
            slack = np.minimum(slack, u_ball_cap)
        slack = np.maximum(slack, 0.0)
        R_t[k] = np.diag(slack**2)
    return R_t

__all__ = [
    "box_to_halfspaces",
    "max_vol_centered_ellipsoid",
    "qmax_for_state_box",
    "qmax_rmax_per_time",
    "qmax_rmax_time_invariant",
    "qmax_time_varying_box",
    "rmax_for_input_box",
    "rmax_time_varying_box",
]
