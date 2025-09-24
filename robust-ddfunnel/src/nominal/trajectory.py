# src/nominal/trajectory.py
"""
Nominal trajectory synthesis on the digital twin via discrete LQR around a goal.

Paper alignment:
- Assumption 3 requires a feasible nominal (x_hat*, u_hat*). We compute it here
  using the digital twin only (no plant data), via a constant DLQR gain around
  the goal (x_goal, u_goal). The resulting closed-loop rollout on the twin is
  treated as the nominal sequence.
- We also compute the bounded-increment constant v (max step-to-step change of
  the concatenated (x,u) pair) as needed in the paper.

This module does NOT plot or animate. Use scripts/run_nominal.py (later) and
viz/anim_arm.py for visualization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from models.discretization import discrete_jacobians_fd, make_stepper

Array = np.ndarray


def _gravity_torque(model, q):
    # Prefer a public method if present; fallback to internal _gravity.
    if hasattr(model, "gravity_torque"):
        return np.asarray(model.gravity_torque(q), float)
    if hasattr(model, "_gravity"):
        return np.asarray(model._gravity(q), float)  # type: ignore[attr-defined]
    return np.zeros(model.m, float)


@dataclass(frozen=True)
class NominalLQRSettings:
    """
    LQR design + rollout parameters for the nominal planner.

    Fields:
    -------
    Q, R, Qf     : state/input/terminal cost matrices. Shapes: (n,n), (m,m), (n,n).
    N            : horizon length; nominal states length is N+1.
    x_goal       : (n,) target state
    u_goal       : (m,) equilibrium input
    u_low        : (m,) input lower bound
    u_high       : (m,) input upper bound
    riccati_tol  : tolerance for DARE fixed-point iteration
    riccati_maxit: maximum number of iterations for DARE fixed-point iteration
    fd_eps       : finite-difference step size for Jacobians
    """

    Q: Array
    R: Array
    Qf: Array
    N: int
    x_goal: Array
    u_goal: Optional[Array] = None
    u_low: Optional[Array] = None
    u_high: Optional[Array] = None
    riccati_tol: float = 1e-9
    riccati_maxit: int = 10_000
    fd_eps: float = 1e-6
    ramp_seconds: float = 2.0
    input_smoothing_alpha: float = 0.35
    gravity_feedforward: bool = True

    def validate(self, n: int, m: int) -> None:
        Q, R, Qf = np.asarray(self.Q, float), np.asarray(self.R, float), np.asarray(self.Qf, float)
        if Q.shape != (n, n) or R.shape != (m, m) or Qf.shape != (n, n):
            raise ValueError(f"Q, R, Qf must have shapes ({n},{n}), ({m},{m}), ({n},{n})")
        if self.N <= 1:
            raise ValueError("N must be at least 2")
        xg = np.asarray(self.x_goal, float).reshape(n)
        if xg.size != n:
            raise ValueError(f"x_goal must have length {n}")
        if self.u_goal is not None and np.asarray(self.u_goal, float).size != m:
            raise ValueError(f"u_goal must have length {m}")
        if self.u_low is not None and np.asarray(self.u_low, float).size != m:
            raise ValueError(f"u_low must have length {m}")
        if self.u_high is not None and np.asarray(self.u_high, float).size != m:
            raise ValueError(f"u_high must have length {m}")
        if self.ramp_seconds < 0.0:
            raise ValueError("ramp_seconds must be non-negative")
        if not (0.0 <= self.input_smoothing_alpha <= 1.0):
            raise ValueError("input_smoothing_alpha must be between 0 and 1")


def _solve_dare_iterative(
    A: Array,
    B: Array,
    Q: Array,
    R: Array,
    tol: float = 1e-9,
    maxit: int = 10_000,
) -> Array:
    """
    Simple fixed-point iteration for discrete-time algebraic Riccati equation (DARE):
        P = Q + A^T P A - A^T P B (R + B^T P B)^{-1} B^T P A
    Returns P (PSD)
    """
    P = Q.copy()
    AT, BT = A.T, B.T
    for _ in range(maxit):
        S = R + BT @ P @ B
        K = np.linalg.solve(S, BT @ P @ A)
        P_next = AT @ (P - P @ B @ K) @ A + Q
        if np.linalg.norm(P_next - P, ord="fro") <= tol * (1.0 + np.linalg.norm(P, ord="fro")):
            return P_next
        P = P_next
    return P


def _dlqr_gain(
    A: Array,
    B: Array,
    Q: Array,
    R: Array,
    tol: float = 1e-9,
    maxit: int = 10_000,
) -> Tuple[Array, Array]:
    """
    Infinite-horizon DLQR gain via DARE iteration.
    Returns (K, P) with K = (R + B' P B)^{-1} B' P A
    """
    P = _solve_dare_iterative(A, B, Q, R, tol, maxit)
    S = R + B.T @ P @ B
    K = np.linalg.solve(S, B.T @ P @ A)
    return K, P


def lqr_nominal_plan(
    twin_model,
    settings: NominalLQRSettings,
    x0: Array,
) -> Tuple[Array, Array, Array, float]:
    """
    Plan a nominal trajectory on the digital twin using constant DLQR around (x_goal, u_goal).

    Inputs
    ------
    twin_model : your RobotArm2DOF instance (must expose .n, .m, .p, and be compatible with make_stepper)
    settings   : NominalLQRSettings (validated inside)
    x0         : (n,) initial twin state

    Returns
    -------
    Xhat : (N+1, n) nominal states on the twin
    Uhat : (N, m)   nominal inputs on the twin
    K    : (m, n)   constant LQR gain used for rollout
    v    : float    increment bound (max step-to-step change of concatenated (x,u))

    Notes
    -----
    - Linearizes the **discrete** twin step map at (x_goal, u_goal).
    - Uses infinite-horizon DLQR gain K*, applied for N steps:
        u_k = u_goal + K* (x_goal - x_k)
      (Inputs are clipped to optional [u_low, u_high] if provided.)
    - The resulting (Xhat, Uhat) are treated as the nominal trajectory in the paper.
    """
    n, m = twin_model.n, twin_model.m
    settings.validate(n, m)

    x0 = np.asarray(x0, float).reshape(n)
    x_goal = np.asarray(settings.x_goal, float).reshape(n)
    u_goal = np.zeros(m, float) if settings.u_goal is None else np.asarray(settings.u_goal, float).reshape(m)

    # Discrete stepper and Jacobian at the goal
    step = make_stepper(twin_model.f, twin_model.dt, method=twin_model._integrator_name)
    A, B = discrete_jacobians_fd(step, x_goal, u_goal, settings.fd_eps)

    # DLQR gain at the goal
    K, _P = _dlqr_gain(A, B, settings.Q, settings.R, settings.riccati_tol, settings.riccati_maxit)

    # Rollout on the twin with K
    Xhat = np.empty((settings.N + 1, n), float)
    Uhat = np.empty((settings.N, m), float)
    Xhat[0] = x0

    def _phi(s):
        return 3.0 * s * s - 2.0 * s * s * s

    alpha = float(settings.input_smoothing_alpha)
    Tramp = float(settings.ramp_seconds)
    dt = float(twin_model.dt)

    for k in range(settings.N):
        s = 1.0 if Tramp <= 0 else min(1.0, (k * dt) / Tramp)
        w = _phi(s)

        x_ref = x0 + w * (x_goal - x0)

        u_ff = _gravity_torque(twin_model, x_ref[:2]) if settings.gravity_feedforward else u_goal

        u_raw = u_ff + K @ (x_ref - Xhat[k])

        if k == 0 or alpha >= 1.0:
            u_cmd = u_raw
        elif alpha <= 0.0:
            u_cmd = Uhat[k - 1]
        else:
            u_cmd = (1.0 - alpha) * Uhat[k - 1] + alpha * u_raw

        if settings.u_low is not None:
            u_cmd = np.maximum(u_cmd, np.asarray(settings.u_low, float))
        if settings.u_high is not None:
            u_cmd = np.minimum(u_cmd, np.asarray(settings.u_high, float))
        Uhat[k] = u_cmd
        Xhat[k + 1] = step(Xhat[k], Uhat[k])

    # Increment bound v over state-input pairs
    v = 0.0
    for j in range(settings.N - 1):
        xu_j = np.concatenate((Xhat[j], Uhat[j]))
        xu_j1 = np.concatenate((Xhat[j + 1], Uhat[j + 1] if j + 1 < settings.N else u_goal))
        v = max(v, np.linalg.norm(xu_j1 - xu_j, ord=2))

    return Xhat, Uhat, K, v
