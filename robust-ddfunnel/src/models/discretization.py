# src/robust_ddfunnel/models/discretization.py
"""
Fixed-step discretization utilities for continuous-time dynamics xdot = f(x, u).

- Inputs are held constant over [t, t+dt] (sample-and-hold).
- Provides Euler and RK4 integrators.
- Includes a factory to create a discrete-time step map F(x, u).
- Optional finite-difference Jacobians for diagnostics (not used by the main algorithm).

This module is model-agnostic and does not implement any controller logic.
"""

from __future__ import annotations

from typing import Callable, Literal, Tuple

import numpy as np

Array = np.ndarray
Dynamics = Callable[[Array, Array], Array]
IntegratorName = Literal["rk4", "euler"]


def euler_step(f: Dynamics, x: Array, u: Array, dt: float) -> Array:
    """
    One forward-Euler step with constant input over [t, t+dt].

    Parameters
    ----------
    f   :   callable, xdot = f(x, u)
    x   :   (n,) array_like
    u   :   (m,) array_like
    dt  :   float

    Returns
    -------
    x_next : (n,) ndarray
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    u = np.asarray(u, dtype=float).reshape(-1)
    return x + dt * f(x, u)


def rk4_step(f: Dynamics, x: Array, u: Array, dt: float) -> Array:
    """
    One classical 4th-order Runge-Kutta step with constant input over [t, t+dt].

    Parameters
    ----------
    f   :   callable, xdot = f(x, u)
    x   :   (n,) array_like
    u   :   (m,) array_like
    dt  :   float

    Returns
    -------
    x_next : (n,) ndarray
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    u = np.asarray(u, dtype=float).reshape(-1)

    k1 = f(x, u)
    k2 = f(x + 0.5 * dt * k1, u)
    k3 = f(x + 0.5 * dt * k2, u)
    k4 = f(x + dt * k3, u)

    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def make_stepper(f: Dynamics, dt: float, method: IntegratorName = "rk4") -> Dynamics:
    """
    Factory that returns a discrete-time step map F(x, u) for the given dynamics f(x, u).

    Parameters
    ----------
    f : callable, xdot = f(x, u)
    dt : float
    method : "rk4" (default) or "euler"

    Returns
    -------
    F : callable, x_next = F(x, u)
    """
    if method == "rk4":
        return lambda x, u: rk4_step(f, x, u, dt)
    elif method == "euler":
        return lambda x, u: euler_step(f, x, u, dt)
    else:
        raise ValueError(f"Invalid integrator: {method}")


def rollout(step: Callable[[Array, Array], Array], x0: Array, U: Array) -> Array:
    """
    Roll out a trajectory given a discrete step map and an input sequence.

    Parameters
    ----------
    step : callable, x_{k+1} = step(x_k, u_k)
    x0 : (n,) array_like, initial state
    U : (T, m) array_like, input sequence

    Returns
    -------
    X : (T+1, n) ndarray, state trajectory (k=0, ..., T)
    """
    x = np.asarray(x0, dtype=float).reshape(-1)
    U = np.asarray(U, dtype=float)
    T = U.shape[0]
    n = x.shape[0]

    X = np.empty((T + 1, n), dtype=float)
    X[0] = x
    for k in range(T):
        X[k + 1] = step(X[k], U[k])
    return X


def discrete_jacobians_fd(
    step: Callable[[Array, Array], Array],
    x: Array,
    u: Array,
    eps: float = 1e-6,
) -> Tuple[Array, Array]:
    """
    Finite-difference Jacobians of the *discrete* step map x_{k+1} = step(x_k, u_k):
        A = ∂F/∂x,  B = ∂F/∂u
    This is only for diagnostics; the main algorithm remains data-driven.

    Parameters
    ----------
    step : callable, x_{k+1} = step(x_k, u_k)
    x : (n,) array_like
    u : (m,) array_like
    eps : float

    Returns
    -------
    (A, B) : (n, n) ndarray, (n, m) ndarray
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    u = np.asarray(u, dtype=float).reshape(-1)

    n, m = x.size, u.size
    A = np.zeros((n, n), dtype=float)
    B = np.zeros((n, m), dtype=float)

    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps
        Fp = step(x + dx, u)
        Fm = step(x - dx, u)
        A[:, i] = (Fp - Fm) / (2.0 * eps)

    for j in range(m):
        du = np.zeros(m)
        du[j] = eps
        Fp = step(x, u + du)
        Fm = step(x, u - du)
        B[:, j] = (Fp - Fm) / (2.0 * eps)

    return A, B


__all__ = [
    "discrete_jacobians_fd",
    "euler_step",
    "make_stepper",
    "rk4_step",
    "rollout",
]
