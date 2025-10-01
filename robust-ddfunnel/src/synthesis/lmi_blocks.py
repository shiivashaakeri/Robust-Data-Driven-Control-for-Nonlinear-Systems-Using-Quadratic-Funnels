# src/synthesis/lmi_blocks.py
from __future__ import annotations

import numpy as np

try:
    import cvxpy as cp  # optional; only needed for build_M_cvxpy

    _HAS_CVXPY = True
except Exception:  # pragma: no cover
    cp = None
    _HAS_CVXPY = False


def _check_shapes(Q, Y):
    Q = np.asarray(Q) if isinstance(Q, np.ndarray) else Q
    n = Q.shape[0]
    if Q.shape != (n, n):
        raise ValueError(f"Q must be square (n,n); got {Q.shape}")
    m = Y.shape[0]
    if Y.shape != (m, n):
        raise ValueError(f"Y must be (m,n); got {Y.shape}")
    return n, m


def build_M_numpy(Q: np.ndarray, Y: np.ndarray, alpha: float, nu: float = 0.0) -> np.ndarray:
    n, m = _check_shapes(Q, Y)
    Znn = np.zeros((n, n))
    Znm = np.zeros((n, m))
    Zmn = np.zeros((m, n))
    Zmm = np.zeros((m, m))

    S = np.block(
        [
            [alpha * Q - nu * np.eye(n), Znn, Znm, Znn, Znm, Znn],
            [Znn, -Q, -Y.T, -Q, -Y.T, Znn],
            [Zmn, -Y, Zmm, -Y, Zmm, Y],
            [Znn, -Q, -Y.T, -Q, -Y.T, Znn],
            [Zmn, -Y, Zmm, -Y, Zmm, Y],
            [Znn, Znn, Y.T, Znn, Y.T, Q],
        ]
    )
    return S


def build_M_cvxpy(Q, Y, alpha: float, nu=None):
    if not _HAS_CVXPY:
        raise RuntimeError("cvxpy is not available; install it to use build_M_cvxpy.")

    n, m = _check_shapes(Q, Y)

    Znn = cp.Constant(np.zeros((n, n)))
    Znm = cp.Constant(np.zeros((n, m)))
    Zmn = cp.Constant(np.zeros((m, n)))
    Zmm = cp.Constant(np.zeros((m, m)))

    nu_term = Znn if nu is None else nu * cp.Constant(np.eye(n))

    S = cp.bmat(
        [
            [alpha * Q - nu_term, Znn, Znm, Znn, Znm, Znn],
            [Znn, -Q, -Y.T, -Q, -Y.T, Znn],
            [Zmn, -Y, Zmm, -Y, Zmm, Y],
            [Znn, -Q, -Y.T, -Q, -Y.T, Znn],
            [Zmn, -Y, Zmm, -Y, Zmm, Y],
            [Znn, Znn, Y.T, Znn, Y.T, Q],
        ]
    )
    return S
