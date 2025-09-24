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
    """Validate shapes and return (n, m). Q: (n,n), Y: (m,n)."""
    Q = np.asarray(Q) if isinstance(Q, np.ndarray) else Q
    n = Q.shape[0]
    if Q.shape != (n, n):
        raise ValueError(f"Q must be square (n,n); got {Q.shape}")
    m = Y.shape[0]
    if Y.shape != (m, n):
        raise ValueError(f"Y must be (m,n); got {Y.shape}")
    return n, m


def build_M_numpy(Q: np.ndarray, Y: np.ndarray, alpha: float) -> np.ndarray:
    """
    Assemble the numeric M block from the paper:

        M = [[ alpha * Q,  0,  0,  0,  0],
             [  0,  0,  0,  0,  Q],
             [  0,  0,  0,  0,  Y],
             [  0,  0,  0,  0,  Q],
             [  0,  Q, Yᵀ,  Q,  Q]]

    Block-row/column sizes = [n, n, m, n, n] with Q∈R^{n x n}, Y∈R^{m x n}.
    """
    n, m = _check_shapes(Q, Y)
    Znn = np.zeros((n, n))
    Znm = np.zeros((n, m))
    Zmn = np.zeros((m, n))
    Zmm = np.zeros((m, m))

    M = np.block(
        [
            # row 1 (n)
            [alpha * Q, Znn, Znm, Znn, Znn],
            # row 2 (n)
            [Znn, Znn, Znm, Znn, Q],
            # row 3 (m)
            [Zmn, Zmn, Zmm, Zmn, Y],
            # row 4 (n)
            [Znn, Znn, Znm, Znn, Q],
            # row 5 (n)
            [Znn, Q, Y.T, Q, Q],
        ]
    )
    return M


def build_M_cvxpy(Q, Y, alpha: float):
    """
    Assemble the CVXPY M block using cp.bmat with the same layout
    (use this in the SDP; Q and Y can be Variables/Params/Expressions).

    Requires cvxpy. Raises if cvxpy isn't available.
    """
    if not _HAS_CVXPY:
        raise RuntimeError("cvxpy is not available; install it to use build_M_cvxpy.")

    n, m = _check_shapes(Q, Y)

    Znn = cp.Constant(np.zeros((n, n)))
    Znm = cp.Constant(np.zeros((n, m)))
    Zmn = cp.Constant(np.zeros((m, n)))
    Zmm = cp.Constant(np.zeros((m, m)))

    M = cp.bmat(
        [
            # row 1 (n)
            [alpha * Q, Znn, Znm, Znn, Znn],
            # row 2 (n)
            [Znn, Znn, Znm, Znn, Q],
            # row 3 (m)
            [Zmn, Zmn, Zmm, Zmn, Y],
            # row 4 (n)
            [Znn, Znn, Znm, Znn, Q],
            # row 5 (n)
            [Znn, Q, Y.T, Q, Q],
        ]
    )
    return M
