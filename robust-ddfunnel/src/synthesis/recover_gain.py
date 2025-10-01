# src/synthesis/recover_gain.py

from __future__ import annotations

import numpy as np


def recover_gain(Y: np.ndarray, Q: np.ndarray, *, jitter: float = 1e-9) -> np.ndarray:
    """
    Recover K from Y = KQ without explicitly inverting Q.

    Solves X Q = Y => (Q^T) X^T = Y^T => X^T = solve(Q^T, Y^T)
    Uses a small diagonal jitter if needed, and falls back to pinv.

    Parameters
    ----------
    Y : (m, n) array
        Decision variable Y = K Q from SDP
    Q : (n, n) array
        Decision variable Q from SDP
    jitter : float, optional
        Tolerance for small diagonal jitter

    Returns
    -------
    K : (m, n) array
        Feedback gain.
    """
    Y = np.asarray(Y, float)
    Q = np.asarray(Q, float)
    m, n = Y.shape
    if Q.shape != (n, n):
        raise ValueError(f"Q must be (n,n); got {Q.shape}")
    if Y.shape != (m, n):
        raise ValueError(f"Y must be (m,n); got {Y.shape}")

    Qs = 0.5 * (Q + Q.T)
    try:
        KT = np.linalg.solve(Qs.T, Y.T)
        return KT.T
    except np.linalg.LinAlgError:
        try:
            KT = np.linalg.solve((Qs + jitter * np.eye(n)).T, Y.T)
            return KT.T
        except np.linalg.LinAlgError:
            return Y @ np.linalg.pinv(Qs, rcond=1e-9)
