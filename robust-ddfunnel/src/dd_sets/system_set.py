# src/dd_sets/system_set.py
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

Array = np.ndarray


def _validate_data_shapes(H: Array, H_plus: Array, Xi: Array) -> Tuple[int, int, int]:
    """
    Ensure H, H_plus, Xi have consistent shapes:
      H, H_plus : (n, L)
      Xi        : (m, L)
    """
    if H.ndim != 2 or H_plus.ndim != 2 or Xi.ndim != 2:
        raise ValueError("H, H_plus, Xi must be 2D arrays.")
    n, L = H.shape
    n2, L2 = H_plus.shape
    m, L3 = Xi.shape
    if n2 != n or L2 != L:
        raise ValueError(f"H ({H.shape}) and H_plus ({H_plus.shape}) must share (n,L).")
    if L3 != L:
        raise ValueError(f"Xi ({Xi.shape}) must have the same L as H/H_plus ({L}).")
    if L <= 0:
        raise ValueError("Data window length L must be positive.")
    return n, m, L


def _build_N1_factors(H: Array, H_plus: Array, Xi: Array, beta: float) -> Tuple[Array, Array]:
    """
    Build the factorization N1 = S W S^T used in the S-lemma block for data consistency.

    S = [
      [ I_n      ,  H_plus ]      (n x (n+L))
      [ 0_{n x n}, -H      ]      (n x (n+L))
      [ 0_{m x n}, -Xi     ]      (m x (n+L))
      [ 0_{n x n},  0      ]      (n x (n+L))
      [ 0_{n x n},  0      ]      (n x (n+L))
    ],   W = diag( beta * I_n ,  - I_L )

    Shapes:
      - H, H_plus ∈ R^{n x L}, Xi ∈ R^{m x L}
      - S ∈ R^{(4n + m) x (n + L)}
      - W ∈ R^{(n + L) x (n + L)}
      - N1 ∈ R^{(4n + m) x (4n + m)}
    """
    n, m, L = _validate_data_shapes(H, H_plus, Xi)
    I_n = np.eye(n)
    Z_nn = np.zeros((n, n))
    Z_nL = np.zeros((n, L))
    Z_mn = np.zeros((m, n))

    # Stack S by rows
    S_blocks = [
        np.hstack([I_n, H_plus]),  # (n, n+L)
        np.hstack([Z_nn, -H]),  # (n, n+L)
        np.hstack([Z_mn, -Xi]),  # (m, n+L)
        np.hstack([Z_nn, Z_nL]),  # (n, n+L)
        np.hstack([Z_nn, Z_nL]),  # (n, n+L)
    ]
    S = np.vstack(S_blocks)  # (4n+m, n+L)

    # W = diag(beta I_n, - I_L)
    W = np.block(
        [
            [beta * np.eye(n), np.zeros((n, L))],
            [np.zeros((L, n)), -np.eye(L)],
        ]
    )
    return S, W


def dense_N1_from_blocks(S: Array, W: Array) -> Array:
    """Form the dense N1 = S W S^T (handy for debugging/printing)."""
    return S @ W @ S.T


def build_system_set_blocks(H: Array, H_plus: Array, Xi: Array, beta: float) -> Dict[str, Array]:
    """
    Lightweight “block builder” for the data-consistent system set.
    Returns the raw data, dimensions, and the factorization N1 = S W S^T.

    Parameters
    ----------
    H      : (n, L)   stacked state deviations over window
    H_plus : (n, L)   one-step shifted state deviations
    Xi     : (m, L)   stacked input deviations over window
    beta   : float    disturbance energy bound over the window

    Returns
    -------
    dict with:
      - "H", "H_plus", "Xi", "beta"
      - "n", "m", "L"
      - "S_N1", "W_N1"   (so the SDP layer can use the factors directly)
    """
    H = np.asarray(H, dtype=float)
    H_plus = np.asarray(H_plus, dtype=float)
    Xi = np.asarray(Xi, dtype=float)
    n, m, L = _validate_data_shapes(H, H_plus, Xi)
    S, W = _build_N1_factors(H, H_plus, Xi, float(beta))

    return {
        "H": H,
        "H_plus": H_plus,
        "Xi": Xi,
        "beta": float(beta),
        "n": int(n),
        "m": int(m),
        "L": int(L),
        "S_N1": S,
        "W_N1": W,
    }


__all__ = [
    "build_system_set_blocks",
    "dense_N1_from_blocks",
]
