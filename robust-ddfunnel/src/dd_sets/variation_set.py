# src/dd_sets/variation_set.py
from __future__ import annotations

from typing import Dict

import numpy as np


def build_variation_set_blocks(C: float, T_tilde: int, n: int, m: int) -> Dict[str, np.ndarray]:
    """
    Build block factors for N2 = S2 @ Z2 @ S2.T where

      S2 = [
        [ I_n,  0,    0   ]   # (n  rows)
        [  0,   0,    0   ]   # (n  rows)
        [  0,   0,    0   ]   # (m  rows)
        [  0,  I_n,   0   ]   # (n  rows)
        [  0,   0,   I_n  ]   # (n  rows)
      ],
      Z2 = diag( (C^2 T_tilde^2) I_n,  -I_n,  -I_m )

    Block-row sizes (to match M/N1): [n, n, m, n, n].
    Column-block sizes for Z2: [n, n, m].
    """
    C2T2 = float(C) ** 2 * float(T_tilde) ** 2

    # Center diagonal
    Z2 = np.block(
        [
            [C2T2 * np.eye(n), np.zeros((n, n)), np.zeros((n, m))],
            [np.zeros((n, n)), -np.eye(n), np.zeros((n, m))],
            [np.zeros((m, n)), np.zeros((m, n)), -np.eye(m)],
        ]
    )

    # Selector S2 (5 block-rows: n, n, m, n, n) by (n + n + m) columns
    zeros_nn = np.zeros((n, n))
    zeros_nm = np.zeros((n, m))
    zeros_mn = np.zeros((m, n))
    zeros_mm = np.zeros((m, m))

    S2 = np.vstack(
        [
            np.hstack([np.eye(n), zeros_nn, zeros_nm]),  # row block 1 (n)
            np.hstack([zeros_nn, zeros_nn, zeros_nm]),  # row block 2 (n)
            np.hstack([zeros_mn, zeros_mn, zeros_mm]),  # row block 3 (m)
            np.hstack([zeros_nn, np.eye(n), zeros_nm]),  # row block 4 (n)
            np.hstack([zeros_nn, zeros_nn, zeros_nm]),  # row block 5 (n)
        ]
    )

    return {"S2": S2, "Z2": Z2, "n": int(n), "m": int(m)}


def assemble_N2(blocks: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Convenience: materialize N2 = S2 @ Z2 @ S2.T
    """
    S2 = blocks["S2"]
    Z2 = blocks["Z2"]
    return S2 @ Z2 @ S2.T
