# src/core/data_stack.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

Array = np.ndarray


@dataclass
class DataStack:
    """
    Segment-level buffer for deviation data.

    It accumulates triples (eta(k), xi(k), eta(k+1)) during the data window
    and exports them as the matrices:
        H      = [eta(k_i^D) ... eta(k_{i+1}-1)] in R^{n x L}
        H_plus = [eta(k_i^D+1) ... eta(k_{i+1})] in R^{n x L}
        Xi     = [xi(k_i^D) ... xi(k_{i+1}-1)] in R^{m x L}

    Notes
    -----
    - This class does 'no' time bookkeeping; use core.segments.SegmentClock instead.
    - to decide when to push and when to reset.
    - Shapes are enforced at push-time to catch bugs early.
    """

    n: int
    m: int
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        self.etas: list[Array] = []
        self.xis: list[Array] = []
        self.etas_next: list[Array] = []
        self._rng = np.random.default_rng(self.seed)

    # ----------------
    # Streaming API
    # ----------------

    def push(self, eta: Array, xi: Array, eta_next: Array) -> None:
        """
        Append one sample triple.

        Parameters
        ----------
        eta           : (n,) state deviation at time k
        xi            : (m,) input deviation at time k
        eta_next      : (n,) state deviation at time k+1
        """

        e = np.asarray(eta, float).reshape(-1)
        u = np.asarray(xi, float).reshape(-1)
        ep = np.asarray(eta_next, float).reshape(-1)

        if e.size != self.n:
            raise ValueError(f"eta has length {e.size}, expected {self.n}")
        if u.size != self.m:
            raise ValueError(f"xi has length {u.size}, expected {self.m}")
        if ep.size != self.n:
            raise ValueError(f"eta_next has length {ep.size}, expected {self.n}")

        self.etas.append(e)
        self.xis.append(u)
        self.etas_next.append(ep)

    def excitation(self, k: int, v_bar: float, m: Optional[int] = None) -> Array:  # noqa: ARG002
        """
        Generate a bounded excitation vector v(k) with ||v(k)||_2 = v_bar.

        Parameters
        ----------
        k           : int, Time index
        v_bar       : float, Bound on the excitation norm
        m           : int, optional, Number of samples to generate. If None, use all samples.

        Returns
        -------
        v : (m,) array, Excitation vector
        """
        if m is not None and int(m) != self.m:
            raise ValueError(f"m={m} is inconsistent with self.m={self.m}")
        if v_bar <= 0.0:
            return np.zeros(self.m, float)

        v = self._rng.normal(size=(self.m,))
        norm = float(np.linalg.norm(v)) + 1e-15
        return (v_bar / norm) * v

    # ----------------
    # Export API
    # ----------------

    def export_mats(self) -> Tuple[Array, Array, Array]:
        """
        Return (H, H_plus, Xi) with shapes (n, L), (n, L), (m, L)

        Raises
        ------
        ValueError if no samples have been pushed or lengths mismatch.
        """
        L = len(self.etas)
        if L == 0 or not (len(self.xis) == len(self.etas_next) == L):
            raise ValueError("DataStack is empty or lists have mismatched lengths.")

        H = np.stack(self.etas, axis=1).astype(float, copy=False)
        H_plus = np.stack(self.etas_next, axis=1).astype(float, copy=False)
        Xi = np.stack(self.xis, axis=1).astype(float, copy=False)

        if H.shape != (self.n, L) or H_plus.shape != (self.n, L) or Xi.shape != (self.m, L):
            raise ValueError("Exported matrices have unexpected shapes.")
        return H, H_plus, Xi

    def pe_rank(self) -> Tuple[int, int]:
        """
        Persistenly-of-excitation quick check.

        Returns
        -------
        (rank, target) where target = n + m.
        """
        H, _, Xi = self.export_mats()
        M = np.vstack([H, Xi])
        r = int(np.linalg.matrix_rank(M))
        return r, (self.n + self.m)

    def length(self) -> int:
        """Number of samples currently buffered (columns of H)."""
        return len(self.etas)

    def reset_for_next_segment(self) -> None:
        """Clear all buffered samples."""
        self.etas.clear()
        self.xis.clear()
        self.etas_next.clear()
