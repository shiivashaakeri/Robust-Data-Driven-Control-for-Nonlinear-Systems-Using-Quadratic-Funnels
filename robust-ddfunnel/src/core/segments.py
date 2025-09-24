# src/core/segments.py

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SegmentClock:
    """
    Segment/time bookkeeping for the online data collection scheme

    Definitions:
        - Horizon indices: k=0, 1, ..., N-1
        - Segment i covers: T_i = {k_i, ..., k_{i+1}-1} with k_i = i*T
        - Data window for segment i:
            T_i^D = {k_i^D, ..., k_{i+1}-1}, with k_i^D = k_{i+1} - L_i
        Here we implement the simple policy L_i = min(L_max, |T_i|)

    API:
        in_data_window(k)        -> True iff k in T_i^D for the segment i containing k
        is_boundary(k)           -> True iff k is the last index in its segment
        segment_start_index(k)   -> k_i for the segment i containing k
    """

    N: int  # number of input steps
    T: int  # segment length
    L_min: int  # minimum segment length (for adaptive window)
    L_max: int  # maximum segment length (for adaptive window)
    v_bar: float  # excitation bound
    delta_min: float  # baseline deviation (for adaptive window)

    # ----------------
    # Public API
    # ----------------

    def in_data_window(self, k: int) -> bool:
        """
        Return True iff k lies in the data window T_i^D of its segment i.
        """
        i = self._seg_index(k)
        k_start, k_end_excl = self._seg_bounds(i)
        L_eff = self._L_eff(i)
        kD_start = max(k_end_excl - L_eff, k_start)
        return kD_start <= k < k_end_excl

    def is_boundary(self, k: int) -> bool:
        """
        Return True iff k is the last index of its segment i (i.e., k=k_{i+1}-1).
        """
        i = self._seg_index(k)
        _, k_end_excl = self._seg_bounds(i)
        return k == (k_end_excl - 1)

    def segment_start_index(self, k: int) -> int:
        """
        Return k_i, the first index of the segment i containing k.
        """
        i = self._seg_index(k)
        k_start, _ = self._seg_bounds(i)
        return k_start

    # ----------------
    # Private helpers
    # ----------------

    def _num_segments(self) -> int:
        """Number of segments covering indices 0, 1, ..., N-1"""
        return (self.N + max(self.T, 1) - 1) // max(self.T, 1)

    def _seg_index(self, k: int) -> int:
        """Segment index i for a given index k"""
        if not (0 <= k < self.N):
            raise IndexError(f"Index k={k} out of range [0, {self.N - 1}]")
        return k // self.T

    def _seg_bounds(self, i: int) -> tuple[int, int]:
        """
        Return (k_start, k_end_excl), for segment i.
        Segment i covers indices [k_start, k_end_excl-1)
        """
        if not (0 <= i < self._num_segments()):
            raise IndexError(f"Segment index i={i} out of range [0, {self._num_segments() - 1}]")
        k_start = i * self.T
        k_end_excl = min((i + 1) * self.T, self.N)
        return k_start, k_end_excl

    def _L_eff(self, i: int) -> int:
        """
        Effective data window length for segment i under the constant-L policy
            L_i = min(L_max, |T_i|)
        """
        k_start, k_end_excl = self._seg_bounds(i)
        seg_len = max(0, k_end_excl - k_start)
        return max(0, min(self.L_max, seg_len))
