# src/synthesis/funnel_utils.py
# src/synthesis/funnel_utils.py
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

# project deps (kept lightweight + same patterns you used elsewhere)
from models.discretization import make_stepper  # type: ignore
from nominal.trajectory import _dlqr_gain  # type: ignore
from synthesis.feasibility_lmis import bounds_at_k  # type: ignore  # reuse your existing logic

# --------- small linear-algebra helpers ---------


def _nearest_psd(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Project a symmetric matrix to PSD by clipping eigs at eps."""
    A = 0.5 * (A + A.T)
    w, V = np.linalg.eigh(A)
    w_clipped = np.clip(w, a_min=eps, a_max=None)
    return (V * w_clipped) @ V.T


def _cap_psd_le(Q: np.ndarray, Qcap: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Return Qproj such that Qproj ⪯ Qcap and Qproj is as close to Q as a single
    uniform scaling allows: Qproj = t * Q with t = min(1, 1/λ_max(S)),
    S = Qcap^{-1/2} Q Qcap^{-1/2}.

    Assumes Qcap is PSD / PD. If Qcap is singular, a tiny jitter is added.
    """
    Q = _nearest_psd(Q, eps)
    Qcap = _nearest_psd(Qcap, eps)

    # Robust inverse square-root of Qcap
    wc, U = np.linalg.eigh(Qcap)
    wc = np.clip(wc, a_min=eps, a_max=None)
    Qcap_mhalf = (U * (wc**-0.5)) @ U.T

    S = Qcap_mhalf @ Q @ Qcap_mhalf
    # numerical symmetrization
    S = 0.5 * (S + S.T)
    wmax = float(np.max(np.linalg.eigvalsh(S)))
    t = 1.0 if wmax <= 1.0 else 1.0 / (wmax + eps)
    return t * Q


# --------- 1) initial_Q_from_dlqr ---------


def initial_Q_from_dlqr(
    plant,
    x_goal: np.ndarray,
    u_goal: np.ndarray,
    Q_lqr: np.ndarray | None = None,
    R_lqr: np.ndarray | None = None,
    *,
    eps: float = 1e-9,
    scale: float = 1.0,
    Q_cap: np.ndarray | None = None,
) -> np.ndarray:
    """
    Build an initial funnel shape Q0 from the Riccati solution at the goal.

    Steps:
      1) Linearize the *discrete* dynamics at (x_goal, u_goal) using the plant.
      2) Solve DLQR(A,B,Q_lqr,R_lqr) to get (K, P).
      3) Set Q0 = scale * (P + eps*I)^{-1}.
      4) If Q_cap is provided, project so Q0 ⪯ Q_cap.

    Returns
    -------
    Q0 : (n,n) SPD
    """
    x_goal = np.asarray(x_goal, float).reshape(-1)
    u_goal = np.asarray(u_goal, float).reshape(-1)

    # fall back to mild weights if not provided
    n = x_goal.shape[0]
    m = u_goal.shape[0]
    Qw = np.asarray(Q_lqr if Q_lqr is not None else np.diag([10.0] * min(2, n) + [1.0] * (n - min(2, n))), float)
    Rw = np.asarray(R_lqr if R_lqr is not None else np.diag([0.1] * m), float)

    # discrete jacobians via the plant's stepper
    step = make_stepper(plant.f, plant.dt, method=getattr(plant, "_integrator_name", "rk4"))
    if hasattr(plant, "discrete_jacobians_fd"):
        A, B = plant.discrete_jacobians_fd(step, x_goal, u_goal)  # type: ignore[attr-defined]
    else:
        # local import to avoid cycles
        from models.discretization import discrete_jacobians_fd  # type: ignore  # noqa: PLC0415

        A, B = discrete_jacobians_fd(step, x_goal, u_goal, eps=1e-6)

    # DLQR; assume _dlqr_gain returns (K, P)
    K, P = _dlqr_gain(A, B, Qw, Rw, tol=1e-9, maxit=10000)  # P is Riccati solution
    P = np.asarray(P, float)

    Q0 = scale * np.linalg.inv(P + eps * np.eye(P.shape[0]))
    Q0 = _nearest_psd(Q0, eps)

    if Q_cap is not None:
        Q_cap = np.asarray(Q_cap, float)
        Q0 = _cap_psd_le(Q0, Q_cap, eps)

    return Q0


# --------- 2) segmentwise_Q_applied ---------


def segmentwise_Q_applied(
    Qs_solved: Sequence[np.ndarray],
    Q0: np.ndarray,
    N: int,
    T: int,
) -> List[np.ndarray]:
    """
    Produce the list of *applied* per-segment Qs (length n_segs = ceil(N/T)).

    Convention:
      - Segment 0 uses Q0.
      - Segment i≥1 uses Qs_solved[i-1] (since it's computed at the end of segment i-1).
      - If fewer solutions than segments-1 exist, repeat the last available.
    """
    n_segs = (N + T - 1) // T
    out: List[np.ndarray] = []
    last = np.asarray(Q0, float)
    out.append(last.copy())
    for i in range(1, n_segs):
        if i - 1 < len(Qs_solved):
            last = np.asarray(Qs_solved[i - 1], float)
        out.append(last.copy())
    return out


# --------- 3) expand_Q_to_per_step ---------


def expand_Q_to_per_step(
    Q_seg: Sequence[np.ndarray],
    N: int,
    T: int,
    *,
    include_terminal: bool = True,
) -> np.ndarray:
    """
    Expand per-segment Q to a per-step sequence for states.

    Returns:
      Q_seq : (N+1, n, n) if include_terminal else (N, n, n)

    Mapping:
      For k in [iT, (i+1)T-1] (clipped by N-1), use Q_seg[i].
      If include_terminal, set Q_seq[N] = Q_seg[-1].
    """
    Q0 = np.asarray(Q_seg[0], float)
    n = Q0.shape[0]
    T_states = N + 1 if include_terminal else N
    Q_seq = np.zeros((T_states, n, n), float)

    for k in range(N):  # 0..N-1
        i = min(k // T, len(Q_seg) - 1)
        Q_seq[k] = np.asarray(Q_seg[i], float)

    if include_terminal:
        Q_seq[N] = np.asarray(Q_seg[-1], float)

    return Q_seq


# --------- 4) clamp_Qseq_to_bounds ---------


def clamp_Qseq_to_bounds(
    Q_seq: np.ndarray,
    Q_bounds: Dict,
    k_states: Sequence[int],
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Enforce Q[k] ⪯ Q_max(k) for the provided state indices.

    Uses `bounds_at_k(k, Q_bounds, R_bounds)` with a dummy R to fetch caps.
    Performs a PSD "≤" projection via uniform scaling in the Q_max metric.
    """
    Q_seq = np.asarray(Q_seq, float).copy()
    T_s, n, _ = Q_seq.shape

    if len(k_states) != T_s:
        raise ValueError(f"len(k_states)={len(k_states)} must match Q_seq.shape[0]={T_s}")

    # dummy R to satisfy the signature; never used
    R_dummy = {"full": np.eye(1)}

    for idx, k in enumerate(k_states):
        Q_cap, _ = bounds_at_k(int(k), Q_bounds, R_dummy)  # (n,n)
        Q_cap = np.asarray(Q_cap, float)
        if Q_cap.shape != (n, n):
            raise ValueError(f"Q_cap at k={k} has shape {Q_cap.shape}, expected {(n, n)}")
        Q_seq[idx] = _cap_psd_le(Q_seq[idx], Q_cap, eps)

    return Q_seq


Array = np.ndarray


def per_step_Q_for_segment0(Q_bounds: Dict, T: int, n: int) -> Array:  # noqa: ARG001
    """
    Return (T, n, n) with Q[k] = Q_max(k) for k in segment 0 (k=0..T-1).
    Handles both {"diag": (T_total, n)} and {"full": (n,n)} cases.
    """
    Q_list: List[Array] = []
    for k in range(T):
        Qk, _ = bounds_at_k(k, Q_bounds, {"full": np.eye(1)})  # dummy R
        Q_list.append(Qk.astype(float))
    return np.stack(Q_list, axis=0)  # (T, n, n)


def build_Q_seq_states(Qs_solved: Sequence[Array], Q_bounds: Dict, *, N: int, T: int, n: int) -> Array:
    """
    Combine:
      - segment 0 (k=0..T-1): use per-step Q_max(k)
      - segment i>=1: use the single Q_i solved at the *end* of segment i-1
    Return (N+1, n, n) for state times 0..N (terminal copied from last step).
    """
    Q_seq = np.zeros((N + 1, n, n), dtype=float)

    # seg 0: per-step caps
    Q0_steps = per_step_Q_for_segment0(Q_bounds, T=min(T, N), n=n)  # (T0, n, n)
    Q_seq[: Q0_steps.shape[0]] = Q0_steps

    # later segments: constant per segment from solved Q_i
    n_segs = int(np.ceil(N / T))
    for i in range(1, n_segs):
        k_start = i * T
        k_end = min((i + 1) * T, N)  # fill steps k_start..k_end-1
        if len(Qs_solved) >= i:  # we solved Q_i for segment i (index i-1)
            Qi = np.asarray(Qs_solved[i - 1], float)
        elif len(Qs_solved) > 0:  # fallback to last known
            Qi = np.asarray(Qs_solved[-1], float)
        else:  # no solutions? fall back to caps again
            Qi = np.eye(n)
        Q_seq[k_start:k_end] = Qi

    # terminal state index N: copy last assigned (or identity if empty)
    Q_seq[N] = Q_seq[N - 1] if N > 0 else (Q0_steps[0] if Q0_steps.size else np.eye(n))
    return Q_seq


__all__ = [
    "clamp_Qseq_to_bounds",
    "expand_Q_to_per_step",
    "initial_Q_from_dlqr",
    "segmentwise_Q_applied",
]


def _spd_eig_fun(Q: np.ndarray, fun, eps: float = 1e-12) -> np.ndarray:
    """Apply scalar function to SPD matrix via eigen-decomposition, with clipping."""
    Q = np.asarray(Q, float)
    w, V = np.linalg.eigh(0.5 * (Q + Q.T))
    w = np.clip(w, eps, None)
    return (V * fun(w)) @ V.T


def spd_log(Q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return _spd_eig_fun(Q, np.log, eps)


def spd_exp(S: np.ndarray) -> np.ndarray:
    # S is symmetric in practice; symmetrize just in case
    S = 0.5 * (S + S.T)
    return _spd_eig_fun(S, np.exp)


def spd_geodesic_blend(QA: np.ndarray, QB: np.ndarray, w: float) -> np.ndarray:
    """Log-Euclidean geodesic between SPD QA and QB at weight w in [0,1]."""
    w = float(np.clip(w, 0.0, 1.0))
    LA = spd_log(QA)
    LB = spd_log(QB)
    return spd_exp((1.0 - w) * LA + w * LB)


def expand_Q_to_per_step_smooth(
    Q_seg: list[np.ndarray],
    N: int,
    T: int,
    *,
    include_terminal: bool = True,
    blend_steps: int = 0,
    ramp: str = "cosine",
) -> np.ndarray:
    """
    Turn per-segment Q's into a per-step sequence and smooth across boundaries.

    Q_seg: list of length n_segs (segment 0..n_segs-1). Segment 0 is initial.
    N    : number of input steps (states have N+1 samples)
    T    : segment length (from config.segmentation.T)
    blend_steps: half-width r of the blending window (in steps) around each boundary.
                 Total blend window length = 2*r+1 centered at k = (i+1)*T.
    ramp : "cosine" (smooth), or "linear"
    """
    Q0 = np.asarray(Q_seg[0], float)
    n = Q0.shape[0]
    T_s = N + 1 if include_terminal else N
    Q_seq = np.zeros((T_s, n, n), float)

    n_segs = int(np.ceil(N / T))
    assert len(Q_seg) >= n_segs, "Q_seg must cover all segments."

    # 1) fill piecewise-constant
    for i in range(n_segs):
        k0 = i * T
        k1 = min((i + 1) * T, N)  # inclusive end index for states
        Q_i = np.asarray(Q_seg[i], float)
        Q_seq[k0 : k1 + 1, :, :] = Q_i

    # 2) blend across boundaries
    r = int(max(0, blend_steps))
    if r > 0:

        def w_smooth(u):
            # u in [0,1] -> [0,1]
            if ramp == "linear":
                return u
            # cosine ease-in/out
            return 0.5 - 0.5 * np.cos(np.pi * u)

        for i in range(n_segs - 1):
            Q_prev = np.asarray(Q_seg[i], float)
            Q_next = np.asarray(Q_seg[i + 1], float)
            k_c = min((i + 1) * T, N)  # boundary index in state timeline

            # window: k in [k_c - r, k_c + r], clamp to [0, N]
            k_lo = max(0, k_c - r)
            k_hi = min(N, k_c + r)
            L = k_hi - k_lo
            if L <= 0:
                continue

            for j, k in enumerate(range(k_lo, k_hi + 1)):
                u = (k - k_lo) / max(1, (k_hi - k_lo))  # 0..1 across window
                w = w_smooth(u)
                Q_seq[k, :, :] = spd_geodesic_blend(Q_prev, Q_next, w)

    # 3) terminal
    if include_terminal:
        Q_seq[-1, :, :] = np.asarray(Q_seg[min(n_segs - 1, len(Q_seg) - 1)], float)

    return Q_seq
