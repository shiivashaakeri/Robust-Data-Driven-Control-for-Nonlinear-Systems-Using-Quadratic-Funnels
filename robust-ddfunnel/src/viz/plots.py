# src/viz/plots.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _as2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        return X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError("Expected 2D array shaped (T, d) or 1D length T.")
    return X


def plot_states(  # noqa: C901
    series: Sequence[Dict[str, Any]],
    names: Sequence[str],
    *,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 8),
    sharex: bool = True,
    legend_ncols: int = 3,
    # NEW: optional bounds for each state (low/high)
    x_bounds: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
    bound_style: Optional[Dict[str, Any]] = None,
    show_bounds_in_legend: bool = False,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot state trajectories in stacked subplots (one per state).

    Parameters
    ----------
    series : list of dicts, each with:
        - "label": str           legend label
        - "t": (T,) array        time stamps (must match rows of X)
        - "X": (T, n) array      state trajectories (columns follow 'names')
        - optional "style": dict matplotlib kwargs for plot()
    names : list[str] length n   state names for y-labels (from config)
    title : figure title
    x_bounds : optional (low, high), each length-n sequence for dashed lines
    bound_style : matplotlib style dict for bounds; defaults to gray dashed
    show_bounds_in_legend : include a single "bounds" entry in the legend
    """
    names = list(names)
    n = len(names)

    fig, axs = plt.subplots(n, 1, figsize=figsize, sharex=sharex)
    if n == 1:
        axs = np.array([axs])

    # Collect handles for a single shared legend
    handles: List[Any] = []
    labels: List[str] = []

    # Prepare bounds
    has_bounds = x_bounds is not None
    if has_bounds:
        x_low = np.asarray(x_bounds[0], dtype=float).reshape(-1)
        x_high = np.asarray(x_bounds[1], dtype=float).reshape(-1)
        if x_low.size != n or x_high.size != n:
            raise ValueError(f"x_bounds must have length {n} for low/high.")
    bstyle = {"linestyle": "--", "color": "k", "alpha": 0.35, "linewidth": 1.0}
    if bound_style:
        bstyle.update(bound_style)

    for i in range(n):
        ax = axs[i]
        # plot series
        for s in series:
            t = np.asarray(s["t"], dtype=float)
            X = _as2d(s["X"])
            if X.shape[1] < n:
                raise ValueError(f"X has {X.shape[1]} columns but names has {n}")
            style = s.get("style", {})
            ln = ax.plot(t, X[:, i], **style)[0]
            if i == 0:  # only collect once
                handles.append(ln)
                labels.append(str(s["label"]))

        # dashed bounds (no legend spam)
        if has_bounds:
            ln_low = ax.axhline(x_low[i], **bstyle)
            ln_high = ax.axhline(x_high[i], **bstyle)  # noqa: F841
            if show_bounds_in_legend and i == 0:
                handles.append(ln_low)
                labels.append("bounds")

        ax.set_ylabel(names[i])
        ax.grid(True, alpha=0.3)

    axs[-1].set_xlabel("time (s)")
    if title:
        fig.suptitle(title, y=0.995)
    if handles:
        fig.legend(
            handles,
            labels,
            ncols=legend_ncols,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            fontsize=9,
        )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig, axs


def plot_inputs(  # noqa: C901
    series: Sequence[Dict[str, Any]],
    names: Sequence[str],
    *,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 5),
    sharex: bool = True,
    legend_ncols: int = 3,
    # NEW: optional bounds for each input (low/high)
    u_bounds: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
    bound_style: Optional[Dict[str, Any]] = None,
    show_bounds_in_legend: bool = False,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot input trajectories in stacked subplots (one per input).

    Parameters
    ----------
    series : list of dicts, each with:
        - "label": str          legend label
        - "t": (T,) array       time stamps (must match rows of U)
        - "U": (T, m) array     input trajectories (columns follow 'names')
        - optional "style": dict matplotlib kwargs for plot()
    names : list[str] length m   input names for y-labels (from config)
    title : figure title
    u_bounds : optional (low, high), each length-m sequence for dashed lines
    bound_style : matplotlib style dict for bounds; defaults to gray dashed
    show_bounds_in_legend : include a single "bounds" entry in the legend
    """
    names = list(names)
    m = len(names)

    fig, axs = plt.subplots(m, 1, figsize=figsize, sharex=sharex)
    if m == 1:
        axs = np.array([axs])

    handles: List[Any] = []
    labels: List[str] = []

    # Prepare bounds
    has_bounds = u_bounds is not None
    if has_bounds:
        u_low = np.asarray(u_bounds[0], dtype=float).reshape(-1)
        u_high = np.asarray(u_bounds[1], dtype=float).reshape(-1)
        if u_low.size != m or u_high.size != m:
            raise ValueError(f"u_bounds must have length {m} for low/high.")
    bstyle = {"linestyle": "--", "color": "k", "alpha": 0.35, "linewidth": 1.0}
    if bound_style:
        bstyle.update(bound_style)

    for j in range(m):
        ax = axs[j]
        for s in series:
            t = np.asarray(s["t"], dtype=float)
            U = _as2d(s["U"])
            if U.shape[1] < m:
                raise ValueError(f"U has {U.shape[1]} columns but names has {m}")
            style = s.get("style", {})
            ln = ax.plot(t, U[:, j], **style)[0]
            if j == 0:
                handles.append(ln)
                labels.append(str(s["label"]))

        if has_bounds:
            ln_low = ax.axhline(u_low[j], **bstyle)
            ln_high = ax.axhline(u_high[j], **bstyle)  # noqa: F841
            if show_bounds_in_legend and j == 0:
                handles.append(ln_low)
                labels.append("bounds")

        ax.set_ylabel(names[j])
        ax.grid(True, alpha=0.3)

    axs[-1].set_xlabel("time (s)")
    if title:
        fig.suptitle(title, y=0.995)
    if handles:
        fig.legend(
            handles,
            labels,
            ncols=legend_ncols,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            fontsize=9,
        )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig, axs


# ---------------------------
# Ellipsoids
# ---------------------------


def _coerce_Qseq(Q_seq: np.ndarray, T: int, n: int, name: str) -> np.ndarray:
    """
    Accept (n,n) or (T,n,n). Return (T,n,n).
    """
    Q_seq = np.asarray(Q_seq, dtype=float)
    if Q_seq.ndim == 2:
        if Q_seq.shape != (n, n):
            raise ValueError(f"{name} must be (n,n) or (T,n,n); got {Q_seq.shape}")
        Q_seq = np.broadcast_to(Q_seq, (T, n, n))
    elif Q_seq.ndim == 3:
        if Q_seq.shape[0] != T or Q_seq.shape[1:] != (n, n):
            raise ValueError(f"{name} must be (T,n,n); got {Q_seq.shape}")
    else:
        raise ValueError(f"{name} must be (n,n) or (T,n,n)")
    return Q_seq


def shade_state_ellipsoid_bands(
    axs: np.ndarray,
    t: np.ndarray,
    X_nom: np.ndarray,
    Q_seq: np.ndarray,
    *,
    names: Optional[Sequence[str]] = None,  # noqa: ARG001
    color: Optional[str] = None,
    alpha: float = 0.18,
    label: Optional[str] = "state ellipsoid band",
) -> List[Any]:
    """
    Shade per-time ellipsoidal bands in each state subplot.

    Parameters
    ----------
    axs   : array of Axes, length n (from plot_states)
    t     : (T_s,) time vector; usually T_s = N+1 for states
    X_nom : (T_s, n) nominal states
    Q_seq : (n,n) or (T_s, n, n) positive definite 'shape' matrices
            for the ellipsoids centered at X_nom[k]
    names : optional state names (used only for sanity checks / label cosmetics)
    color : optional color for the band; defaults to Matplotlib cycle on each ax
    alpha : fill transparency
    label : legend label (only attached on the first axis)

    Returns
    -------
    fills : list of PolyCollection handles (one per state)
    """
    X_nom = np.asarray(X_nom, float)
    t = np.asarray(t, float)
    if X_nom.ndim != 2:
        raise ValueError("X_nom must be (T_s, n)")
    T_s, n = X_nom.shape
    if t.shape[0] != T_s:
        raise ValueError("t and X_nom must have the same first dimension")

    Q_seq = _coerce_Qseq(Q_seq, T_s, n, "Q_seq(states)")

    fills = []
    for i in range(n):
        ax = axs[i]
        # half-width along coordinate i is sqrt(Q_ii)
        qii = np.clip(Q_seq[:, i, i], a_min=0.0, a_max=None)
        half = np.sqrt(qii)
        center = X_nom[:, i]
        lower = center - half
        upper = center + half
        # choose color if not provided
        col = color or None
        fb = ax.fill_between(
            t, lower, upper, alpha=alpha, color=col, label=(label if i == 0 and label else None), linewidth=0
        )
        fills.append(fb)
    return fills


def shade_input_ellipsoid_bands(
    axs: np.ndarray,
    t: np.ndarray,
    U_nom: np.ndarray,
    R_seq: np.ndarray,
    *,
    names: Optional[Sequence[str]] = None,  # noqa: ARG001
    color: Optional[str] = None,
    alpha: float = 0.18,
    label: Optional[str] = "input ellipsoid band",
) -> List[Any]:
    """
    Shade per-time ellipsoidal bands in each input subplot.

    Parameters
    ----------
    axs   : array of Axes, length m (from plot_inputs)
    t     : (T_u,) time vector; usually T_u = N for inputs
    U_nom : (T_u, m) nominal inputs
    R_seq : (m,m) or (T_u, m, m) positive definite 'shape' matrices
            for ellipsoids centered at U_nom[k]
    names : optional input names
    color : optional band color
    alpha : fill transparency
    label : legend label (only on first axis)

    Returns
    -------
    fills : list of PolyCollection handles (one per input)
    """
    U_nom = np.asarray(U_nom, float)
    t = np.asarray(t, float)
    if U_nom.ndim != 2:
        raise ValueError("U_nom must be (T_u, m)")
    T_u, m = U_nom.shape
    if t.shape[0] != T_u:
        raise ValueError("t and U_nom must have the same first dimension")

    R_seq = _coerce_Qseq(R_seq, T_u, m, "R_seq(inputs)")

    fills = []
    for j in range(m):
        ax = axs[j]
        rii = np.clip(R_seq[:, j, j], a_min=0.0, a_max=None)
        half = np.sqrt(rii)
        center = U_nom[:, j]
        lower = center - half
        upper = center + half
        col = color or None
        fb = ax.fill_between(
            t, lower, upper, alpha=alpha, color=col, label=(label if j == 0 and label else None), linewidth=0
        )
        fills.append(fb)
    return fills
