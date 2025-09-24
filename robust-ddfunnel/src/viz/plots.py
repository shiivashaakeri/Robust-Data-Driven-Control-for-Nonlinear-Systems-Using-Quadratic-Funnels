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


def plot_states(
    series: Sequence[Dict[str, Any]],
    names: Sequence[str],
    *,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 8),
    sharex: bool = True,
    legend_ncols: int = 3,
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
    """
    names = list(names)
    n = len(names)

    fig, axs = plt.subplots(n, 1, figsize=figsize, sharex=sharex)
    if n == 1:
        axs = np.array([axs])

    # Collect handles for a single shared legend
    handles: List[Any] = []
    labels: List[str] = []

    for i in range(n):
        ax = axs[i]
        for k, s in enumerate(series):
            t = np.asarray(s["t"], dtype=float)
            X = _as2d(s["X"])
            if X.shape[1] < n:
                raise ValueError(f"X has {X.shape[1]} columns but names has {n}")
            style = s.get("style", {})
            ln = ax.plot(t, X[:, i], **style)[0]
            if i == 0:  # only collect once
                handles.append(ln)
                labels.append(str(s["label"]))
        ax.set_ylabel(names[i])
        ax.grid(True, alpha=0.3)

    axs[-1].set_xlabel("time (s)")
    if title:
        fig.suptitle(title, y=0.995)
    if handles:
        fig.legend(handles, labels, ncols=legend_ncols, loc="upper center",
                   bbox_to_anchor=(0.5, 1.02), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig, axs


def plot_inputs(
    series: Sequence[Dict[str, Any]],
    names: Sequence[str],
    *,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 5),
    sharex: bool = True,
    legend_ncols: int = 3,
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
    """
    names = list(names)
    m = len(names)

    fig, axs = plt.subplots(m, 1, figsize=figsize, sharex=sharex)
    if m == 1:
        axs = np.array([axs])

    handles: List[Any] = []
    labels: List[str] = []

    for j in range(m):
        ax = axs[j]
        for k, s in enumerate(series):
            t = np.asarray(s["t"], dtype=float)
            U = _as2d(s["U"])
            if U.shape[1] < m:
                raise ValueError(f"U has {U.shape[1]} columns but names has {m}")
            style = s.get("style", {})
            ln = ax.plot(t, U[:, j], **style)[0]
            if j == 0:
                handles.append(ln)
                labels.append(str(s["label"]))
        ax.set_ylabel(names[j])
        ax.grid(True, alpha=0.3)

    axs[-1].set_xlabel("time (s)")
    if title:
        fig.suptitle(title, y=0.995)
    if handles:
        fig.legend(handles, labels, ncols=legend_ncols, loc="upper center",
                   bbox_to_anchor=(0.5, 1.02), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig, axs
