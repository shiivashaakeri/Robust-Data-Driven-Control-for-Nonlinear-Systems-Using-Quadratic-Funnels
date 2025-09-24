from __future__ import annotations

import pathlib
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

# ---------------------------
# Helpers (FK + precompute)
# ---------------------------


def fk_2link(q1: float, q2: float, l1: float, l2: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Forward kinematics (planar 2R).
    Returns xs=[0,x1,x2], ys=[0,y1,y2].
    """
    x1 = l1 * np.cos(q1)
    y1 = l1 * np.sin(q1)
    x2 = x1 + l2 * np.cos(q1 + q2)
    y2 = y1 + l2 * np.sin(q1 + q2)
    return np.array([0.0, x1, x2]), np.array([0.0, y1, y2])


def _precompute_coords(X: np.ndarray, l1: float, l2: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Precompute joint and EE coordinates for an entire trajectory.
    X: (T+1, 4) with columns [q1,q2,dq1,dq2]
    Returns:
        XS, YS each shape (T+1, 3), columns for base, joint1, ee.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] < 2:
        raise ValueError("X must be (T+1, >=2) with q1,q2 at columns 0,1")
    q1 = X[:, 0]
    q2 = X[:, 1]
    T1 = l1 * np.stack([np.cos(q1), np.sin(q1)], axis=1)  # (T+1,2)
    T2 = l2 * np.stack([np.cos(q1 + q2), np.sin(q1 + q2)], axis=1)
    base = np.zeros((X.shape[0], 2))
    j1 = T1
    ee = T1 + T2
    XS = np.stack([base[:, 0], j1[:, 0], ee[:, 0]], axis=1)  # (T+1,3)
    YS = np.stack([base[:, 1], j1[:, 1], ee[:, 1]], axis=1)  # (T+1,3)
    return XS, YS


# ---------------------------
# Public dataclass
# ---------------------------


@dataclass
class ArmSpec:
    """
    One arm to animate.

    Attributes
    ----------
    name      : legend label.
    X         : (T+1, 4) trajectory with columns [q1,q2,dq1,dq2].
    l1, l2    : link lengths (m).
    color     : optional fixed color; default = Matplotlib cycle.
    linestyle : '-', '--', etc. (default: '-')
    linewidth : float (default: 3.0)
    trail_linestyle : (default: same as linestyle)
    trail_linewidth : float (default: 1.2)
    marker_size     : float for joint markers (default: 6.0)
    """

    name: str
    X: np.ndarray
    l1: float
    l2: float
    color: Optional[str] = None
    linestyle: str = "-"
    linewidth: float = 3.0
    trail_linestyle: Optional[str] = None
    trail_linewidth: float = 1.2
    marker_size: float = 6.0

    # cached FK (filled by animator)
    _XS: np.ndarray = field(init=False, repr=False)
    _YS: np.ndarray = field(init=False, repr=False)

    def precompute(self) -> None:
        self._XS, self._YS = _precompute_coords(self.X, self.l1, self.l2)


# ---------------------------
# Animator
# ---------------------------


class TwoLinkAnimator:
    def __init__(
        self,
        specs: Sequence[ArmSpec],
        dt: float,
        speed: float = 1.0,
        trail: int = 150,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None,
    ) -> None:
        if not specs:
            raise ValueError("Provide at least one ArmSpec.")
        self.specs: List[ArmSpec] = list(specs)
        self.dt = float(dt)
        self.speed = float(speed)
        self.trail = int(trail)
        self.title = title or "2-DoF Planar Arm"

        # Precompute FK and basic checks
        Tlens = []
        for sp in self.specs:
            sp.precompute()
            Tlens.append(sp.X.shape[0])
        if len(set(Tlens)) != 1:
            raise ValueError("All specs must share the same trajectory length (T+1).")
        self.frames = Tlens[0]  # T+1
        self.t_axis = np.arange(self.frames) * self.dt

        # Axis limits (square)
        Ls = [sp.l1 + sp.l2 for sp in self.specs]
        L = float(max(Ls))
        pad = 0.15 * L
        self.xlim = xlim if xlim is not None else (-L - pad, L + pad)
        self.ylim = ylim if ylim is not None else (-L - pad, L + pad)

        # Matplotlib
        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_xlim(*self.xlim)
        self.ax.set_ylim(*self.ylim)
        self.ax.grid(True, alpha=0.25)
        if self.title:
            self.ax.set_title(self.title)
        self.ax.set_xlabel("x (m)")
        self.ax.set_ylabel("y (m)")

        # Artists per spec
        self._lines = []
        self._j1dots = []
        self._eedots = []
        self._trails = []
        self._paths = [deque(maxlen=max(10, self.trail)) for _ in self.specs]

        # Use Matplotlib color cycle if color is None
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", None)

        for i, sp in enumerate(self.specs):
            color = sp.color or (color_cycle[i % len(color_cycle)] if color_cycle else None)
            trail_linestyle = sp.trail_linestyle or sp.linestyle

            # links
            (ln,) = self.ax.plot(
                [], [], linestyle=sp.linestyle, linewidth=sp.linewidth, label=sp.name, color=color, alpha=0.95
            )
            self._lines.append(ln)

            # joint markers
            (j1,) = self.ax.plot([], [], "o", markersize=sp.marker_size, color=color)
            (ee,) = self.ax.plot([], [], "o", markersize=sp.marker_size, color=color)
            self._j1dots.append(j1)
            self._eedots.append(ee)

            # trail
            (tr,) = self.ax.plot(
                [],
                [],
                linestyle=trail_linestyle,
                linewidth=sp.trail_linewidth,
                color=color,
                alpha=0.8,
                label=f"{sp.name} trail",
            )
            self._trails.append(tr)

        # legend + time
        self.ax.legend(loc="upper right", fontsize=9)
        self._time_text = self.ax.text(0.02, 0.98, "", transform=self.ax.transAxes, va="top")

        # Build animation
        interval_ms = max(10, int(1000 * self.dt / max(1e-6, self.speed)))
        self.anim = FuncAnimation(
            self.fig, self._update, frames=range(self.frames), init_func=self._init, interval=interval_ms, blit=True
        )

    # init + update
    def _init(self):
        artists = []
        for ln, j1, ee, tr in zip(self._lines, self._j1dots, self._eedots, self._trails):
            ln.set_data([], [])
            j1.set_data([], [])
            ee.set_data([], [])
            tr.set_data([], [])
            artists.extend([ln, j1, ee, tr])
        self._time_text.set_text("")
        return (*artists, self._time_text)

    def _update(self, k: int):
        artists = []
        for idx, sp in enumerate(self.specs):
            xs = sp._XS[k]
            ys = sp._YS[k]  # length-3 arrays
            # links
            self._lines[idx].set_data(xs, ys)
            # markers (1-length sequences)
            self._j1dots[idx].set_data([xs[1]], [ys[1]])
            self._eedots[idx].set_data([xs[2]], [ys[2]])
            # trail
            self._paths[idx].append((xs[2], ys[2]))
            if len(self._paths[idx]) > 1:
                tx, ty = zip(*self._paths[idx])
                self._trails[idx].set_data(tx, ty)
            artists.extend([self._lines[idx], self._j1dots[idx], self._eedots[idx], self._trails[idx]])

        self._time_text.set_text(f"t = {self.t_axis[k]:.2f} s")
        return (*artists, self._time_text)

    def show(self):
        plt.show()


# ---------------------------
# Convenience functions
# ---------------------------


def spec_from_model(model, X: np.ndarray, name: str, **style) -> ArmSpec:
    """
    Build an ArmSpec from a RobotArm2DOF-like object (expects .p.l1, .p.l2).

    Parameters
    ----------
    model : any with attributes 'p.l1' and 'p.l2' (e.g., RobotArm2DOF)
    X     : (T+1,4) trajectory
    name  : label
    style : optional ArmSpec style kwargs (color, linestyle, linewidth, ...)

    Returns
    -------
    ArmSpec
    """
    l1 = float(getattr(model, "p").l1)
    l2 = float(getattr(model, "p").l2)
    return ArmSpec(name=name, X=np.asarray(X, dtype=float), l1=l1, l2=l2, **style)


def animate_two_link(
    specs: Sequence[ArmSpec],
    dt: float,
    speed: float = 1.0,
    trail: int = 200,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,   # noqa: ARG001
):
    """
    Build and return (fig, ani) for a 2-link arm animation using ArmSpec.

    Notes
    -----
    - Accepts one or more ArmSpec instances.
    - Does NOT call plt.show(); caller owns showing/saving.
    - Returns (fig, FuncAnimation).
    """
    # Backward-compat: if someone accidentally passes dicts, adapt them.
    if len(specs) > 0 and isinstance(specs[0], dict):
        adapted = []
        for s in specs:  # type: ignore[index]
            adapted.append(
                ArmSpec(
                    name=s.get("name", "arm"),
                    X=np.asarray(s["X"], dtype=float),
                    l1=float(s.get("l1", 1.0)),
                    l2=float(s.get("l2", 1.0)),
                    color=s.get("color"),
                    linestyle=s.get("linestyle", "-"),
                    linewidth=float(s.get("linewidth", 3.0)),
                    trail_linestyle=s.get("trail_linestyle"),
                    trail_linewidth=float(s.get("trail_linewidth", 1.2)),
                    marker_size=float(s.get("marker_size", 6.0)),
                )
            )
        specs = adapted  # type: ignore[assignment]

    animator = TwoLinkAnimator(
        specs=specs,  # type: ignore[arg-type]
        dt=dt,
        speed=speed,
        trail=trail,
        title=title,
    )
    return animator.fig, animator.anim


def save_gif(fig: plt.Figure, ani: FuncAnimation, path: pathlib.Path, fps: int = 30, dpi: int = 120):  # noqa: ARG001
    """Save the given animation as a GIF using Pillow."""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=int(fps))
    ani.save(str(path), writer=writer, dpi=int(dpi))


__all__ = ["ArmSpec", "TwoLinkAnimator", "animate_two_link", "fk_2link", "spec_from_model"]
