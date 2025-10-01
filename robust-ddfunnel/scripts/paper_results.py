#!/usr/bin/env python3
"""Publication-ready 3D funnel visualization for q1–q2–time."""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ensure src/ is importable
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from utils.config import build_robot_arm_demo_config, load_yaml  # pyright: ignore[reportMissingImports]

# reuse the smoothing helper from plot_funnels
from plot_funnels import smooth_Qseq_around_boundaries, spd_geodesic_blend  # pyright: ignore[reportMissingImports]


def _project_ellipse(
    center: np.ndarray,
    Q: np.ndarray,
    *,
    dims: Tuple[int, int] = (0, 1),
    num_points: int = 120,
) -> np.ndarray:
    """Return boundary points (2, num_points) of the ellipse defined by Q in the selected dims."""

    idx0, idx1 = dims
    Q_sub = Q[np.ix_(dims, dims)]
    Q_sub = 0.5 * (Q_sub + Q_sub.T)

    w, V = np.linalg.eigh(Q_sub)
    w = np.clip(w, 0.0, None)
    radii = np.sqrt(w)

    theta = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=True)
    circle = np.vstack((np.cos(theta), np.sin(theta)))
    ellipse = (V @ (radii[:, None] * circle))
    center_2d = center[[idx0, idx1]].reshape(2, 1)
    pts = center_2d + ellipse
    return pts


def _upsample_funnel(
    X_center: np.ndarray,
    Q_seq: np.ndarray,
    t: np.ndarray,
    *,
    factor: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    factor = max(1, int(factor))
    if factor <= 1:
        return X_center, Q_seq, t

    X_list: list[np.ndarray] = []
    Q_list: list[np.ndarray] = []
    t_list: list[float] = []
    steps = len(t)
    for k in range(steps - 1):
        X0, X1 = X_center[k], X_center[k + 1]
        Q0, Q1 = Q_seq[k], Q_seq[k + 1]
        t0, t1 = t[k], t[k + 1]
        for i in range(factor):
            u = i / factor
            X_list.append((1.0 - u) * X0 + u * X1)
            Q_list.append(spd_geodesic_blend(Q0, Q1, u))
            t_list.append((1.0 - u) * t0 + u * t1)
    X_list.append(X_center[-1])
    Q_list.append(Q_seq[-1])
    t_list.append(float(t[-1]))
    return np.asarray(X_list), np.asarray(Q_list), np.asarray(t_list)


def _upsample_inputs(U_center: np.ndarray, R_diag: np.ndarray, t: np.ndarray, *, factor: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    factor = max(1, int(factor))
    if factor <= 1:
        return U_center, R_diag, t

    U_list: list[np.ndarray] = []
    R_list: list[np.ndarray] = []
    t_list: list[float] = []
    steps = len(t)
    for k in range(steps - 1):
        U0, U1 = U_center[k], U_center[k + 1]
        R0, R1 = R_diag[k], R_diag[k + 1]
        t0, t1 = t[k], t[k + 1]
        for i in range(factor):
            u = i / factor
            U_list.append((1.0 - u) * U0 + u * U1)
            R_list.append((1.0 - u) * R0 + u * R1)
            t_list.append((1.0 - u) * t0 + u * t1)
    U_list.append(U_center[-1])
    R_list.append(R_diag[-1])
    t_list.append(float(t[-1]))
    return np.asarray(U_list), np.asarray(R_list), np.asarray(t_list)


def build_funnel_surface(
    X_center: np.ndarray,
    Q_seq: np.ndarray,
    t: np.ndarray,
    *,
    dims: Tuple[int, int] = (0, 1),
    num_points: int = 120,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct mesh grids (X, Y, Z) describing the funnel boundary surface."""

    steps = X_center.shape[0]
    X_mesh = np.zeros((steps, num_points))
    Y_mesh = np.zeros((steps, num_points))
    Z_mesh = np.repeat(t.reshape(-1, 1), num_points, axis=1)

    for k in range(steps):
        pts = _project_ellipse(X_center[k], Q_seq[k], dims=dims, num_points=num_points)
        X_mesh[k, :] = pts[0]
        Y_mesh[k, :] = pts[1]

    return X_mesh, Y_mesh, Z_mesh


def build_input_funnel_surface(
    U_center: np.ndarray,
    R_diag: np.ndarray,
    t: np.ndarray,
    *,
    num_points: int = 120,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    steps = U_center.shape[0]
    X_mesh = np.zeros((steps, num_points))
    Y_mesh = np.zeros((steps, num_points))
    Z_mesh = np.repeat(t.reshape(-1, 1), num_points, axis=1)

    theta = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=True)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    for k in range(steps):
        r1 = np.sqrt(max(R_diag[k, 0], 0.0))
        r2 = np.sqrt(max(R_diag[k, 1], 0.0))
        X_mesh[k, :] = U_center[k, 0] + r1 * cos_t
        Y_mesh[k, :] = U_center[k, 1] + r2 * sin_t

    return X_mesh, Y_mesh, Z_mesh


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate 3D funnel plot for q1-q2-time")
    p.add_argument("--cfg", required=True, help="Robot config YAML (for bounds & names)")
    p.add_argument("--run", required=True, help="Path to saved online_run_*.npz")
    p.add_argument("--max-steps", type=int, default=600, help="Maximum number of steps to visualize")
    p.add_argument("--blend-frac", type=float, default=0.15, help="Blend half-window as fraction of T")
    p.add_argument("--blend-steps", type=int, default=None, help="Override blend window in steps")
    p.add_argument("--ramp", choices=["cosine", "linear"], default="cosine", help="Smoothing ramp")
    p.add_argument("--num-theta", type=int, default=200, help="Number of angular samples for ellipse boundary")
    p.add_argument("--upsample", type=int, default=4, help="Per-step time upsample factor for smoother surface")
    p.add_argument("--save", type=str, default=None, help="Save 3D q1–q2 funnel to this PNG path")
    p.add_argument("--save-slices", type=str, default=None, help="Save q/qdot band figure to this PNG path")
    p.add_argument("--save-lyapunov", type=str, default=None, help="Save Lyapunov plot to this PNG path")
    p.add_argument("--save-input-funnel", type=str, default=None, help="Save τ1–τ2 funnel figure to this PNG path")
    p.add_argument("--no-show", action="store_true", help="Skip interactive display")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg_path = pathlib.Path(args.cfg)
    run_path = pathlib.Path(args.run)

    # --- Load config (for labels & bounds) ---
    raw_cfg = load_yaml(cfg_path)
    demo, extras = build_robot_arm_demo_config(raw_cfg)
    state_names = extras.get("names_x") or ["q1", "q2", "dq1", "dq2"]
    q1_bounds = (float(demo.X.low[0]), float(demo.X.high[0]))
    q2_bounds = (float(demo.X.low[1]), float(demo.X.high[1]))
    U_low = np.asarray(demo.U.low, float)
    U_high = np.asarray(demo.U.high, float)

    # --- Load run ---
    Z = np.load(run_path, allow_pickle=True)
    Xhat = np.asarray(Z["Xhat"], float)
    Xplant = np.asarray(Z["Xplant"], float)
    Uhat = np.asarray(Z["Uhat"], float)
    Uplant = np.asarray(Z.get("Uapplied", Z.get("Up")), float)
    Q_seq_states = np.asarray(Z["Q_seq_states"], float)
    dt = float(Z["dt"])
    T = int(Z["T"])
    R_caps = None
    constants_raw = Z.get("constants", None)
    constants = None
    if isinstance(constants_raw, dict):
        constants = constants_raw
    elif hasattr(constants_raw, "item"):
        try:
            maybe = constants_raw.item()
            if isinstance(maybe, dict):
                constants = maybe
        except Exception:
            constants = None
    if isinstance(constants, dict) and "R_tv_diag" in constants:
        R_caps = np.asarray(constants["R_tv_diag"], float)

    max_steps = int(max(1, args.max_steps))
    N_total = Xhat.shape[0] - 1
    N_keep = min(N_total, max_steps)

    sl = slice(0, N_keep + 1)
    Xhat = Xhat[sl]
    Xplant = Xplant[sl]
    Q_seq_states = Q_seq_states[sl]
    if Uhat.shape[0] >= N_keep:
        Uhat = Uhat[: N_keep]
        Uplant = Uplant[: N_keep]
    if R_caps is not None:
        R_caps = R_caps[: min(R_caps.shape[0], N_keep)]

    t_states = np.arange(Xhat.shape[0], dtype=float) * dt

    # Lyapunov values prior to shrink
    eta = np.asarray(Xplant - Xhat, float)
    V_vals = np.zeros(eta.shape[0])
    eye_n = np.eye(Q_seq_states.shape[1])
    Q0 = Q_seq_states[0]
    for k in range(eta.shape[0]):
        Qk = Q0 if k < T else Q_seq_states[k]
        try:
            sol = np.linalg.solve(Qk, eta[k])
        except np.linalg.LinAlgError:
            sol = np.linalg.solve(Qk + 1e-8 * eye_n, eta[k])
        V_vals[k] = float(eta[k].dot(sol))

    # --- Smooth funnel like plot_funnels ---
    blend_steps = args.blend_steps
    if blend_steps is None:
        blend_steps = int(max(0.0, args.blend_frac) * min(T, N_keep if N_keep else T))
    if blend_steps > 0:
        Q_plot = smooth_Qseq_around_boundaries(Q_seq_states, T, blend_steps=blend_steps, ramp=args.ramp)
    else:
        Q_plot = Q_seq_states

    shrink_factor = 0.98
    Q_plot = np.asarray(Q_plot, float) * shrink_factor

    # --- Build funnel surface ---
    Xhat_s, Q_plot_s, t_states_s = _upsample_funnel(Xhat, Q_plot, t_states, factor=args.upsample)
    num_theta = max(64, args.num_theta)
    X_mesh, Y_mesh, Z_mesh = build_funnel_surface(Xhat_s, Q_plot_s, t_states_s, dims=(0, 1), num_points=num_theta)

    # --- 3D funnel figure ---
    fig3d = plt.figure(figsize=(6.8, 5.4))
    ax3d = fig3d.add_subplot(111, projection="3d")

    surf = ax3d.plot_surface(
        X_mesh,
        Y_mesh,
        Z_mesh,
        color="0.7",
        alpha=0.35,
        linewidth=0,
        antialiased=True,
    )

    ax3d.plot(Xhat_s[:, 0], Xhat_s[:, 1], t_states_s, color="black", linewidth=1.9)
    plant_q1 = np.interp(t_states_s, t_states, Xplant[:, 0])
    plant_q2 = np.interp(t_states_s, t_states, Xplant[:, 1])
    ax3d.plot(plant_q1, plant_q2, t_states_s, color="red", linewidth=1.4)

    ax3d.set_xlabel(f"{state_names[0]} (rad)", labelpad=3)
    ax3d.set_ylabel(f"{state_names[1]} (rad)", labelpad=5)
    ax3d.set_zlabel("time (s)", labelpad=7)
    ax3d.set_title("q1–q2 funnel", pad=1.0, fontsize=11)
    ax3d.dist = 8.4
    for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        axis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        axis._axinfo["grid"]["linewidth"] = 0.32
        axis._axinfo["grid"]["color"] = (0.75, 0.75, 0.75, 0.25)
    ax3d.tick_params(axis="both", which="major", labelsize=7)
    ax3d.tick_params(axis="z", which="major", labelsize=7)

    # add bounding box for q1/q2 limits
    t0 = float(t_states[0])
    t1 = float(t_states[-1])
    q1_lo, q1_hi = q1_bounds
    q2_lo, q2_hi = q2_bounds
    verts = [
        [(q1_lo, q2_lo, t0), (q1_hi, q2_lo, t0), (q1_hi, q2_hi, t0), (q1_lo, q2_hi, t0)],
        [(q1_lo, q2_lo, t1), (q1_hi, q2_lo, t1), (q1_hi, q2_hi, t1), (q1_lo, q2_hi, t1)],
        [(q1_lo, q2_lo, t0), (q1_hi, q2_lo, t0), (q1_hi, q2_lo, t1), (q1_lo, q2_lo, t1)],
        [(q1_hi, q2_lo, t0), (q1_hi, q2_hi, t0), (q1_hi, q2_hi, t1), (q1_hi, q2_lo, t1)],
        [(q1_hi, q2_hi, t0), (q1_lo, q2_hi, t0), (q1_lo, q2_hi, t1), (q1_hi, q2_hi, t1)],
        [(q1_lo, q2_hi, t0), (q1_lo, q2_lo, t0), (q1_lo, q2_lo, t1), (q1_lo, q2_hi, t1)],
    ]
    face_color = (0.42, 0.76, 0.50, 0.03)
    edge_color = "#3a8f4c"
    box = Poly3DCollection(verts, alpha=0.03, facecolor=face_color, edgecolor=edge_color, linewidths=0.45)
    ax3d.add_collection3d(box)

    # draw dashed edges to accentuate the prism
    edge_segments = [
        [(q1_lo, q2_lo, t0), (q1_lo, q2_lo, t1)],
        [(q1_hi, q2_lo, t0), (q1_hi, q2_lo, t1)],
        [(q1_hi, q2_hi, t0), (q1_hi, q2_hi, t1)],
        [(q1_lo, q2_hi, t0), (q1_lo, q2_hi, t1)],
        [(q1_lo, q2_lo, t0), (q1_hi, q2_lo, t0)],
        [(q1_hi, q2_lo, t0), (q1_hi, q2_hi, t0)],
        [(q1_hi, q2_hi, t0), (q1_lo, q2_hi, t0)],
        [(q1_lo, q2_hi, t0), (q1_lo, q2_lo, t0)],
        [(q1_lo, q2_lo, t1), (q1_hi, q2_lo, t1)],
        [(q1_hi, q2_lo, t1), (q1_hi, q2_hi, t1)],
        [(q1_hi, q2_hi, t1), (q1_lo, q2_hi, t1)],
        [(q1_lo, q2_hi, t1), (q1_lo, q2_lo, t1)],
    ]
    for seg in edge_segments:
        xs, ys, zs = zip(*seg)
        ax3d.plot(xs, ys, zs, color=edge_color, linestyle=(0, (2.0, 3.0)), linewidth=0.6)

    ax3d.view_init(elev=28, azim=-55)
    try:
        ax3d.set_box_aspect((1.0, 1.0, 0.7), zoom=1.0)
    except AttributeError:  # matplotlib < 3.7 fallback
        pass

    handles = [
        Line2D([0], [0], color="black", linewidth=1.9),
        Line2D([0], [0], color="red", linewidth=1.4),
        Line2D([0], [0], color="0.55", linewidth=5.4, alpha=0.35),
        Line2D([0], [0], color=edge_color, linewidth=0.8, linestyle=(0, (2.0, 3.0))),
    ]
    labels = ["nominal", "plant", "funnel", "feasible bounds"]
    ax3d.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.88),
        frameon=False,
        fontsize=7,
        handlelength=2.1,
        borderpad=0.08,
        labelspacing=0.25,
    )

    fig3d.tight_layout(pad=0.3)

    # --- q/qdot band figure ---
    fig_slices = plt.figure(figsize=(8.5, 5.3))
    gs_slices = GridSpec(2, 2, figure=fig_slices, hspace=0.36, wspace=0.28)
    ax_q1 = fig_slices.add_subplot(gs_slices[0, 0])
    ax_q2 = fig_slices.add_subplot(gs_slices[0, 1])
    ax_dq1 = fig_slices.add_subplot(gs_slices[1, 0])
    ax_dq2 = fig_slices.add_subplot(gs_slices[1, 1])

    plant_interp_all = np.vstack([np.interp(t_states_s, t_states, Xplant[:, i]) for i in range(Xplant.shape[1])])
    diag_Q = np.sqrt(np.maximum(np.diagonal(Q_plot_s, axis1=1, axis2=2), 0.0))

    def _plot_band(ax, idx: int, title: str, ylabel: str, show_xlabel: bool) -> None:
        center = Xhat_s[:, idx]
        plant_series = plant_interp_all[idx]
        half = diag_Q[:, idx]
        lower = center - half
        upper = center + half
        ax.fill_between(t_states_s, lower, upper, color="0.7", alpha=0.3, linewidth=0)
        ax.plot(t_states_s, center, color="black", linewidth=1.7)
        ax.plot(t_states_s, plant_series, color="red", linewidth=1.3)
        low = float(demo.X.low[idx])
        high = float(demo.X.high[idx])
        ax.axhline(low, color=edge_color, linestyle="-", linewidth=1.3)
        ax.axhline(high, color=edge_color, linestyle="-", linewidth=1.3)
        ax.set_xlim(t_states_s[0], t_states_s[-1])
        ax.set_title(title, fontsize=10, pad=3)
        if show_xlabel:
            ax.set_xlabel("time (s)", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(alpha=0.2, linewidth=0.4)

    _plot_band(ax_q1, 0, r"$q_1$", "rad", show_xlabel=False)
    _plot_band(ax_q2, 1, r"$q_2$", "rad", show_xlabel=False)
    _plot_band(ax_dq1, 2, r"$\dot{q}_1$", "rad/s", show_xlabel=False)
    _plot_band(ax_dq2, 3, r"$\dot{q}_2$", "rad/s", show_xlabel=False)

    handles_2d = [
        Line2D([0], [0], color="black", linewidth=1.7),
        Line2D([0], [0], color="red", linewidth=1.3),
        Line2D([0], [0], color="0.55", linewidth=6, alpha=0.35),
        Line2D([0], [0], color=edge_color, linewidth=1.3, linestyle="-"),
    ]
    ax_q1.legend(
        handles_2d,
        ["nominal", "plant", "funnel", "feasible bounds"],
        loc="lower right",
        bbox_to_anchor=(1.0, 0.12),
        frameon=False,
        fontsize=8,
        borderpad=0.2,
        labelspacing=0.25,
    )

    fig_slices.tight_layout(pad=0.5)

    fig_inputs3d = None
    if R_caps is not None and (args.save_input_funnel or not args.no_show):
        U_center = Uhat[: min(len(R_caps), Uhat.shape[0])]
        R_diag_inputs = R_caps[: U_center.shape[0]]
        t_inputs = np.arange(U_center.shape[0], dtype=float) * dt
        U_center_up, R_diag_up, t_inputs_up = _upsample_inputs(U_center, R_diag_inputs, t_inputs, factor=args.upsample)
        num_theta_inputs = max(64, args.num_theta)
        X_in, Y_in, Z_in = build_input_funnel_surface(U_center_up, R_diag_up, t_inputs_up, num_points=num_theta_inputs)

        fig_inputs3d = plt.figure(figsize=(6.2, 5.0))
        ax_inputs3d = fig_inputs3d.add_subplot(111, projection="3d")
        ax_inputs3d.plot_surface(X_in, Y_in, Z_in, color="0.7", alpha=0.35, linewidth=0, antialiased=True)
        plant_tau1 = np.interp(t_inputs_up, np.arange(Uplant.shape[0], dtype=float) * dt, Uplant[:, 0])
        plant_tau2 = np.interp(t_inputs_up, np.arange(Uplant.shape[0], dtype=float) * dt, Uplant[:, 1])
        ax_inputs3d.plot(U_center_up[:, 0], U_center_up[:, 1], t_inputs_up, color="black", linewidth=1.8, label="nominal")
        ax_inputs3d.plot(plant_tau1, plant_tau2, t_inputs_up, color="red", linewidth=1.4, label="plant")
        ax_inputs3d.set_xlabel("τ1 (N·m)", labelpad=4)
        ax_inputs3d.set_ylabel("τ2 (N·m)", labelpad=4)
        ax_inputs3d.set_zlabel("time (s)", labelpad=8)
        ax_inputs3d.set_title("Inputs funnel", pad=1.0, fontsize=11)
        ax_inputs3d.dist = 8.6
        for axis in (ax_inputs3d.xaxis, ax_inputs3d.yaxis, ax_inputs3d.zaxis):
            axis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            axis._axinfo["grid"]["linewidth"] = 0.28
            axis._axinfo["grid"]["color"] = (0.75, 0.75, 0.75, 0.25)
        ax_inputs3d.view_init(elev=28, azim=-50)
        try:
            ax_inputs3d.set_box_aspect((1.0, 1.0, 0.7), zoom=1.0)
        except AttributeError:
            pass

        verts_in = [
            [(U_low[0], U_low[1], t_inputs_up[0]), (U_high[0], U_low[1], t_inputs_up[0]), (U_high[0], U_high[1], t_inputs_up[0]), (U_low[0], U_high[1], t_inputs_up[0])],
            [(U_low[0], U_low[1], t_inputs_up[-1]), (U_high[0], U_low[1], t_inputs_up[-1]), (U_high[0], U_high[1], t_inputs_up[-1]), (U_low[0], U_high[1], t_inputs_up[-1])],
            [(U_low[0], U_low[1], t_inputs_up[0]), (U_high[0], U_low[1], t_inputs_up[0]), (U_high[0], U_low[1], t_inputs_up[-1]), (U_low[0], U_low[1], t_inputs_up[-1])],
            [(U_high[0], U_low[1], t_inputs_up[0]), (U_high[0], U_high[1], t_inputs_up[0]), (U_high[0], U_high[1], t_inputs_up[-1]), (U_high[0], U_low[1], t_inputs_up[-1])],
            [(U_high[0], U_high[1], t_inputs_up[0]), (U_low[0], U_high[1], t_inputs_up[0]), (U_low[0], U_high[1], t_inputs_up[-1]), (U_high[0], U_high[1], t_inputs_up[-1])],
            [(U_low[0], U_high[1], t_inputs_up[0]), (U_low[0], U_low[1], t_inputs_up[0]), (U_low[0], U_low[1], t_inputs_up[-1]), (U_low[0], U_high[1], t_inputs_up[-1])],
        ]
        box_inputs = Poly3DCollection(verts_in, alpha=0.03, facecolor=face_color, edgecolor=edge_color, linewidths=0.45)
        ax_inputs3d.add_collection3d(box_inputs)
        edge_idx = [
            (0, 4), (1, 5), (2, 6), (3, 7),
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
        ]
        verts_arr = np.array([
            [U_low[0], U_low[1], t_inputs_up[0]],
            [U_high[0], U_low[1], t_inputs_up[0]],
            [U_high[0], U_high[1], t_inputs_up[0]],
            [U_low[0], U_high[1], t_inputs_up[0]],
            [U_low[0], U_low[1], t_inputs_up[-1]],
            [U_high[0], U_low[1], t_inputs_up[-1]],
            [U_high[0], U_high[1], t_inputs_up[-1]],
            [U_low[0], U_high[1], t_inputs_up[-1]],
        ])
        for i0, i1 in edge_idx:
            xs, ys, zs = zip(verts_arr[i0], verts_arr[i1])
            ax_inputs3d.plot(xs, ys, zs, color=edge_color, linestyle=(0, (2.0, 3.0)), linewidth=0.6)
        ax_inputs3d.legend(loc="upper left", frameon=False, fontsize=8)
        fig_inputs3d.tight_layout(pad=0.3)

    # --- Lyapunov figure ---
    fig_lya = None
    if args.save_lyapunov or not args.no_show:
        fig_lya, ax_lya = plt.subplots(1, 1, figsize=(7.0, 2.6))
        ax_lya.plot(t_states, V_vals, color="red", linewidth=1.8, label=r"$V(k)$")
        ax_lya.axhline(1.0, color="black", linestyle="--", linewidth=1.1, label="invariance")
        N_total = Xhat.shape[0] - 1
        n_segs = int(np.ceil(N_total / T))
        for i in range(n_segs):
            k0 = i * T
            k1 = min((i + 1) * T, N_total)
            t0_seg = t_states[k0]
            t1_seg = t_states[min(k1 + 1, len(t_states) - 1)]
            color_seg = "#c9e4ff" if (i % 2 == 0) else "#f0f5ff"
            ax_lya.axvspan(
                t0_seg,
                t1_seg,
                color=color_seg,
                alpha=0.2,
                linewidth=0,
                zorder=-1,
            )
            if i < n_segs - 1:
                t_boundary = t_states[min((i + 1) * T, len(t_states) - 1)]
                ax_lya.axvline(t_boundary, color="#999999", linestyle="--", linewidth=0.6, alpha=0.6)
            if i > 0:
                y0 = V_vals[k0]
                y1 = y0 + 0.01 * (t1_seg - t0_seg)
                ax_lya.plot([t0_seg, t1_seg], [y0, y1], color="gray", linewidth=0.6, linestyle="-")
        ax_lya.set_xlabel("time (s)", fontsize=9)
        ax_lya.set_ylabel(r"$V(k)$", fontsize=9)
        ax_lya.set_xlim(t_states[0], t_states[-1])
        ax_lya.tick_params(axis="both", labelsize=8)
        ax_lya.grid(alpha=0.25, linewidth=0.5)
        ax_lya.legend(loc="upper right", bbox_to_anchor=(1.00, 0.88), frameon=False, fontsize=8)
        fig_lya.tight_layout(pad=0.4)

    if args.save:
        out_path = pathlib.Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig3d.savefig(out_path, dpi=300)
        print(f"Saved 3D figure ➜ {out_path}")

    if args.save_slices:
        out_slices = pathlib.Path(args.save_slices)
        out_slices.parent.mkdir(parents=True, exist_ok=True)
        fig_slices.savefig(out_slices, dpi=300)
        print(f"Saved slice figure ➜ {out_slices}")

    if args.save_lyapunov and fig_lya is not None:
        out_lya = pathlib.Path(args.save_lyapunov)
        out_lya.parent.mkdir(parents=True, exist_ok=True)
        fig_lya.savefig(out_lya, dpi=300)
        print(f"Saved Lyapunov figure ➜ {out_lya}")

    if args.save_input_funnel:
        if fig_inputs3d is not None:
            out_inputs = pathlib.Path(args.save_input_funnel)
            out_inputs.parent.mkdir(parents=True, exist_ok=True)
            fig_inputs3d.savefig(out_inputs, dpi=300)
            print(f"Saved input funnel ➜ {out_inputs}")
        else:
            print("[paper_results] Warning: R_tv_diag not found in run; skipping input funnel figure.")

    if args.no_show:
        plt.close(fig3d)
        plt.close(fig_slices)
        if fig_lya is not None:
            plt.close(fig_lya)
        if fig_inputs3d is not None:
            plt.close(fig_inputs3d)
    else:
        plt.show()


if __name__ == "__main__":
    main()
