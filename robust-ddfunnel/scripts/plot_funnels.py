# scripts/plot_funnels.py
"""
Plot nominal/plant trajectories with per-step state funnel bands.

Usage:
  python scripts/plot_funnels.py \
    --cfg configs/robot_arm_2dof.yaml \
    --run results/online_runs/q1_+4.00_q2_-1.00_N1000/online_run_q1_+4.00_q2_-1.00_N1000.npz \
    --save
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import List

import matplotlib.pyplot as plt
import numpy as np

# ensure src/ is importable
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from synthesis.recover_gain import recover_gain  # pyright: ignore[reportMissingImports]
from utils.config import build_robot_arm_demo_config, load_yaml  # pyright: ignore[reportMissingImports]
from viz.plots import (  # pyright: ignore[reportMissingImports]
    overlay_state_funnel,
    plot_inputs,
    plot_states,
    shade_input_ellipsoid_bands,
)


# ---------- SPD helpers for smoothing ----------
def _spd_eig_fun(Q: np.ndarray, fun, eps: float = 1e-12) -> np.ndarray:
    Q = np.asarray(Q, float)
    S = 0.5 * (Q + Q.T)
    w, V = np.linalg.eigh(S)
    w = np.clip(w, eps, None)
    return (V * fun(w)) @ V.T


def spd_log(Q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return _spd_eig_fun(Q, np.log, eps)


def spd_exp(S: np.ndarray) -> np.ndarray:
    S = 0.5 * (S + S.T)
    return _spd_eig_fun(S, np.exp)


def spd_geodesic_blend(QA: np.ndarray, QB: np.ndarray, w: float) -> np.ndarray:
    w = float(np.clip(w, 0.0, 1.0))
    LA = spd_log(QA)
    LB = spd_log(QB)
    return spd_exp((1.0 - w) * LA + w * LB)


def smooth_Qseq_around_boundaries(Q_seq: np.ndarray, T: int, *, blend_steps: int, ramp: str = "cosine") -> np.ndarray:
    """
    Smooth piecewise-constant per-step Q_seq (T_s,n,n) around segment boundaries k=(i+1)*T.
    Uses log-Euclidean blending between Q[k_c-1] and Q[k_c] in a window [k_c-r, k_c+r].
    """
    Q_seq = np.array(Q_seq, float, copy=True)
    T_s, n, _ = Q_seq.shape
    N = T_s - 1  # states length
    r = int(max(0, blend_steps))
    if r == 0:
        return Q_seq

    def w_smooth(u):
        if ramp == "linear":
            return u
        return 0.5 - 0.5 * np.cos(np.pi * u)  # cosine ease

    # iterate over segment boundaries (skip the final terminal boundary)
    n_segs = int(np.ceil(N / T))
    for i in range(n_segs - 1):
        k_c = min((i + 1) * T, N)
        if k_c <= 0 or k_c >= T_s:
            continue

        Q_prev = Q_seq[k_c - 1]  # last step of prev segment
        Q_next = Q_seq[k_c]  # first step of next segment

        k_lo = max(0, k_c - r)
        k_hi = min(N, k_c + r)
        denom = max(1, k_hi - k_lo)

        for k in range(k_lo, k_hi + 1):
            u = (k - k_lo) / denom
            w = w_smooth(u)
            Q_seq[k] = spd_geodesic_blend(Q_prev, Q_next, w)

    # ensure terminal is consistent with last segment
    Q_seq[-1] = Q_seq[-2]
    return Q_seq


# ------------------------------------------------


def _default_state_names(n: int) -> List[str]:
    if n == 4:
        return ["q1", "q2", "dq1", "dq2"]
    return [f"x{i + 1}" for i in range(n)]


def _default_input_names(m: int) -> List[str]:
    if m == 2:
        return ["tau1", "tau2"]
    return [f"u{i + 1}" for i in range(m)]


def main():  # noqa: C901, PLR0915, PLR0912
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", type=str, required=True, help="Robot config YAML (for names & bounds)")
    p.add_argument("--run", type=str, required=True, help="Path to saved online_run_*.npz")
    p.add_argument("--save", action="store_true", help="Save figure to <run_dir>/states_with_funnel.png")
    p.add_argument("--no-show", action="store_true", help="Do not display the figure")
    p.add_argument(
        "--blend-frac", type=float, default=0.15, help="Blend half-window as a fraction of T (0 disables if <=0)"
    )
    p.add_argument(
        "--blend-steps",
        type=int,
        default=None,
        help="Override blend half-window size in steps (takes precedence over --blend-frac)",
    )
    p.add_argument("--ramp", type=str, choices=["cosine", "linear"], default="cosine", help="Smoothing ramp shape")
    args = p.parse_args()

    cfg_path = pathlib.Path(args.cfg)
    run_path = pathlib.Path(args.run)

    # --- Load config (bounds & names) ---
    raw = load_yaml(cfg_path)
    demo, extras = build_robot_arm_demo_config(raw)
    X_low = np.asarray(demo.X.low, dtype=float).reshape(-1)
    X_high = np.asarray(demo.X.high, dtype=float).reshape(-1)
    U_low = np.asarray(demo.U.low, dtype=float).reshape(-1)
    U_high = np.asarray(demo.U.high, dtype=float).reshape(-1)

    # --- Load run artifact ---
    Z = np.load(run_path, allow_pickle=True)
    Xhat = np.asarray(Z["Xhat"], float)
    Xplant = np.asarray(Z["Xplant"], float)
    Uhat = np.asarray(Z["Uhat"], float)
    Uplant = np.asarray(Z["Uapplied"], float) if "Uapplied" in Z else np.asarray(Z["Up"], float)
    Ks = np.asarray(Z["Ks"], float)
    Qs_sdp = np.asarray(Z["Qs"], float) if "Qs" in Z else None
    Ys_sdp = np.asarray(Z["Ys"], float) if "Ys" in Z else None
    constants = {}
    if "constants" in Z:
        const_raw = Z["constants"]
        if isinstance(const_raw, dict):
            constants = const_raw
        elif hasattr(const_raw, "item"):
            try:
                maybe_dict = const_raw.item()
                if isinstance(maybe_dict, dict):
                    constants = maybe_dict
            except ValueError:
                pass
    dt = float(Z["dt"])
    T = int(Z["T"])
    Q_seq_states = np.asarray(Z["Q_seq_states"], float)

    # sanity
    T_s, n = Xhat.shape
    assert Xplant.shape == (T_s, n), "Xplant must be (N+1, n)"
    assert Q_seq_states.shape == (T_s, n, n), "Q_seq_states must be (N+1, n, n)"

    assert Uhat.shape[0] == T_s - 1, "Uhat must have length N = T_s - 1"
    assert Uplant.shape == Uhat.shape, "Uapplied must match Uhat shape"

    names_x = extras.get("names_x") or _default_state_names(n)
    names_u = extras.get("names_u") or _default_input_names(Uhat.shape[1])
    t_states = np.arange(T_s, dtype=float) * dt
    t_inputs = np.arange(Uhat.shape[0], dtype=float) * dt

    # --- Smooth the funnel across segment boundaries (optional) ---
    r = max(0, int(args.blend_steps)) if args.blend_steps is not None else max(0, int(max(0.0, args.blend_frac) * T))

    if r > 0:
        Q_seq_plot = smooth_Qseq_around_boundaries(Q_seq_states, T, blend_steps=r, ramp=args.ramp)
    else:
        Q_seq_plot = Q_seq_states

    # --- Plot trajectories + bounds ---
    series = [
        {"label": "nominal", "t": t_states, "X": Xhat, "style": {"lw": 1.6}},
        {"label": "plant", "t": t_states, "X": Xplant, "style": {"lw": 1.4, "ls": "--"}},
    ]
    fig_states, axs_states = plot_states(
        series,
        names_x,
        title="States with Funnel Bands",
        x_bounds=(X_low, X_high),
        show_bounds_in_legend=True,
    )

    # --- Overlay funnel bands ---
    overlay_state_funnel(axs_states, t_states, Xhat, Q_seq_plot, label="funnel")

    # --- Build per-step gain sequence ---
    if Ks.ndim == 2:
        Ks = Ks[np.newaxis, ...]
    if Ks.ndim != 3:
        raise ValueError(f"Ks must be (n_segs, m, n); got shape {Ks.shape}")
    n_segs = int(np.ceil(Uhat.shape[0] / T))
    m = Ks.shape[1]
    if Ks.shape[2] != n:
        raise ValueError("Gain shape mismatch between Ks and state dimension")
    if Ks.shape[0] < n_segs:
        pad = np.repeat(Ks[-1:, :, :], n_segs - Ks.shape[0], axis=0)
        Ks = np.concatenate([Ks, pad], axis=0)

    K_seq = np.zeros((Uhat.shape[0], m, n), dtype=float)
    for i in range(n_segs):
        k0 = i * T
        k1 = min((i + 1) * T, Uhat.shape[0])
        K_seq[k0:k1] = Ks[min(i, Ks.shape[0] - 1)]

    # --- Rebuild gains from raw SDP outputs for plotting-only funnels ---
    K_seq_plot = np.array(K_seq, copy=True)
    if Qs_sdp is not None and Ys_sdp is not None and Qs_sdp.size > 0 and Ys_sdp.size > 0:
        Qs_sdp = np.asarray(Qs_sdp, float)
        Ys_sdp = np.asarray(Ys_sdp, float)
        if Qs_sdp.ndim == 3 and Ys_sdp.ndim == 3 and Qs_sdp.shape == (Ys_sdp.shape[0], n, n):
            sdp_Ks: list[np.ndarray] = []
            for idx in range(Qs_sdp.shape[0]):
                try:
                    K_raw = recover_gain(Ys_sdp[idx], Qs_sdp[idx])
                except Exception:
                    K_raw = Ks[min(idx + 1, Ks.shape[0] - 1)]
                sdp_Ks.append(np.asarray(K_raw, float))
            if sdp_Ks:
                for seg in range(1, n_segs):
                    k0 = seg * T
                    k1 = min((seg + 1) * T, Uhat.shape[0])
                    K_seg = sdp_Ks[min(seg - 1, len(sdp_Ks) - 1)]
                    K_seq_plot[k0:k1] = K_seg


    # --- Optional caps from nominal R_max (segment 0 reference) ---
    R_caps_inputs = None
    if constants:
        R_diag = constants.get("R_tv_diag")
        if R_diag is not None:
            R_diag = np.asarray(R_diag, float)
            if R_diag.ndim == 2 and R_diag.shape[0] >= Uhat.shape[0] and R_diag.shape[1] == m:
                R_caps_inputs = np.array([np.diag(R_diag[k]) for k in range(Uhat.shape[0])], dtype=float)

    # --- Push state funnels to input space ---
    Q_for_inputs = Q_seq_plot[:-1]
    R_seq_inputs = np.empty((Uhat.shape[0], m, m), dtype=float)
    K_seq_T = np.swapaxes(K_seq_plot, 1, 2)
    min(T, Uhat.shape[0])
    for k in range(Uhat.shape[0]):
        R_nom = K_seq_plot[k] @ Q_for_inputs[k] @ K_seq_T[k]
        diag_nom = np.clip(np.diag(R_nom), 0.0, None)

        if R_caps_inputs is not None:
            cap_diag = np.clip(np.diag(R_caps_inputs[k]), 0.0, None)
            diag_use = cap_diag
        else:
            diag_use = diag_nom

        R_seq_inputs[k] = np.diag(diag_use)

    # --- Plot inputs + overlay funnel ---
    input_series = [
        {"label": "nominal", "t": t_inputs, "U": Uhat, "style": {"lw": 1.6}},
        {"label": "plant", "t": t_inputs, "U": Uplant, "style": {"lw": 1.4, "ls": "--"}},
    ]
    fig_inputs, axs_inputs = plot_inputs(
        input_series,
        names_u,
        title="Inputs with Funnel Bands",
        u_bounds=(U_low, U_high),
        show_bounds_in_legend=True,
    )
    shade_input_ellipsoid_bands(axs_inputs, t_inputs, Uhat, R_seq_inputs, label="input funnel")

    fig_states.tight_layout()

    figs = [fig_states, fig_inputs]

    if args.save:
        out_state_path = run_path.parent / "states_with_funnel.png"
        fig_states.savefig(out_state_path, dpi=200)
        print(f"Saved ➜ {out_state_path}")

        out_input_path = run_path.parent / "inputs_with_funnel.png"
        fig_inputs.savefig(out_input_path, dpi=200)
        print(f"Saved ➜ {out_input_path}")

    if not args.no_show:
        plt.show()
    else:
        for fig in figs:
            plt.close(fig)


if __name__ == "__main__":
    main()
