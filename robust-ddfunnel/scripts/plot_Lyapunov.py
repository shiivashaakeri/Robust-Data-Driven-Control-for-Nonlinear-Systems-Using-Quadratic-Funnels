# scripts/plot_lyapunov.py
"""
Compute/plot Lyapunov values using the *smoothed per-step* Q:
    V(k) = eta_k^T Q_k^{-1} eta_k  with  Q_k = Q_seq_states[k]

Normalization options:
  --norm radius|unit|bymax|byp95|segment_p95|zscore|none
  --clip-above-1  (only meaningful for 'unit' or 'radius')

Usage:
  python scripts/plot_lyapunov.py \
    --cfg configs/robot_arm_2dof.yaml \
    --run results/online_runs/q1_+4.00_q2_-1.00_N1000/online_run_q1_+4.00_q2_-1.00_N1000.npz \
    --norm radius --clip-above-1 --save
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

# ensure src/ is importable (for names/bounds, optional)
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from utils.config import build_robot_arm_demo_config, load_yaml  # type: ignore


def _quadform_solve(Q: np.ndarray, x: np.ndarray, jitter: float = 1e-9) -> float:
    Q = np.asarray(Q, float)
    x = np.asarray(x, float).reshape(-1)
    try:
        y = np.linalg.solve(Q, x)
    except np.linalg.LinAlgError:
        y = np.linalg.solve(Q + jitter * np.eye(Q.shape[0]), x)
    return float(x @ y)


def _compute_V_from_Qseq(Xhat: np.ndarray, Xplant: np.ndarray, Q_seq_states: np.ndarray) -> np.ndarray:
    eta = np.asarray(Xplant - Xhat, float)      # (N+1, n)
    Qs  = np.asarray(Q_seq_states, float)       # (N+1, n, n)
    if eta.shape[0] != Qs.shape[0] or Qs.shape[1:] != (eta.shape[1], eta.shape[1]):
        raise ValueError("Shape mismatch between Xhat/Xplant and Q_seq_states")
    V = np.empty(eta.shape[0], float)
    for k in range(eta.shape[0]):
        V[k] = _quadform_solve(Qs[k], eta[k])
    return V


def _normalize(V: np.ndarray, T: int, method: str, clip_above_1: bool) -> tuple[np.ndarray, str]:
    V = np.asarray(V, float)
    label = "V(k)"
    if method == "none":
        out = V
    elif method == "radius":
        out = np.sqrt(np.maximum(V, 0.0))
        label = r"$\rho(k)=\sqrt{V(k)}$"
    elif method == "unit":
        out = V.copy()
        label = r"$V(k)$"
    elif method == "bymax":
        m = np.max(V) if np.max(V) > 0 else 1.0
        out = V / m
        label = r"$V/\max V$"
    elif method == "byp95":
        p = np.percentile(V, 95) if np.max(V) > 0 else 1.0
        p = p if p > 0 else 1.0
        out = V / p
        label = r"$V/\mathrm{P95}(V)$"
    elif method == "segment_p95":
        N = len(V) - 1
        n_segs = int(np.ceil(N / T))
        out = np.zeros_like(V)
        for i in range(n_segs):
            k0 = i * T
            k1 = min((i + 1) * T, N)
            sl = slice(k0, k1 + 1)  # include the step at k1
            p = np.percentile(V[sl], 95)
            p = p if p > 0 else 1.0
            out[sl] = V[sl] / p
        label = r"$V/\mathrm{P95}(V)$ (per-seg)"
    elif method == "zscore":
        mu = float(np.mean(V))
        sd = float(np.std(V)) if np.std(V) > 0 else 1.0
        out = (V - mu) / sd
        label = r"$\frac{V-\mu}{\sigma}$"
    else:
        raise ValueError(f"Unknown --norm '{method}'")

    if clip_above_1 and method in ("unit", "radius", "bymax", "byp95", "segment_p95"):
        out = np.minimum(out, 1.0)
    return out, label


def main():  # noqa: PLR0915
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", type=str, required=True, help="Robot config YAML (optional here)")
    p.add_argument("--run", type=str, required=True, help="Path to saved online_run_*.npz")
    p.add_argument("--norm", type=str, default="radius",
                   choices=["radius", "unit", "bymax", "byp95", "segment_p95", "zscore", "none"],
                   help="Normalization to apply before plotting")
    p.add_argument("--clip-above-1", action="store_true", help="Clamp values >1 down to 1 (if applicable)")
    p.add_argument("--save", action="store_true", help="Save figure to <run_dir>/lyapunov_values_<norm>.png")
    p.add_argument("--no-show", action="store_true", help="Do not display the figure")
    args = p.parse_args()

    cfg_path = pathlib.Path(args.cfg)
    run_path = pathlib.Path(args.run)
    run_dir = run_path.parent

    # optional: keep for symmetry
    raw = load_yaml(cfg_path)
    build_robot_arm_demo_config(raw)

    Z = np.load(run_path, allow_pickle=True)
    Xhat = np.asarray(Z["Xhat"], float)
    Xplant = np.asarray(Z["Xplant"], float)
    dt = float(Z["dt"])
    T = int(Z["T"])

    if "Q_seq_states" not in Z:
        raise KeyError("Run file missing 'Q_seq_states'. Save per-step smoothed Qk in run_online_ddfunnel.py first.")
    Q_seq_states = np.asarray(Z["Q_seq_states"], float)

    V = _compute_V_from_Qseq(Xhat, Xplant, Q_seq_states)
    Vn, ylabel = _normalize(V, T, args.norm, args.clip_above_1)

    t = np.arange(Xhat.shape[0], dtype=float) * dt
    fig, ax = plt.subplots(1, 1, figsize=(12, 3.8))
    ax.plot(t, Vn, lw=1.8, label=ylabel)

    # Draw the “boundary” at 1 where that makes sense
    if args.norm in ("radius", "unit", "bymax", "byp95", "segment_p95"):
        ax.axhline(1.0, ls="--", lw=1.0, c="k", alpha=0.6, label="normalized boundary = 1")

    ax.set_xlabel("time (s)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)

    # light segment shading
    N = Xhat.shape[0] - 1
    n_segs = int(np.ceil(N / T))
    for i in range(n_segs):
        k0 = i * T
        k1 = min((i + 1) * T, N)
        ax.axvspan(t[k0], t[k1], color="k", alpha=0.05 if (i % 2 == 0) else 0.0, lw=0)

    # % inside (interpreting “inside” as <=1 if boundary-based normalization)
    if args.norm in ("radius", "unit", "bymax", "byp95", "segment_p95"):
        inside = np.mean(Vn <= 1.0) * 100.0
        ax.legend(loc="upper right", ncols=2, fontsize=9, title=f"{inside:.1f}% steps ≤ boundary")
    else:
        ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()

    # save artifacts
    out_npz = run_dir / f"lyapunov_values_{args.norm}.npz"
    np.savez_compressed(out_npz, t=t, V=V, V_norm=Vn, norm=args.norm)
    print(f"Saved values ➜ {out_npz}")

    if args.save:
        out_png = run_dir / f"lyapunov_values_{args.norm}.png"
        fig.savefig(out_png, dpi=200)
        print(f"Saved figure ➜ {out_png}")

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
