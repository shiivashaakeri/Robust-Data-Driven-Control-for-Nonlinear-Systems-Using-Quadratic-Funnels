"""
Compute time-varying (box-closed-form) ellipsoids Q_k (state) and R_k (input)
for the given nominal trajectory, then UPDATE the matching constants YAML.

Usage:
  python scripts/compute_feasible_ellipsoids.py \
    --cfg configs/robot_arm_2dof.yaml \
    --nom data/nominal_trajectory/nominal_q1_+4.00_q2_-1.00_N1000.npz \
    --plot --save-plots results/nominal_plots

Notes:
- Writes into: data/nominal_constants/constants_<slug>.yaml
  (same slug as the nominal .npz)
- Stores only the diagonal entries (compact) as Q_tv_diag, R_tv_diag in YAML.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# make src importable
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from nominal.max_elliposoids import (  # pyright: ignore[reportMissingImports]
    qmax_time_varying_box,
    rmax_time_varying_box,
)
from utils.config import (  # pyright: ignore[reportMissingImports]
    build_robot_arm_demo_config,
    load_yaml,
    save_yaml,
)
from viz.plots import (  # pyright: ignore[reportMissingImports]
    plot_inputs,
    plot_states,
    shade_input_ellipsoid_bands,
    shade_state_ellipsoid_bands,
)

# ---------------------------
# Helpers
# ---------------------------


def _slug_from_npz(npz_path: pathlib.Path, Xhat: np.ndarray, Uhat: np.ndarray) -> str:
    with np.load(npz_path) as D:
        if "x_goal" in D:
            q1 = float(D["x_goal"][0])
            q2 = float(D["x_goal"][1])
        else:
            q1 = float(Xhat[-1, 0])
            q2 = float(Xhat[-1, 1])
    N = Uhat.shape[0]
    return f"q1_{q1:+.2f}_q2_{q2:+.2f}_N{N}"


def _ensure_dir(p: pathlib.Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _arr(obj: Any) -> np.ndarray:
    return np.asarray(obj, dtype=float)


def _diag_stack_to_full(diag_stack: np.ndarray) -> np.ndarray:
    # diag_stack: (T, d) -> (T, d, d)
    T, d = diag_stack.shape
    out = np.zeros((T, d, d), dtype=float)
    for k in range(T):
        out[k, np.arange(d), np.arange(d)] = diag_stack[k]
    return out


# ---------------------------
# Main
# ---------------------------


def main():  # noqa: PLR0915
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg", type=str, default="configs/robot_arm_2dof.yaml", help="Robot config YAML (for X/U boxes)."
    )
    parser.add_argument("--nom", type=str, required=True, help="Path to nominal .npz saved by run_nominal.py")
    parser.add_argument(
        "--x-ball-cap", type=float, default=None, help="Optional per-dim cap for state radii (units of state)."
    )
    parser.add_argument(
        "--u-ball-cap", type=float, default=None, help="Optional per-dim cap for input radii (units of input)."
    )
    parser.add_argument("--plot", action="store_true", help="Preview plots with shaded Q_k/R_k tubes.")
    parser.add_argument("--save-plots", type=str, default=None, help="If set, save preview PNGs to this directory.")
    args = parser.parse_args()

    cfg_path = pathlib.Path(args.cfg)
    nom_path = pathlib.Path(args.nom)

    # Load robot config (for bounds)
    raw_cfg = load_yaml(cfg_path)
    demo, _extras = build_robot_arm_demo_config(raw_cfg)
    x_low, x_high = demo.X.low, demo.X.high
    u_low, u_high = demo.U.low, demo.U.high

    # Load nominal
    with np.load(nom_path) as D:
        Xhat = _arr(D["Xhat"])
        Uhat = _arr(D["Uhat"])

    # --- Time-varying ellipsoids (box closed-form) ---
    Q_t = qmax_time_varying_box(Xhat, x_low, x_high, x_ball_cap=args.x_ball_cap)  # (N+1,n,n)
    R_t = rmax_time_varying_box(Uhat, u_low, u_high, u_ball_cap=args.u_ball_cap)  # (N,m,m)

    # --- Optional preview of tubes around nominal ---
    if args.plot or args.save_plots:
        dt = float(demo.discretization.dt)
        t_states = np.arange(Xhat.shape[0]) * dt
        t_inputs = np.arange(Uhat.shape[0]) * dt

        try:
            state_names = list(demo.X.names)
        except Exception:
            state_names = ["q1 (rad)", "q2 (rad)", "dq1 (rad/s)", "dq2 (rad/s)"]
        try:
            input_names = list(demo.U.names)
        except Exception:
            input_names = ["tau1 (N·m)", "tau2 (N·m)"]

        fig_s, axs_s = plot_states(
            [{"label": "nominal", "t": t_states, "X": Xhat, "style": {"linewidth": 1.6}}],
            names=state_names,
            title="Nominal states with time-varying tubes",
        )
        fig_u, axs_u = plot_inputs(
            [{"label": "nominal", "t": t_inputs, "U": Uhat, "style": {"linewidth": 1.6}}],
            names=input_names,
            title="Nominal inputs with time-varying tubes",
        )

        shade_state_ellipsoid_bands(axs_s, t_states, Xhat, Q_t, label="state tube")
        shade_input_ellipsoid_bands(axs_u, t_inputs, Uhat, R_t, label="input tube")

        if args.save_plots:
            out_dir = pathlib.Path(args.save_plots)
            out_dir.mkdir(parents=True, exist_ok=True)
            slug = _slug_from_npz(nom_path, Xhat, Uhat)
            fig_s.savefig(out_dir / f"states_tube_tv_{slug}.png", dpi=200)
            fig_u.savefig(out_dir / f"inputs_tube_tv_{slug}.png", dpi=200)
            print(f"Saved tube previews to {out_dir}")
            plt.close(fig_s)
            plt.close(fig_u)
        if args.plot and not args.save_plots:
            plt.show()

    # --- Save to constants YAML (compact: diagonals only) ---
    slug = _slug_from_npz(nom_path, Xhat, Uhat)
    const_path = pathlib.Path("data/nominal_constants") / f"constants_{slug}.yaml"
    _ensure_dir(const_path)

    constants = load_yaml(const_path) if const_path.exists() else {}

    # store diagonals only to keep YAML small
    Q_tv_diag = np.array([np.diag(Q_t[k]) for k in range(Q_t.shape[0])])  # (N+1,n)
    R_tv_diag = np.array([np.diag(R_t[k]) for k in range(R_t.shape[0])])  # (N,m)

    constants.update(
        {
            "ellipsoids_mode": "time_varying_box",
            "Q_tv_diag": Q_tv_diag.tolist(),
            "R_tv_diag": R_tv_diag.tolist(),
            "Q_tv_diag_shape": list(Q_tv_diag.shape),
            "R_tv_diag_shape": list(R_tv_diag.shape),
            "x_ball_cap": (float(args.x_ball_cap) if args.x_ball_cap is not None else None),
            "u_ball_cap": (float(args.u_ball_cap) if args.u_ball_cap is not None else None),
        }
    )
    save_yaml(constants, const_path)

    # Small report
    def _minmax(a):
        return float(np.min(a)), float(np.max(a))

    qmin, qmax = _minmax(Q_tv_diag)
    rmin, rmax = _minmax(R_tv_diag)
    print("\n=== Time-Varying Ellipsoids (box closed-form) ===")
    print(f"X box: low={x_low}, high={x_high}")
    print(f"U box: low={u_low}, high={u_high}")
    print(f"Q_tv_diag shape {Q_tv_diag.shape}, diag[min,max]=({qmin:.3e}, {qmax:.3e})")
    print(f"R_tv_diag shape {R_tv_diag.shape}, diag[min,max]=({rmin:.3e}, {rmax:.3e})")
    print(f"\nUpdated constants YAML ➜ {const_path}")


if __name__ == "__main__":
    main()
