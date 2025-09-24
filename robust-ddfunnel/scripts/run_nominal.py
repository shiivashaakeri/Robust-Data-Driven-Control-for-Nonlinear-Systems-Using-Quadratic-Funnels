# scripts/run_nominal.py
"""
Generate a nominal trajectory on the digital twin using the LQR-based nominal planner,
save plots & trajectory, and (optionally) save a GIF animation (no timestamps in filenames).

Defaults (can be changed via CLI):
  data/nominal_trajectory/nominal_<slug>.npz
  results/nominal_plots/states_<slug>.png
  results/nominal_plots/inputs_<slug>.png
  results/nominal_anim/anim_<slug>.gif    (when --save-anim is used)

Examples:
  python scripts/run_nominal.py
  python scripts/run_nominal.py --N 1500 --tag far_goal --speed 1.2 --save-anim --fps 30
  python scripts/run_nominal.py --out-data data/nominal --out-plots results/nominal --out-anim results/anim
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

# ensure src/ is importable
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from core.constants_io import NominalConstants, save_constants_yaml  # pyright: ignore[reportMissingImports]
from models.discretization import discrete_jacobians_fd, make_stepper  # pyright: ignore[reportMissingImports]
from models.robot_arm_2dof import RobotArm2DOF  # pyright: ignore[reportMissingImports]
from nominal.trajectory import NominalLQRSettings, lqr_nominal_plan  # pyright: ignore[reportMissingImports]
from utils.config import build_robot_arm_demo_config, load_yaml  # pyright: ignore[reportMissingImports]
from viz.anim_arm import animate_two_link, save_gif, spec_from_model  # pyright: ignore[reportMissingImports]
from viz.plots import plot_inputs, plot_states  # pyright: ignore[reportMissingImports]

# ---------------------------
# Helpers
# ---------------------------


def _as_vec(x: Any, dim: int) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size != dim:
        raise ValueError(f"expected vector of length {dim}, got {arr.size}")
    return arr


def _as_matrix(diag_or_full: Any, dim: int) -> np.ndarray:
    arr = np.asarray(diag_or_full, dtype=float)
    if arr.ndim == 0:  # scalar -> scalar*I
        return np.diag(np.full(dim, float(arr)))
    if arr.ndim == 1:  # diagonal entries
        if arr.size != dim:
            raise ValueError(f"diag length {arr.size} != {dim}")
        return np.diag(arr)
    if arr.shape == (dim, dim):
        return arr
    raise ValueError(f"matrix must be ({dim},{dim}) or diag length {dim}")


def _load_nominal_from_yaml(path: pathlib.Path, U_box: Tuple[np.ndarray, np.ndarray], n: int, m: int) -> Dict[str, Any]:
    raw = load_yaml(path)
    u_low_default, u_high_default = U_box
    u_goal_raw = raw.get("u_goal", "auto")
    u_goal = "auto" if isinstance(u_goal_raw, str) and u_goal_raw.lower() == "auto" else _as_vec(u_goal_raw, m)
    return {
        "x_goal": _as_vec(raw.get("x_goal", np.zeros(n)), n),
        "u_goal": u_goal,  # may be "auto" or (m,)
        "Q": _as_matrix(raw.get("Q", np.ones(n)), n),
        "R": _as_matrix(raw.get("R", 0.1 * np.ones(m)), m),
        "Qf": _as_matrix(raw.get("Qf", 5.0 * np.ones(n)), n),
        "N": int(raw.get("N", 1000)),
        "u_low": _as_vec(raw.get("u_low", u_low_default), m),
        "u_high": _as_vec(raw.get("u_high", u_high_default), m),
        "riccati_tol": float(raw.get("riccati_tol", 1e-9)),
        "riccati_maxit": int(raw.get("riccati_maxit", 10000)),
        "fd_eps": float(raw.get("fd_eps", 1e-6)),
    }


def _gravity_torque_for_goal(twin: RobotArm2DOF, x_goal: np.ndarray) -> np.ndarray:
    qg = np.asarray(x_goal[:2], float)
    # Public wrapper would be nicer; internal is fine here.
    return twin._gravity(qg).copy()  # type: ignore[attr-defined]


def _resolve_nominal(
    args,
    cfg_path: pathlib.Path,
    extras: Dict[str, Any],
    U_box: Tuple[np.ndarray, np.ndarray],
    n: int,
    m: int,
) -> Tuple[Dict[str, Any], str]:
    """Precedence:
    1) --nom
    2) robot cfg: nominal_file
    3) configs/nominal_lqr.yaml if present
    4) robot cfg: embedded 'nominal' block
    5) built-in defaults
    """
    # 1) --nom
    if args.nom:
        return _load_nominal_from_yaml(pathlib.Path(args.nom), U_box, n, m), f"--nom {args.nom}"

    # 2) nominal_file in robot cfg (top-level)
    nom_file = extras.get("nominal_file")
    if isinstance(nom_file, str):
        p = pathlib.Path(nom_file)
        if not p.is_absolute():
            proj_root = cfg_path.parents[1]
            p_cfg = cfg_path.parent / nom_file
            p = p_cfg if p_cfg.exists() else (proj_root / nom_file)
        if p.exists():
            return _load_nominal_from_yaml(p, U_box, n, m), f"robot_cfg.nominal_file: {p}"

    # 3) default nominal next to cfg
    default_nom = cfg_path.parent / "nominal_lqr.yaml"
    if default_nom.exists():
        return _load_nominal_from_yaml(default_nom, U_box, n, m), f"default nominal: {default_nom}"

    # 4) embedded nominal block
    if "nominal" in extras and isinstance(extras["nominal"], dict):
        nom = extras["nominal"]
        if isinstance(nom.get("u_goal", "auto"), str) and nom["u_goal"].lower() == "auto":
            nom["u_goal"] = "auto"
        return nom, "robot_cfg.nominal block"

    # 5) hardcoded safe defaults
    u_low, u_high = U_box
    nom = {
        "x_goal": np.array([0.6, -0.3, 0.0, 0.0]),
        "u_goal": "auto",
        "Q": np.diag([20.0, 20.0, 2.0, 2.0]),
        "R": np.diag([0.1, 0.1]),
        "Qf": np.diag([80.0, 80.0, 8.0, 8.0]),
        "N": 1000,
        "u_low": u_low,
        "u_high": u_high,
        "riccati_tol": 1e-9,
        "riccati_maxit": 10000,
        "fd_eps": 1e-6,
    }
    return nom, "built-in defaults"


def _slug_from_goal(x_goal: np.ndarray, N: int, tag: str | None) -> str:
    q1, q2 = float(x_goal[0]), float(x_goal[1])
    base = f"q1_{q1:+.2f}_q2_{q2:+.2f}_N{N}"
    return f"{tag}_{base}" if tag else base


# ---------------------------
# Main
# ---------------------------


def main():  # noqa: PLR0915
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        default=str(pathlib.Path("configs/robot_arm_2dof.yaml")),
        help="Robot config YAML (plant/twin/constraints/discretization)",
    )
    parser.add_argument(
        "--nom",
        type=str,
        default=None,
        help="Optional separate nominal LQR YAML; else auto-detects configs/nominal_lqr.yaml",
    )
    parser.add_argument("--N", type=int, default=None, help="Override nominal horizon N")
    parser.add_argument("--speed", type=float, default=1.0, help="Animation speed factor")
    parser.add_argument("--trail", type=int, default=200, help="End-effector trail length")
    parser.add_argument("--no-anim", action="store_true", help="Skip on-screen animation")
    parser.add_argument(
        "--no-save-anim", action="store_true",
        help="Do NOT save the GIF animation (by default a GIF is saved to --out-anim/anim_<slug>.gif)"
    )
    parser.add_argument("--fps", type=int, default=30, help="GIF frames per second")
    parser.add_argument("--debug", action="store_true", help="Print stability & saturation diagnostics")
    parser.add_argument(
        "--out-data", type=str, default="data/nominal_trajectory", help="Directory to save nominal .npz"
    )
    parser.add_argument("--out-plots", type=str, default="results/nominal_plots", help="Directory to save plots")
    parser.add_argument(
        "--out-anim", type=str, default="results/nominal_anim", help="Directory to save GIF when --save-anim is set"
    )
    parser.add_argument("--tag", type=str, default=None, help="Slug prefix for filenames")
    args = parser.parse_args()

    # --- Load robot config ---
    cfg_path = pathlib.Path(args.cfg)
    raw = load_yaml(cfg_path)
    demo, extras = build_robot_arm_demo_config(raw)

    dt = float(demo.discretization.dt)
    n, m = 4, 2
    Ulow, Uhigh = demo.U.low, demo.U.high

    # --- Nominal parameters (auto-detect source) ---
    nom, nom_src = _resolve_nominal(args, cfg_path, extras, (Ulow, Uhigh), n, m)
    if args.N is not None:
        nom["N"] = int(args.N)
    print(f"[run_nominal] Using nominal from: {nom_src}")

    # Initial twin state
    x0_twin = extras.get("x0_twin", np.array([0.28, -0.22, 0.0, 0.0], dtype=float))

    # --- Build twin ---
    twin = RobotArm2DOF(demo.twin, dt=dt, integrator=demo.discretization.method)

    # Resolve u_goal: "auto" -> gravity feedforward at x_goal
    u_goal_raw = nom.get("u_goal", "auto")
    if isinstance(u_goal_raw, str) and u_goal_raw.lower() == "auto":
        u_goal_final = _gravity_torque_for_goal(twin, nom["x_goal"])
        auto_note = " (auto: gravity feedforward)"
    else:
        u_goal_final = np.asarray(u_goal_raw, float).reshape(m)
        auto_note = ""

    # --- Plan nominal on the twin ---
    settings = NominalLQRSettings(
        Q=np.asarray(nom["Q"], float),
        R=np.asarray(nom["R"], float),
        Qf=np.asarray(nom["Qf"], float),
        N=int(nom["N"]),
        x_goal=np.asarray(nom["x_goal"], float),
        u_goal=u_goal_final,
        u_low=np.asarray(nom["u_low"], float) if nom.get("u_low") is not None else None,
        u_high=np.asarray(nom["u_high"], float) if nom.get("u_high") is not None else None,
        riccati_tol=float(nom.get("riccati_tol", 1e-9)),
        riccati_maxit=int(nom.get("riccati_maxit", 10000)),
        fd_eps=float(nom.get("fd_eps", 1e-6)),
        ramp_seconds=float(nom.get("ramp_seconds", 2.0)),
        input_smoothing_alpha=float(nom.get("input_smoothing_alpha", 0.35)),
        gravity_feedforward=bool(nom.get("gravity_feedforward", True)),
    )

    Xhat, Uhat, K, v = lqr_nominal_plan(twin, settings, x0=np.asarray(x0_twin, float))

    # --- Optional diagnostics ---
    if args.debug:
        g_goal = _gravity_torque_for_goal(twin, settings.x_goal)
        print("gravity at goal G(q*):", g_goal, "   limits:", demo.U.low, demo.U.high)
        step = make_stepper(twin.f, twin.dt, method=twin._integrator_name)
        A_goal, B_goal = discrete_jacobians_fd(step, settings.x_goal, settings.u_goal, eps=settings.fd_eps)
        Acl = A_goal - B_goal @ K
        eig = np.linalg.eigvals(Acl)
        print("rho(A-BK) at goal =", np.max(np.abs(eig)))
        sat = np.logical_or(Uhat <= demo.U.low + 1e-9, Uhat >= demo.U.high - 1e-9)
        print("saturation %  [tau1, tau2] =", sat.mean(axis=0) * 100.0)

    # --- Report (concise) ---
    np.set_printoptions(precision=4, suppress=True)
    print("\n=== Nominal (Twin) ===")
    print(f"Horizon N     : {settings.N}   dt: {dt}")
    print(f"x0 (twin)     : {x0_twin}")
    print(f"x_goal        : {settings.x_goal}")
    print(f"u_goal        : {settings.u_goal}{auto_note}")
    print("K (DLQR gain) :\n", K)
    print(f"increment bound v: {v:.6f}")
    print(f"Xhat shape {Xhat.shape}, Uhat shape {Uhat.shape}")

    # --- Plots ---
    t_states = np.arange(settings.N + 1) * dt
    t_inputs = np.arange(settings.N) * dt
    try:
        state_names = list(demo.X.names)
    except Exception:
        state_names = ["q1 (rad)", "q2 (rad)", "dq1 (rad/s)", "dq2 (rad/s)"]
    try:
        input_names = list(demo.U.names)
    except Exception:
        input_names = ["tau1 (N·m)", "tau2 (N·m)"]

    fig_s, _ = plot_states(
    [{"label": "Twin nominal", "t": t_states, "X": Xhat, "style": {"linewidth": 1.6}}],
    names=state_names,
    title="Nominal states (digital twin)",
    x_bounds=(demo.X.low, demo.X.high),
    show_bounds_in_legend=False,
    )

    fig_u, _ = plot_inputs(
        [{"label": "Twin nominal", "t": t_inputs, "U": Uhat, "style": {"linewidth": 1.6}}],
        names=input_names,
        title="Nominal inputs (digital twin)",
        u_bounds=(demo.U.low, demo.U.high),
        show_bounds_in_legend=False,
    )

    # --- Save artifacts (no timestamps) ---
    out_data_dir = pathlib.Path(args.out_data)
    out_data_dir.mkdir(parents=True, exist_ok=True)
    out_plots_dir = pathlib.Path(args.out_plots)
    out_plots_dir.mkdir(parents=True, exist_ok=True)
    out_anim_dir = pathlib.Path(args.out_anim)
    out_anim_dir.mkdir(parents=True, exist_ok=True)

    slug = _slug_from_goal(settings.x_goal, settings.N, args.tag)

    data_path = out_data_dir / f"nominal_{slug}.npz"
    np.savez_compressed(
        data_path,
        Xhat=Xhat,
        Uhat=Uhat,
        K=K,
        v=v,
        dt=dt,
        x_goal=settings.x_goal,
        u_goal=settings.u_goal,
        Q=settings.Q,
        R=settings.R,
        Qf=settings.Qf,
        N=settings.N,
        u_low=(settings.u_low if settings.u_low is not None else np.array([])),
        u_high=(settings.u_high if settings.u_high is not None else np.array([])),
    )

    fig_s_path = out_plots_dir / f"states_{slug}.png"
    fig_u_path = out_plots_dir / f"inputs_{slug}.png"
    fig_s.savefig(fig_s_path, dpi=200)
    fig_u.savefig(fig_u_path, dpi=200)
    plt.close(fig_s)
    plt.close(fig_u)

    print(f"\nSaved nominal NPZ ➜ {data_path}")
    print(f"Saved plots       ➜ {fig_s_path}\n                       {fig_u_path}")

    # --- Animate (save by default) ---
    if not args.no_anim or not args.no_save_anim:
        spec = spec_from_model(twin, Xhat, name="Twin nominal", linestyle="-", linewidth=3.0)
        fig, anim = animate_two_link(
            [spec], dt=dt, speed=args.speed, trail=args.trail, title="Nominal trajectory (digital twin)"
        )

        # Save GIF unless the user opted out
        if not args.no_save_anim:
            gif_path = out_anim_dir / f"anim_{slug}.gif"
            try:
                save_gif(fig, anim, gif_path, fps=args.fps, dpi=120)
                print(f"Saved GIF         ➜ {gif_path}")
            except Exception as e:
                print(f"[warn] Failed to save GIF: {e}")

        # Show on-screen if not suppressed
        if not args.no_anim:
            plt.show()
        else:
            plt.close(fig)

    consts_dir = Path("data/nominal_constants")
    consts_dir.mkdir(parents=True, exist_ok=True)
    consts_path = consts_dir / f"constants_{slug}.yaml"

    consts = NominalConstants(
        dt = float(dt),
        N = int(settings.N),
        v = float(v),
        gamma = None,
        L_J = None,
        L_r = None,
        C = None,
        T_tilde = None,
        x_goal = settings.x_goal,
        u_goal = settings.u_goal,
    )
    save_constants_yaml(consts_path, consts)
    print(f"Saved constants   ➜ {consts_path}")


if __name__ == "__main__":
    main()
