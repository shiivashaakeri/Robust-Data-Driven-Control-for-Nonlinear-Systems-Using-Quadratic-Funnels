# src/nominal/plant_rollout.py
"""
Apply a saved nominal (twin) trajectory's inputs to the physical plant and compare.

What it does
------------
- Loads nominal .npz produced by scripts/run_nominal.py  (Xhat, Uhat, dt, ...)
- Builds the PHYSICAL PLANT from your robot YAML (using plant params & x0_plant)
- Rolls the plant forward OPEN-LOOP with the nominal inputs Uhat
- Plots:
    1) states (nominal vs plant)
    2) inputs (the same Uhat you applied)
    3) state deviation (plant - nominal)
- (Optional) animation overlay: nominal vs plant

CLI examples
------------
python -m nominal.plant_rollout --nom data/nominal_trajectory/nominal_q1_+4.00_q2_-1.00_N1000.npz
python -m nominal.plant_rollout --cfg configs/robot_arm_2dof.yaml --nom <path> --no-anim
python -m nominal.plant_rollout --nom <path> --no-save-anim  # don't save GIF

Notes
-----
- Saves figures to results/plant_vs_nominal/ by default.
- Saves GIF to results/plant_vs_nominal/anim_<slug>.gif by default.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np

# ensure src/ import
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from models.discretization import make_stepper  # pyright: ignore[reportMissingImports]
from models.robot_arm_2dof import RobotArm2DOF  # pyright: ignore[reportMissingImports]
from utils.config import build_robot_arm_demo_config, load_yaml  # pyright: ignore[reportMissingImports]
from viz.anim_arm import animate_two_link, save_gif, spec_from_model  # pyright: ignore[reportMissingImports]
from viz.plots import plot_inputs, plot_states  # pyright: ignore[reportMissingImports]


def _rollout(step, x0: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Roll discrete stepper with open-loop inputs U. Returns X of shape (N+1, n)."""
    x0 = np.asarray(x0, float).reshape(-1)
    U = np.asarray(U, float)
    if U.ndim != 2:
        raise ValueError("U must be (N, m)")
    N = U.shape[0]
    n = x0.size
    X = np.zeros((N + 1, n), dtype=float)
    X[0] = x0
    for k in range(N):
        X[k + 1] = step(X[k], U[k])
    return X


def _slug_from_goal(meta: Dict[str, Any]) -> str:
    # Try to keep same slug convention as run_nominal.py if present in the .npz
    try:
        xg = np.asarray(meta.get("x_goal"), float).reshape(-1)
        N = int(meta.get("N"))
        return f"q1_{xg[0]:+,.2f}_q2_{xg[1]:+,.2f}_N{N}".replace(",", "")
    except Exception:
        return "plant_eval"


def apply_nominal_to_plant(  # noqa: PLR0915
    nominal_npz: pathlib.Path,
    cfg_path: pathlib.Path = pathlib.Path("configs/robot_arm_2dof.yaml"),
    out_dir: pathlib.Path = pathlib.Path("results/plant_vs_nominal"),
    show: bool = True,
    save_anim: bool = True,
    anim_fps: int = 30,
    no_anim: bool = False,
) -> Dict[str, Any]:
    """
    Core API (can also be called from other scripts).
    Returns dict with Xhat, Uhat, Xplant, dt, and figure/paths.
    """
    # --- Load nominal data ---
    data = np.load(nominal_npz, allow_pickle=True)
    Xhat = np.asarray(data["Xhat"], float)
    Uhat = np.asarray(data["Uhat"], float)
    dt = float(data["dt"])
    meta = {k: data[k] for k in data.files if k not in ("Xhat", "Uhat")}
    slug = _slug_from_goal(meta)

    # --- Build plant from YAML ---
    raw = load_yaml(cfg_path)
    demo, extras = build_robot_arm_demo_config(raw)
    x0_plant = np.asarray(extras.get("x0_plant", np.array([0.3, -0.2, 0.0, 0.0])), float)
    # Use nominal dt for consistent stepping (warn if mismatch)
    dt_cfg = float(demo.discretization.dt)
    if abs(dt_cfg - dt) > 1e-12:
        print(f"[warn] cfg dt={dt_cfg} != nominal dt={dt}; using nominal dt for rollout.")
    plant = RobotArm2DOF(demo.plant, dt=dt, integrator=demo.discretization.method)

    # --- Roll plant with nominal inputs ---
    step_p = make_stepper(plant.f, plant.dt, method=plant._integrator_name)
    Xplant = _rollout(step_p, x0=x0_plant, U=Uhat)

    # --- Sanity: lengths ---
    if Xplant.shape != Xhat.shape:
        T = min(Xplant.shape[0], Xhat.shape[0])
        Xplant = Xplant[:T]
        Xhat = Xhat[:T]
        Uhat = Uhat[: T - 1]

    # --- Prepare names for plots ---
    try:
        state_names = list(demo.X.names)
    except Exception:
        state_names = ["q1 (rad)", "q2 (rad)", "dq1 (rad/s)", "dq2 (rad/s)"]
    try:
        input_names = list(demo.U.names)
    except Exception:
        input_names = ["tau1 (N·m)", "tau2 (N·m)"]

    # --- Time axes ---
    N = Uhat.shape[0]
    t_states = np.arange(N + 1) * dt
    t_inputs = np.arange(N) * dt

    # --- Plots: states overlay ---
    fig_s, _ = plot_states(
        [
            {"label": "Twin nominal", "t": t_states, "X": Xhat, "style": {"linewidth": 1.6}},
            {
                "label": "Plant (applied nominal U)",
                "t": t_states,
                "X": Xplant,
                "style": {"linewidth": 1.4, "linestyle": "--"},
            },
        ],
        names=state_names,
        title="States: nominal (twin) vs plant (open-loop with nominal inputs)",
    )

    # --- Plots: inputs (nominal) ---
    fig_u, _ = plot_inputs(
        [{"label": "Nominal inputs (applied to plant)", "t": t_inputs, "U": Uhat, "style": {"linewidth": 1.6}}],
        names=input_names,
        title="Inputs (nominal Uhat applied to plant)",
    )

    # --- Plots: state deviation (plant - nominal) ---
    Xerr = Xplant - Xhat
    fig_e, _ = plot_states(
        [{"label": "error = plant - nominal", "t": t_states, "X": Xerr, "style": {"linewidth": 1.6}}],
        names=state_names,
        title="State deviation: plant - nominal",
    )

    # --- Save figures ---
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    f_states = out_dir / f"states_vs_nominal_{slug}.png"
    f_inputs = out_dir / f"inputs_applied_{slug}.png"
    f_error = out_dir / f"state_error_{slug}.png"
    fig_s.savefig(f_states, dpi=200)
    fig_u.savefig(f_inputs, dpi=200)
    fig_e.savefig(f_error, dpi=200)
    print(f"\nSaved plots ➜ {f_states}\n             {f_inputs}\n             {f_error}")

    # --- Animation overlay (optional) ---
    gif_path = out_dir / f"anim_plant_vs_nominal_{slug}.gif"
    if not no_anim or save_anim:
        spec_nom = spec_from_model(plant, Xhat, name="Twin nominal", linestyle="-", linewidth=3.0)
        spec_plt = spec_from_model(plant, Xplant, name="Plant", linestyle="--", linewidth=2.5)
        fig_anim, anim = animate_two_link(
            [spec_nom, spec_plt], dt=dt, speed=1.0, trail=200, title="Plant vs. Nominal (inputs = Uhat)"
        )

        if save_anim:
            try:
                save_gif(fig_anim, anim, gif_path, fps=anim_fps, dpi=120)
                print(f"Saved GIF  ➜ {gif_path}")
            except Exception as e:
                print(f"[warn] Failed to save GIF: {e}")

        if not no_anim:
            plt.show()
        else:
            plt.close(fig_anim)

    # Close static figures if no on-screen display requested
    if not show:
        plt.close(fig_s)
        plt.close(fig_u)
        plt.close(fig_e)

    return {
        "Xhat": Xhat,
        "Uhat": Uhat,
        "Xplant": Xplant,
        "dt": dt,
        "fig_states": fig_s,
        "fig_inputs": fig_u,
        "fig_error": fig_e,
        "paths": {"states": f_states, "inputs": f_inputs, "error": f_error, "gif": (gif_path if save_anim else None)},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", type=str, default="configs/robot_arm_2dof.yaml", help="Robot config (plant/twin/...)")
    p.add_argument("--nom", type=str, required=True, help="Path to nominal .npz produced by run_nominal.py")
    p.add_argument("--out", type=str, default="results/plant_vs_nominal", help="Output dir for plots/GIF")
    p.add_argument("--no-anim", action="store_true", help="Do not show animation window")
    p.add_argument("--no-save-anim", action="store_true", help="Do not save GIF (saved by default)")
    p.add_argument("--no-show", action="store_true", help="Do not show static figures")
    p.add_argument("--fps", type=int, default=30, help="GIF frames per second")
    args = p.parse_args()

    apply_nominal_to_plant(
        nominal_npz=pathlib.Path(args.nom),
        cfg_path=pathlib.Path(args.cfg),
        out_dir=pathlib.Path(args.out),
        show=not args.no_show,
        save_anim=not args.no_save_anim,
        anim_fps=int(args.fps),
        no_anim=args.no_anim,
    )


if __name__ == "__main__":
    main()
