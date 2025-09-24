# scripts/compute_bounds.py
"""
Compute data-driven constants:
  - gamma : sup_k || step_plant(x*_k,u*_k) - step_twin(x*_k,u*_k) ||
  - L_J   : Lipschitz constant for the Jacobian of the discrete map over XxU
  - L_r   : quadratic linearization error coefficient for the discrete map

Reads:
  --cfg  : robot config (to build plant/twin and boxes)
  --nom  : nominal .npz produced by scripts/run_nominal.py  (Xhat, Uhat, v, dt, ...)

Writes:
  data/nominal_constants/constants_<slug>.yaml
  where <slug> comes from the nominal filename nominal_<slug>.npz
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Any, Dict, Tuple

import numpy as np
import yaml

# make src/ importable
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from core.bounds import (  # pyright: ignore[reportMissingImports]
    estimate_LJ_discrete,
    estimate_Lr_discrete,
    gamma_pointwise_on_nominal,
)
from models.discretization import make_stepper  # pyright: ignore[reportMissingImports]
from models.robot_arm_2dof import RobotArm2DOF  # pyright: ignore[reportMissingImports]
from utils.config import build_robot_arm_demo_config, load_yaml  # pyright: ignore[reportMissingImports]

# ---------------------------
# helpers
# ---------------------------


def _infer_slug_from_npz(npz_path: pathlib.Path) -> str:
    stem = npz_path.stem
    return stem[len("nominal_") :] if stem.startswith("nominal_") else stem


def _load_nominal_npz(path: pathlib.Path) -> Dict[str, Any]:
    z = np.load(path, allow_pickle=True)
    out = {
        "Xhat": np.asarray(z["Xhat"], float),
        "Uhat": np.asarray(z["Uhat"], float),
        "dt": float(z["dt"]),
        "N": int(z["N"]) if "N" in z else (int(z["Xhat"].shape[0]) - 1),
        "v": float(z["v"]) if "v" in z else None,
        "x_goal": np.asarray(z["x_goal"], float) if "x_goal" in z else None,
        "u_goal": np.asarray(z["u_goal"], float) if "u_goal" in z else None,
    }
    return out


def _get_state_box(demo, Xhat: np.ndarray, pad_frac: float) -> Tuple[np.ndarray, np.ndarray]:
    try:
        low = np.asarray(demo.X.low, float).reshape(-1)
        high = np.asarray(demo.X.high, float).reshape(-1)
        if low.shape == high.shape == (Xhat.shape[1],):
            return low, high
    except Exception:
        pass
    # infer from nominal, then pad by a fraction of the span
    x_min = Xhat.min(axis=0)
    x_max = Xhat.max(axis=0)
    span = np.maximum(1e-6, x_max - x_min)
    return x_min - pad_frac * span, x_max + pad_frac * span


def _get_input_box(demo) -> Tuple[np.ndarray, np.ndarray]:
    return np.asarray(demo.U.low, float), np.asarray(demo.U.high, float)


# ---------------------------
# main
# ---------------------------


def main():  # noqa: PLR0915
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", type=str, default="configs/robot_arm_2dof.yaml", help="Robot config YAML")
    p.add_argument("--nom", type=str, required=True, help="Nominal NPZ file from scripts/run_nominal.py")
    p.add_argument(
        "--out-constants",
        type=str,
        default=None,
        help="Target YAML path; default: data/nominal_constants/constants_<slug>.yaml",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--debug", action="store_true")

    # gamma options
    p.add_argument("--gamma-rel", type=float, default=0.05, help="Relative safety margin on gamma")
    p.add_argument("--gamma-abs", type=float, default=1e-9, help="Absolute safety margin on gamma")
    p.add_argument("--gamma-norm", type=str, choices=["l2", "linf"], default="l2")

    # boxes
    p.add_argument("--box-pad", type=float, default=0.10, help="Pad fraction when inferring X-box from nominal")

    # L_J options (match bounds.py signature)
    p.add_argument("--lj-samples", type=int, default=400)
    p.add_argument("--lj-hrel", type=float, default=1e-4, help="Relative step size for the Hessian FD in L_J estimator")
    p.add_argument("--lj-fd-eps", type=float, default=1e-6, help="Finite-diff epsilon for inner Jacobian calls")
    p.add_argument("--lj-norm", type=str, choices=["2", "fro"], default="2")
    p.add_argument(
        "--lj-random-dirs", action="store_true", help="Use random unit directions instead of coordinate axes"
    )
    p.add_argument("--lj-dirs-per-point", type=int, default=6)

    # L_r options (match bounds.py signature)
    p.add_argument("--lr-hrel", type=float, default=1e-4, help="Relative step size for Hessian part of L_r")
    p.add_argument("--lr-dirs-hess", type=int, default=4, help="# directions per nominal point for Hessian part")
    p.add_argument(
        "--lr-delta-rel", type=float, default=1e-2, help="Relative perturbation magnitude for remainder-ratio part"
    )
    p.add_argument("--lr-dirs-rem", type=int, default=8, help="# directions per nominal point for remainder-ratio part")
    p.add_argument("--lr-fd-eps", type=float, default=1e-6, help="Finite-diff epsilon for Jacobians inside L_r")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    # --- Load config ---
    cfg_path = pathlib.Path(args.cfg)
    raw = load_yaml(cfg_path)
    demo, extras = build_robot_arm_demo_config(raw)
    dt = float(demo.discretization.dt)
    method = demo.discretization.method

    # --- Load nominal ---
    nom_path = pathlib.Path(args.nom)
    nom = _load_nominal_npz(nom_path)
    Xhat = nom["Xhat"]
    Uhat = nom["Uhat"]
    N_nom = int(nom["N"])
    v_nom = nom["v"]
    x_goal = nom["x_goal"]
    u_goal = nom["u_goal"]
    if abs(nom["dt"] - dt) > 1e-12 and args.debug:
        print(f"[warn] dt mismatch: cfg dt={dt} vs npz dt={nom['dt']}. Using cfg dt={dt}.")

    # --- Build models + steppers ---
    plant = RobotArm2DOF(demo.plant, dt=dt, integrator=method)
    twin = RobotArm2DOF(demo.twin, dt=dt, integrator=method)
    plant_step = make_stepper(plant.f, dt=dt, method=method)
    twin_step = make_stepper(twin.f, dt=dt, method=method)

    # --- Boxes ---
    X_low, X_high = _get_state_box(demo, Xhat, pad_frac=args.box_pad)
    U_low, U_high = _get_input_box(demo)

    # --- gamma along nominal (pointwise, no rollout) ---
    gamma = gamma_pointwise_on_nominal(
        step_plant=plant_step,
        step_twin=twin_step,
        X_nom=Xhat,
        U_nom=Uhat,
        norm=args.gamma_norm,
        abs_margin=args.gamma_abs,
        rel_margin=args.gamma_rel,
        return_per_step=False,
    )

    # --- L_J over XxU ---
    L_J = estimate_LJ_discrete(
        step=plant_step,
        X_box=(X_low, X_high),
        U_box=(U_low, U_high),
        samples=int(args.lj_samples),
        h_rel=float(args.lj_hrel),
        fd_eps=float(args.lj_fd_eps),
        rng=rng,
        norm=args.lj_norm,
        abs_margin=0.0,
        rel_margin=0.0,
        use_random_dirs=bool(args.lj_random_dirs),
        dirs_per_point=int(args.lj_dirs_per_point),
    )

    # --- L_r along nominal (Hessian + remainder ratio) ---
    L_r = estimate_Lr_discrete(
        step=plant_step, X_nom=Xhat, U_nom=Uhat,
        X_box=(X_low, X_high), U_box=(U_low, U_high),
        use_hessian_bound=False,
        use_remainder_ratio=True,
        local_only=True, eps_x=1e-4, eps_u=5e-3,
        fd_eps=1e-6, norm="2"
    )

    # --- Compose and save YAML ---
    slug = _infer_slug_from_npz(nom_path)
    out_path = (
        pathlib.Path(args.out_constants)
        if args.out_constants
        else (pathlib.Path("data/nominal_constants") / f"constants_{slug}.yaml")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # merge if exists
    existing: Dict[str, Any] = {}
    if out_path.exists():
        try:
            with open(out_path, "r") as f:
                existing = yaml.safe_load(f) or {}
        except Exception:
            existing = {}

    # carry through prior T_tilde if any
    T_tilde = existing.get("T_tilde")

    C_val = (float(L_J) * float(v_nom)) if (v_nom is not None) else None

    payload: Dict[str, Any] = {
        "dt": float(dt),
        "N": int(N_nom),
        "v": (float(v_nom) if v_nom is not None else None),
        "gamma": float(gamma),
        "L_J": float(L_J),
        "L_r": float(L_r),
        "C": (float(C_val) if C_val is not None else None),
        "T_tilde": T_tilde,
        "x_goal": (x_goal.tolist() if x_goal is not None else None),
        "u_goal": (u_goal.tolist() if u_goal is not None else None),
        "X_box": {"low": X_low.tolist(), "high": X_high.tolist()},
        "U_box": {"low": U_low.tolist(), "high": U_high.tolist()},
    }

    with open(out_path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)

    # --- Report ---
    np.set_printoptions(precision=6, suppress=True)
    print("\n=== Computed constants ===")
    print(f"gamma : {gamma:.6f}")
    print(f"L_J   : {L_J:.6f}")
    print(f"L_r   : {L_r:.6f}")
    if v_nom is not None:
        print(f"v     : {v_nom:.6f}")
        print(f"C=L_J*v : {(L_J * float(v_nom)):.6f}")
    print(f"\nSaved constants ➜ {out_path}\n")

    if args.debug:
        print("[debug] X_box low/high:\n ", X_low, "\n ", X_high)
        print("[debug] U_box low/high:\n ", U_low, "\n ", U_high)


if __name__ == "__main__":
    main()
