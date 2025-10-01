# scripts/run_online_ddfunnel.py
"""
Online Data-Driven Funnel Synthesis.

What it does
------------
- Loads a nominal (Xhat, Uhat, dt, ...) produced by scripts/run_nominal.py
- Loads robot config (plant/twin, constraints, segmentation) and constants YAML
  (gamma, L_J, L_r, C = L_J * v, and Q/R bounds: time-varying)
- Runs an online segmented loop:
    * Every segment i of length T: apply u = Uhat + K_i * (x - Xhat) (+ excitation in last L_i steps)
    * Collect (eta, xi, eta+) over the data window; compute beta_i
    * Build data-consistent and variation set blocks (N1, N2)
    * Build feasibility caps from Q/R bound for that segment
    * Solve SDP for (Q_i, Y_i) and recover K_{i+1} = Y_i Q_i^-1
- Saves logs and (optional) simple preview plots

Usage examples
--------------
# Dry-run (twin as “plant”), preview-only
python scripts/online_ddfunnel.py \
  --cfg configs/robot_arm_2dof.yaml \
  --nom data/nominal_trajectory/nominal_q1_+4.00_q2_-1.00_N1000.npz \
  --constants data/nominal_constants/constants_q1_+4.00_q2_-1.00_N1000.yaml \
  --dry-run --solver SCS --no-save-plots

# With physical plant model (open-loop sim here), save outputs
python scripts/online_ddfunnel.py \
  --cfg configs/robot_arm_2dof.yaml \
  --nom data/nominal_trajectory/nominal_q1_+4.00_q2_-1.00_N1000.npz \
  --constants data/nominal_constants/constants_q1_+4.00_q2_-1.00_N1000.yaml \
  --out results/online_runs --solver SCS
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

# ensure src/ is importable
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))

# ----- project imports -----
from core.bounds import beta_from_segment  # pyright: ignore[reportMissingImports]
from core.data_stack import DataStack  # pyright: ignore[reportMissingImports]
from core.segments import SegmentClock  # pyright: ignore[reportMissingImports]
from dd_sets.system_set import build_system_set_blocks  # pyright: ignore[reportMissingImports]
from dd_sets.variation_set import build_variation_set_blocks  # pyright: ignore[reportMissingImports]
from models.discretization import make_stepper  # pyright: ignore[reportMissingImports]
from models.robot_arm_2dof import RobotArm2DOF  # pyright: ignore[reportMissingImports]
from nominal.trajectory import _dlqr_gain  # pyright: ignore[reportMissingImports]
from synthesis.feasibility_lmis import per_step_feasibility_specs  # pyright: ignore[reportMissingImports]
from synthesis.recover_gain import recover_gain  # pyright: ignore[reportMissingImports]
from synthesis.sdp_problem import solve_funnel_sdp  # pyright: ignore[reportMissingImports]
from utils.config import build_robot_arm_demo_config, load_yaml  # pyright: ignore[reportMissingImports]

# ---------------------------
# Utilities
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


def _load_nominal(npz_path: pathlib.Path) -> Dict[str, Any]:
    Z = np.load(npz_path, allow_pickle=True)
    out = {
        "Xhat": np.asarray(Z["Xhat"], float),
        "Uhat": np.asarray(Z["Uhat"], float),
        "dt": float(Z["dt"]),
        "N": int(Z["N"]) if "N" in Z else (Z["Uhat"].shape[0]),
        "x_goal": np.asarray(Z["x_goal"], float) if "x_goal" in Z else None,
        "u_goal": np.asarray(Z["u_goal"], float) if "u_goal" in Z else None,
        "K0": np.asarray(Z["K"], float) if "K" in Z else None,
        "Q_lqr": np.asarray(Z["Q"], float) if "Q" in Z else None,
        "R_lqr": np.asarray(Z["R"], float) if "R" in Z else None,
        "v": float(Z["v"]) if "v" in Z else None,
    }
    return out


def _load_constants(constants_yaml: pathlib.Path) -> Dict[str, Any]:
    return load_yaml(constants_yaml)


def _q_bounds_from_constants(constants: Dict[str, Any]) -> Dict[str, np.ndarray]:
    # time varying bounds
    if "Q_tv_diag" in constants:
        Q_tv = np.asarray(constants["Q_tv_diag"], float)
        return {"diag": Q_tv}
    elif "Q_max" in constants:
        return {"full": np.asarray(constants["Q_max"], float)}
    else:
        raise KeyError("constants YAML missing Q bounds.")


def _r_bounds_from_constants(constants: Dict[str, Any]) -> Dict[str, np.ndarray]:
    # time varying bounds
    if "R_tv_diag" in constants:
        R_tv = np.asarray(constants["R_tv_diag"], float)
        return {"diag": R_tv}
    elif "R_max" in constants:
        return {"full": np.asarray(constants["R_max"], float)}
    else:
        raise KeyError("constants YAML missing R bounds.")


def _initial_gain(
    twin: RobotArm2DOF, x_goal: np.ndarray, u_goal: np.ndarray, Q_lqr: np.ndarray | None, R_lqr: np.ndarray | None
) -> np.ndarray:
    """
    If nominal saved K, use it. Else DLQR at (x_goal, u_goal) on the twin with
    mild weights as a fallback.
    """
    step = make_stepper(twin.f, twin.dt, method=twin._integrator_name)
    A, B = twin.discrete_jacobians_fd(step, x_goal, u_goal) if hasattr(twin, "discrete_jacobians_fd") else (None, None)

    if A is None or B is None:
        from models.discretization import (  # pyright: ignore[reportMissingImports]  # noqa: PLC0415
            discrete_jacobians_fd,  # pyright: ignore[reportMissingImports]
        )

        A, B = discrete_jacobians_fd(step, x_goal, u_goal, eps=1e-6)

    Qw = Q_lqr if Q_lqr is not None else np.diag([10.0, 10.0, 1.0, 1.0])
    Rw = R_lqr if R_lqr is not None else np.diag([0.1, 0.1])

    K, _ = _dlqr_gain(A, B, Qw, Rw, tol=1e-9, maxit=10000)
    return K


# ---------------------------
# Main
# ---------------------------


def main():  # noqa: PLR0915, C901, PLR0912
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", type=str, default="configs/robot_arm_2dof.yaml", help="Robot config YAML")
    p.add_argument("--nom", type=str, required=True, help="Nominal .npz (from scripts/run_nominal.py)")
    p.add_argument("--constants", type=str, required=True, help="Constants YAML (gamma,LJ,Lr, C, Q/R bounds)")
    p.add_argument("--dry-run", action="store_true", help="Use twin as the plant (simulation dry run)")
    p.add_argument("--alpha", type=float, default=0.92, help="Lyapunov contraction rate (used by SDP layer)")
    p.add_argument("--solver", type=str, default="SCS", help="CVXPY solver (SCS is safe for log_det)")
    p.add_argument("--no-excitation", action="store_true", help="Disable excitation in data windows")
    p.add_argument("--out", type=str, default="results/online_runs", help="Output dir for logs/plots")
    p.add_argument("--no-plots", action="store_true", help="Do not show/save preview plots")
    p.add_argument("--save-plots", action="store_true", help="Save preview plots to <out>/<slug>/")
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--ignore-k0", action="store_true", help="Ignore K stored in nominal .npz and compute plant DLQR for K0"
    )
    # argparse additions
    p.add_argument("--monotone-q", action="store_true",
                help="Enforce per-segment Q_i <= rho * Q_{i-1} (PSD order).")
    p.add_argument("--rho-taper", type=float, default=1.0,
                help="rho in (0,1] for geometric taper in the monotone cap (default 1.0).")
    args = p.parse_args()

    cfg_path = pathlib.Path(args.cfg)
    nom_path = pathlib.Path(args.nom)
    const_path = pathlib.Path(args.constants)

    # --- Load config + models ---
    raw = load_yaml(cfg_path)
    demo, extras = build_robot_arm_demo_config(raw)

    # models (dt from config)
    dt = float(demo.discretization.dt)
    method = demo.discretization.method
    twin = RobotArm2DOF(demo.twin, dt=dt, integrator=method)
    plant_model = twin if args.dry_run else RobotArm2DOF(demo.plant, dt=dt, integrator=method)

    # constraints
    U_low, U_high = demo.U.low, demo.U.high
    X_low, X_high = demo.X.low, demo.X.high  # noqa: F841

    # segmentation
    T = int(demo.segmentation.T)
    L_min = int(demo.segmentation.L_min)
    L_max = int(demo.segmentation.L_max)
    v_bar = float(demo.segmentation.v_bar)
    delta_min = float(demo.segmentation.delta_min)

    # initial states
    x0_plant = np.asarray(extras.get("x0_plant", np.array([0.3, -0.2, 0.0, 0.0])), float)

    # --- Load nominal ---
    nom = _load_nominal(nom_path)
    Xhat = nom["Xhat"]
    Uhat = nom["Uhat"]
    N = int(nom["N"])
    dt_nom = float(nom["dt"])
    if abs(dt_nom - dt) > 1e-12 and args.verbose:
        print(f"[warn] dt mismatch: cfg dt={dt} vs npz dt={dt_nom}. Using cfg dt={dt}.")

    x_goal = nom["x_goal"] if nom["x_goal"] is not None else Xhat[-1]
    u_goal = nom["u_goal"] if nom["u_goal"] is not None else Uhat[-1]
    K0 = None if args.ignore_k0 else nom["K0"]

    # --- Load constants + bounds ---
    constants = _load_constants(const_path)
    gamma = float(constants.get("gamma", 0.0))
    L_J = float(constants.get("L_J", 0.0))
    L_r = float(constants.get("L_r", 0.0))
    C = float(constants.get("C", L_J * float(nom.get("v") or 0.0)))
    T_tilde = int(constants.get("T_tilde", 2 * T - 1))

    Q_bounds = _q_bounds_from_constants(constants)
    R_bounds = _r_bounds_from_constants(constants)

    # --- Steppers ---
    step_p = make_stepper(plant_model.f, plant_model.dt, method=plant_model._integrator_name)

    # --- Initial K ---
    if K0 is None:
        if args.verbose:
            print("[info] no initial K; using DLQR at (x_goal, u_goal) on the plant.")
        K_lqr = _initial_gain(
            plant_model, np.asarray(x_goal, float), np.asarray(u_goal, float), nom["Q_lqr"], nom["R_lqr"]
        )
        K = -K_lqr
    else:
        K = -np.asarray(K0, float)

    # --- Segment clock ---
    clock = SegmentClock(N=N, T=T, L_min=L_min, L_max=L_max, v_bar=v_bar, delta_min=delta_min)  # noqa: F841

    # --- Online loop state ---
    n, m = plant_model.n, plant_model.m
    Xp = np.zeros((N + 1, n), float)
    Up = np.zeros((N, m), float)
    Xp[0] = x0_plant
    seg_idx = 0
    seg_starts: List[int] = []  # noqa: F841
    Ks: List[np.ndarray] = [K.copy()]
    Qs: List[np.ndarray] = []
    Ys: List[np.ndarray] = []

    # data stack
    stack = DataStack(n=n, m=m)

    k_states0 = list(range(0, min(T, N) + 1))
    feas0_for_cap = per_step_feasibility_specs(Q_bounds, R_bounds, k_states0, [])
    Q0_cap = np.array(feas0_for_cap["Qk_list"][0], dtype=float)  # use k=0 cap

    # --- Online time loop ---
    for k in range(N):
        # segmnet index and data window
        i = k // T
        if i != seg_idx:
            seg_idx = i

        k_seg_start = i * T
        k_seg_end = min((i + 1) * T - 1, N - 1)
        L_i = L_max
        in_data = k >= (k_seg_end - L_i + 1)

        # deviation vs nominal
        eta_k = Xp[k] - Xhat[k]
        xi_cmd = K @ eta_k

        # add excitation only in data window
        v_ex = stack.excitation(k, v_bar=v_bar, m=m) if not args.no_excitation and in_data else np.zeros(m)

        u_k = Uhat[k] + xi_cmd + v_ex

        # enforce constraints
        u_k = np.minimum(np.maximum(u_k, U_low), U_high)
        Up[k] = u_k

        # plant step
        Xp[k + 1] = step_p(Xp[k], u_k)
        eta_next = Xp[k + 1] - Xhat[k + 1]

        # collect data if in data window
        if in_data:
            stack.push(eta_k, xi_cmd + v_ex, eta_next)
        if (i == 0 and k < 8) or k == k_seg_end:  # a few samples
            print(
                f"k={k:4d} seg={i:3d} "
                f"||eta||={np.linalg.norm(eta_k):.3e} "
                f"||K_use||={np.linalg.norm(K):.3e} "
                f"||xi||={np.linalg.norm(xi_cmd):.3e} "
                f"||v_ex||={np.linalg.norm(v_ex):.3e} "
                f"in_data={in_data}"
            )

        # segment boundary update (after applying step at k == k_seg_end)
        if k == k_seg_end and (k < N - 1):
            # Export data (if none collected due to tiny L_i, skip)
            if len(stack.etas) >= max(L_min, 1):
                H, H_plus, Xi = stack.export_mats()

                # beta_i aggerates over the time window (idx base is 0,...,L-1); eq uses |k-k_i|, ok up to scaling by dt
                beta_i = beta_from_segment(stack.etas, stack.xis, k_i=0, C=C, gamma=gamma, L_r=L_r, dt=dt, z_clip=5.0)

                # build uncertainty set blocks
                N1_blocks = build_system_set_blocks(H, H_plus, Xi, beta_i)
                N2_blocks = build_variation_set_blocks(C, T_tilde, n, m)

                # feasibility caps from Q/R bound for that segment
                k_states = list(range(k_seg_start, min(k_seg_end + 1, Xhat.shape[0])))
                k_inputs = list(range(k_seg_start, min(k_seg_end + 1, Uhat.shape[0])))
                feas_specs = per_step_feasibility_specs(Q_bounds, R_bounds, k_states, k_inputs)

                Q_caps_extra = []
                if args.monotone_q:
                    if len(Qs) == 0:  # noqa: SIM108
                        # first solved Q is for segment 1 -> cap by Q0
                        Q_prev = Q0_cap
                    else:
                        Q_prev = Qs[-1]
                    rho = float(args.rho_taper)
                    rho = max(1e-6, min(1.0, rho))  # clamp to (0,1]
                    Q_caps_extra = [rho * Q_prev]

                # Solve SDP for (Q_i, Y_i) for feasibility caps only
                res = solve_funnel_sdp(
                    alpha=float(args.alpha),
                    sys_blocks=N1_blocks,
                    var_blocks=N2_blocks,
                    feas_bounds=feas_specs,
                    tau_penalty=1e-3,
                    tau1_cap=1e3,
                    tau2_cap=1e3,
                    Q_caps_extra=Q_caps_extra,
                    y_ref=K,  # <— new
                    y_ref_weight=5e-3,  # tune 1e-4..1e-2
                    solver=args.solver,
                    verbose=args.verbose,
                )
                print(
                    f"seg {i}: status={res['status']}, obj={res['optval']:.3f}, "
                    f"tau1={res['tau1']:.2e}, tau2={res['tau2']:.2e}, "
                    f"cond(Q)={np.linalg.cond(res['Q']):.2e}, ||Y||F={np.linalg.norm(res['Y']):.2e}"
                )

                Q_i, Y_i = res["Q"], res["Y"]
                if res["status"] not in ("optimal", "optimal_inaccurate") or Q_i is None or Y_i is None:
                    print(f"[warn] SDP status={res['status']} for segment {i}; keeping previous K.")
                else:
                    Qs.append(Q_i)
                    Ys.append(Y_i)
                    # K = recover_gain(Y_i, Q_i)
                    K_new = recover_gain(Y_i, Q_i)
                    if np.linalg.norm(K_new) < 0.25 * np.linalg.norm(K):
                        K = K  # keep previous (too tiny)  # noqa: PLW0127
                    else:
                        # blend to avoid abrupt changes
                        lam = 0.4
                        K = (1 - lam) * K + lam * K_new
                    Ks.append(K.copy())

            # reset data
            stack.reset_for_next_segment()

    slug = _slug_from_npz(nom_path, Xhat, Uhat)
    out_dir = pathlib.Path(args.out) / slug
    out_dir.mkdir(parents=True, exist_ok=True)

        # --- Build per-step Q sequence for plotting: segment 0 = caps, later = solved Q_i ---
    n = plant_model.n
    n_segs = int(np.ceil(N / T))
    Q_seq_states = np.zeros((N + 1, n, n))

    # Segment 0: use the *per-step* caps from constants
    k_states0 = list(range(0, min(T, N) + 1))
    feas0 = per_step_feasibility_specs(Q_bounds, R_bounds, k_states0, [])
    Qk_list0 = feas0["Qk_list"]
    for idx, k in enumerate(k_states0):
        Q_seq_states[k] = Qk_list0[idx]

    # Segments i >= 1: piecewise-constant, use solved Q_i (note: Qs[i-1] is for seg i)
    for i in range(1, n_segs):
        Qi = Qs[i - 1] if (i - 1) < len(Qs) else Qs[-1]
        k0 = i * T
        k1 = min((i + 1) * T, N)
        Q_seq_states[k0 : k1 + 1] = Qi

    # Terminal: ensure last state uses last segment value
    if not np.any(Q_seq_states[N]):
        Q_seq_states[N] = Q_seq_states[N - 1]

    np.savez_compressed(
        out_dir / f"online_run_{slug}.npz",
        Xhat=Xhat,
        Uhat=Uhat,
        Xplant=Xp,
        Uapplied=Up,
        Ks=np.array(Ks, dtype=float),
        Qs=np.array(Qs, dtype=float),
        Ys=np.array(Ys, dtype=float),
        Q_seq_states=Q_seq_states,  # <-- add this line
        dt=dt,
        T=T,
        L_max=L_max,
        alpha=float(args.alpha),
        constants=constants,
    )
    print(f"\nSaved run ➜ {out_dir / ('online_run_' + slug + '.npz')}")

    # --- Plots (optional) ---
    if not args.no_plots or args.save_plots:
        t_states = np.arange(Xp.shape[0]) * dt
        t_inputs = np.arange(Up.shape[0]) * dt

        fig1, axs1 = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
        names_x = ["q1", "q2", "dq1", "dq2"]
        for i in range(4):
            axs1[i].plot(t_states, Xhat[:, i], label="nominal", lw=1.6)
            axs1[i].plot(t_states, Xp[:, i], "--", label="plant", lw=1.4)
            axs1[i].set_ylabel(names_x[i])
            axs1[i].grid(alpha=0.3)
        axs1[-1].set_xlabel("time (s)")
        axs1[0].legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.15))
        fig1.tight_layout()
        if args.save_plots:
            fig1.savefig(out_dir / "states_online.png", dpi=200)

        fig2, axs2 = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        names_u = ["tau1", "tau2"]
        for j in range(2):
            axs2[j].plot(t_inputs, Uhat[:, j], label="nominal U", lw=1.6)
            axs2[j].plot(t_inputs, Up[:, j], "--", label="applied U", lw=1.4)
            axs2[j].set_ylabel(names_u[j])
            axs2[j].grid(alpha=0.3)
        axs2[-1].set_xlabel("time (s)")
        axs2[0].legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.15))
        fig2.tight_layout()
        if args.save_plots:
            fig2.savefig(out_dir / "inputs_online.png", dpi=200)

        if not args.save_plots:
            plt.show()
        else:
            plt.close(fig1)
            plt.close(fig2)


if __name__ == "__main__":
    main()
