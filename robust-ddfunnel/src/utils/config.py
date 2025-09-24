# src/robust_ddfunnel/utils/config.py
"""
Config loader for the robot-arm demo.

- Loads a YAML file.
- Builds typed dataclasses from robust_ddfunnel.models.parameters.
- Validates shapes/ranges.
- Returns (RobotArmDemoConfig, extras) where extras may include:
    - N (horizon length) if provided under "horizon: { N: ... }"
    - seed if provided at top level

Expected YAML keys (example):

plant:            { m1: 1.0, m2: 0.8, l1: 0.7, l2: 0.6, I1: 0.05, I2: 0.04, b1: 0.02, b2: 0.02, g: 9.81 }
twin:             { m1: 0.95, m2: 0.84, l1: 0.73, l2: 0.58, I1: 0.055, I2: 0.038, b1: 0.018, b2: 0.022, g: 9.81 }
discretization:   { dt: 0.01, method: rk4 }
horizon:          { N: 1500 }
constraints:
  X: { q1: [-1.2, 1.2], q2: [-1.2, 1.2], dq1: [-3.0, 3.0], dq2: [-3.0, 3.0] }
  U: { tau1: [-6.0, 6.0], tau2: [-6.0, 6.0] }
funnel:           { alpha: 0.92, L_J: 3.0, L_r: 1.0, gamma: 0.15 }
segmentation:     { T: 200, L_min: 6, L_max: 60, v_bar: 0.6, delta_min: 0.05 }
solver:           { name: scs, options: { max_iters: 20000, eps: 1e-5 } }
seed:             42
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import yaml

from models.parameters import (  # pyright: ignore[reportMissingImports]
    ArmParams,
    ConstraintBox,
    DiscretizationConfig,
    FunnelConfig,
    RobotArmDemoConfig,
    SegmentationConfig,
    SolverConfig,
    armparams_from_dict,
)

STATE_ORDER = ["q1", "q2", "dq1", "dq2"]
INPUT_ORDER = ["tau1", "tau2"]


def _as_state(vec: Any, expected_dim: int = 4) -> np.ndarray:
    arr = np.asarray(vec, dtype=float).reshape(-1)
    if arr.size != expected_dim:
        raise ValueError(f"initial state must have length {expected_dim}, got {arr.size}")
    return arr


def _as_vec(x, dim):
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size != dim:
        raise ValueError(f"expected vector of length {dim}, got {arr.size}")
    return arr


def _as_matrix(diag_or_full, dim):
    arr = np.asarray(diag_or_full, dtype=float)
    if arr.ndim == 0:  # scalar
        return np.diag(np.full(dim, float(arr)))
    if arr.ndim == 1:  # diag entries
        if arr.size != dim:
            raise ValueError(f"diag length {arr.size} != {dim}")
        return np.diag(arr)
    if arr.shape == (dim, dim):
        return arr
    raise ValueError(f"matrix must be ({dim},{dim}) or diag length {dim}")


def _constraint_box_from_named(mapping: Dict[str, Any], order: list[str]) -> ConstraintBox:
    """
    Build a ConstraintBox from a dict of name -> (low, high) on an ordering list.
    """
    lows, highs = [], []
    for name in order:
        if name not in mapping:
            raise ValueError(f"Missing constraint for {name} in constraint mapping")
        low, hi = mapping[name]
        lows.append(float(low))
        highs.append(float(hi))
    box = ConstraintBox(low=np.array(lows, dtype=float), high=np.array(highs, dtype=float))
    box.validate()
    return box


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """
    Load a YAML file into a Python dict.
    """
    p = Path(path)
    with p.open("r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Top-level YAML must be a mapping")
    return data


def save_yaml(data: Dict[str, Any], path: str | Path) -> None:
    """
    Save a Python dict to a YAML file (creates parent dirs).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def build_robot_arm_demo_config(cfg: Dict[str, Any]) -> Tuple[RobotArmDemoConfig, Dict[str, Any]]:  # noqa: C901, PLR0912, PLR0915
    """
    Construct a RobotArmDemoConfig from a YAML dict.

    Returns:
    -------
    (demo_config, extras) where extras may include:
        - "N": horizon length (int) if provided under cfg["horizon"]["N"]
        - "seed": seed (int) if provided at top level
    """
    # --- plant/twin ---
    if "plant" not in cfg:
        raise ValueError("Missing 'plant' section in YAML")
    plant: ArmParams = armparams_from_dict(cfg["plant"])

    if "twin" in cfg and cfg["twin"] is not None:
        twin: ArmParams = armparams_from_dict(cfg["twin"])
    else:
        twin = plant

    # --- discretization ---
    if "discretization" not in cfg:
        raise KeyError("Config missing 'discretization' section.")
    disc_raw = cfg["discretization"]
    disc = DiscretizationConfig(
        dt=float(disc_raw["dt"]),
        method=str(disc_raw.get("method", "rk4")),
    )
    disc.validate()

    # --- constraints ---
    if "constraints" not in cfg:
        raise KeyError("Config missing 'constraints' section.")
    cons_raw = cfg["constraints"]
    if "X" not in cons_raw or "U" not in cons_raw:
        raise KeyError("Config missing 'X' or 'U' in 'constraints' section.")
    X = _constraint_box_from_named(cons_raw["X"], STATE_ORDER)
    U = _constraint_box_from_named(cons_raw["U"], INPUT_ORDER)

    # --- funnel ---
    if "funnel" not in cfg:
        raise KeyError("Config missing 'funnel' section.")
    fun_raw = cfg["funnel"]
    funnel = FunnelConfig(
        alpha=float(fun_raw["alpha"]),
        L_J=float(fun_raw["L_J"]),
        L_r=float(fun_raw["L_r"]),
        gamma=float(fun_raw["gamma"]),
    )
    funnel.validate()

    # --- segmentation ---
    if "segmentation" not in cfg:
        raise KeyError("Config missing 'segmentation' section.")
    seg_raw = cfg["segmentation"]
    segmentation = SegmentationConfig(
        T=int(seg_raw["T"]),
        L_min=int(seg_raw["L_min"]),
        L_max=int(seg_raw["L_max"]),
        v_bar=float(seg_raw["v_bar"]),
        delta_min=float(seg_raw["delta_min"]),
    )
    segmentation.validate()

    # --- solver ---
    if "solver" not in cfg:
        raise KeyError("Config missing 'solver' section.")
    sol_raw = cfg.get("solver", {"name": "scs"})
    solver = SolverConfig(
        name=str(sol_raw.get("name", "scs")),
        options=sol_raw.get("options", None),
    )

    demo = RobotArmDemoConfig(
        plant=plant,
        twin=twin,
        discretization=disc,
        X=X,
        U=U,
        funnel=funnel,
        segmentation=segmentation,
        solver=solver,
    )
    demo.validate()

    # --- extras ---
    extras: Dict[str, Any] = {}
    if "horizon" in cfg and isinstance(cfg["horizon"], dict) and "N" in cfg["horizon"]:
        extras["N"] = int(cfg["horizon"]["N"])
    if "seed" in cfg:
        extras["seed"] = int(cfg["seed"])

    # --- initial ---
    if "initial" in cfg and isinstance(cfg["initial"], dict):
        ini = cfg["initial"]
        if "plant_x0" in ini:
            extras["x0_plant"] = _as_state(ini["plant_x0"], expected_dim=4)
        if "twin_x0" in ini:
            extras["x0_twin"] = _as_state(ini["twin_x0"], expected_dim=4)
    return demo, extras

    # --- nominal ---
    if "nominal" in cfg and isinstance(cfg["nominal"], dict):
        nom = cfg["nominal"]
        # Defaults
        x_goal = _as_vec(nom.get("x_goal", [0.0, 0.0, 0.0, 0.0]), 4)
        u_goal = _as_vec(nom.get("u_goal", [0.0, 0.0]), 2)
        Q = _as_matrix(nom.get("Q", [10.0, 10.0, 1.0, 1.0]), 4)
        R = _as_matrix(nom.get("R", [0.1, 0.1]), 2)
        Qf = _as_matrix(nom.get("Qf", [50.0, 50.0, 5.0, 5.0]), 4)
        N_nom = int(nom.get("N", extras.get("N", 800)))
        # Optional limits: default to U box from this config
        u_low = _as_vec(nom.get("u_low", U.low.tolist()), 2)
        u_high = _as_vec(nom.get("u_high", U.high.tolist()), 2)
        extras["nominal"] = {
            "x_goal": x_goal,
            "u_goal": u_goal,
            "Q": Q,
            "R": R,
            "Qf": Qf,
            "N": N_nom,
            "u_low": u_low,
            "u_high": u_high,
            "riccati_tol": float(nom.get("riccati_tol", 1e-9)),
            "riccati_maxit": int(nom.get("riccati_maxit", 10000)),
            "fd_eps": float(nom.get("fd_eps", 1e-6)),
        }


def save_config_example(path: str | Path) -> None:
    """
    Write a minimal example YAML to 'path'
    """
    example = {
        "plant": {
            "m1": 1.0,
            "m2": 0.8,
            "l1": 0.7,
            "l2": 0.6,
            "I1": 0.05,
            "I2": 0.04,
            "b1": 0.02,
            "b2": 0.02,
            "g": 9.81,
        },
        "twin": {
            "m1": 0.95,
            "m2": 0.84,
            "l1": 0.73,
            "l2": 0.58,
            "I1": 0.055,
            "I2": 0.038,
            "b1": 0.018,
            "b2": 0.022,
            "g": 9.81,
        },
        "discretization": {"dt": 0.01, "method": "rk4"},
        "horizon": {"N": 1500},
        "constraints": {
            "X": {"q1": [-1.2, 1.2], "q2": [-1.2, 1.2], "dq1": [-3.0, 3.0], "dq2": [-3.0, 3.0]},
            "U": {"tau1": [-6.0, 6.0], "tau2": [-6.0, 6.0]},
        },
        "funnel": {"alpha": 0.92, "L_J": 3.0, "L_r": 1.0, "gamma": 0.15},
        "segmentation": {"T": 200, "L_min": 6, "L_max": 60, "v_bar": 0.6, "delta_min": 0.05},
        "solver": {"name": "scs", "options": {"max_iters": 20000, "eps": 1e-5}},
        "seed": 42,
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        yaml.safe_dump(example, f, sort_keys=False)


__all__ = [
    "INPUT_ORDER",
    "STATE_ORDER",
    "build_robot_arm_demo_config",
    "load_yaml",
    "save_config_example",
    "save_yaml",
]
