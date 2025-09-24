# src/robust_ddfunnel/models/parameters.py
"""
Parameter and configuration dataclasses for the 2-DoF planar robot arm demo.

This module contains ONLY data containers and light validation helpers.
No algorithmic logic is implemented here (stays faithful to the paper).

Recommended usage:
- Define plant and twin ArmParams here (the twin can use slight mismatches).
- Keep discretization, constraints, funnel, segmentation, and solver configs
  as explicit, typed containers loaded from YAML in utils/config.py.
- In robot_arm_2dof.py, import ArmParams from here.

Paper notation mapping (selection):
  - Dimensions: n=4, m=2 for the 2-DoF arm
  - Discretization: dt, method∈{"rk4","euler"}
  - Constraints: X (state), U (input)
  - Funnel/regularity constants: alpha, L_J, L_r, gamma
  - Segmentation: T, L_min, L_max, v̄, δ_min
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Tuple

import numpy as np

Array = np.ndarray
IntegratorName = Literal["rk4", "euler"]

# ---------------------------
# Robot arm (plant/twin) params
# ---------------------------


@dataclass(frozen=True)
class ArmParams:
    """
    Physical parameters for a planar 2-link robot arm (SI units).
    If lc1/lc2 are None, they default to half-link lengths.

    Fields:
    - m1, m2: link masses (kg)
    - l1, l2: link lengths (m)
    - I1, I2: link moments of inertia (kg·m²)
    - b1, b2: link viscous friction coefficients (N·m·s/rad)
    - g: gravitational acceleration (m/s²)
    - lc1, lc2: link center-of-mass distances from joints (m)
    """

    m1: float
    m2: float
    l1: float
    l2: float
    I1: float
    I2: float
    b1: float = 0.0
    b2: float = 0.0
    g: float = 9.81
    lc1: float | None = None
    lc2: float | None = None

    def with_defaults(self) -> ArmParams:
        lc1 = self.lc1 if self.lc1 is not None else 0.5 * self.l1
        lc2 = self.lc2 if self.lc2 is not None else 0.5 * self.l2
        return ArmParams(
            m1=self.m1,
            m2=self.m2,
            l1=self.l1,
            l2=self.l2,
            I1=self.I1,
            I2=self.I2,
            b1=self.b1,
            b2=self.b2,
            g=self.g,
            lc1=lc1,
            lc2=lc2,
        )


def armparams_from_dict(d: Mapping[str, Any]) -> ArmParams:
    """
    Constructs ArmParams from a flat dict. (loaded from YAML)

    Expected keys:
        m1, m2, l1, l2, I1, I2, b1, b2, g, lc1, lc2
    """
    return ArmParams(
        m1=float(d["m1"]),
        m2=float(d["m2"]),
        l1=float(d["l1"]),
        l2=float(d["l2"]),
        I1=float(d["I1"]),
        I2=float(d["I2"]),
        b1=float(d.get("b1", 0.0)),
        b2=float(d.get("b2", 0.0)),
        g=float(d.get("g", 9.81)),
        lc1=float(d["lc1"]) if "lc1" in d and d["lc1"] is not None else None,
        lc2=float(d["lc2"]) if "lc2" in d and d["lc2"] is not None else None,
    ).with_defaults()


def make_twin_from_plant(
    plant: ArmParams,
    deltas: Mapping[str, float],
) -> ArmParams:
    """
    Convenience: build  slightly mismatched twin from a plant spec.

    deltas: dict of relative changes (e.g., {"m1": -0.05, "l1": +0.03})
            meaning twin.param = plant.param * (1 + delta[param])
    """
    fields = {k: getattr(plant, k) for k in plant.__dataclass_fields__}  # type: ignore[attr-defined]
    for k, dv in deltas.items():
        if k in fields and isinstance(fields[k], (int, float)):
            fields[k] = float(fields[k]) * (1.0 + float(dv))
    return ArmParams(**fields).with_defaults()


# ---------------------------
# Discretization config
# ---------------------------


@dataclass(frozen=True)
class DiscretizationConfig:
    """
    Discretization configuration for the 2-DoF planar robot arm demo.
    """

    dt: float
    method: IntegratorName = "rk4"

    def validate(self) -> None:
        if not (self.dt > 0.0):
            raise ValueError("dt must be positive")
        if self.method not in ("rk4", "euler"):
            raise ValueError("method must be 'rk4' or 'euler'")


# ---------------------------
# Constraint boxes X, U
# ---------------------------
@dataclass(frozen=True)
class ConstraintBox:
    """
    Generic box constraints: low <= z <= high (elementwise)
    """

    low: Array
    high: Array

    def validate(self) -> None:
        if self.low.shape != self.high.shape:
            raise ValueError("low and high must have the same shape")
        if not np.all(self.low <= self.high):
            raise ValueError("low must be less than or equal to high")

    @property
    def dim(self) -> int:
        return int(self.low.size)

    @staticmethod
    def from_named_limits(
        mapping: Mapping[str, Tuple[float, float]],
        order: list[str],
    ) -> "ConstraintBox":
        """
        Build a box from a dict of name -> (low, high) on an ordering list.
        Example for state:
            order = ["q1", "q2", "dq1", "dq2"]
            mapping = {"q1": [-1.2, 1.2], "q2": [-1.2, 1.2], "dq1": [-3, 3], "dq2": [-3, 3]}
        """
        lows, highs = [], []
        for name in order:
            lo, hi = mapping[name]
            lows.append(float(lo))
            highs.append(float(hi))
        return ConstraintBox(low=np.array(lows, dtype=float), high=np.array(highs, dtype=float))


# ---------------------------
# Funnel/regularity constants
# ---------------------------
@dataclass(frozen=True)
class FunnelConfig:
    """
    Paper constants:
    - alpha: safety factor for funnel
    - L_J: Lipschitz constant for Jacobian
    - L_r: Lipschitz constant for residuals
    - gamma: safety factor for residuals
    """

    alpha: float
    L_J: float
    L_r: float
    gamma: float

    def validate(self) -> None:
        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must be between 0 and 1")
        if not (self.L_J > 0.0 and self.L_r > 0.0 and self.gamma >= 0.0):
            raise ValueError("L_J, L_r, and gamma must be positive")


# ---------------------------
# Segmentation config
# ---------------------------
@dataclass(frozen=True)
class SegmentationConfig:
    """
    Segment length T, adaptive window limits L_min, L_max,
    excitation bound v̄, and baseline deviation δ
    """

    T: int
    L_min: int
    L_max: int
    v_bar: float
    delta_min: float

    def validate(self) -> None:
        if not (self.T > 0 and self.L_min > 0 and self.L_max >= self.L_min):
            raise ValueError("Require T>0, L_min>0, L_max>=L_min")
        if not (self.v_bar >= 0.0 and self.delta_min > 0.0):
            raise ValueError("Require v_bar>=0 and delta_min>0")


# ---------------------------
# Solver config (for CVXPY)
# ---------------------------


@dataclass(frozen=True)
class SolverConfig:
    """
    CVXPY solver configuration.
    'name' must be a CVXPY-recognized solver. Options are free form.

    Example:
        name="scs", options={"max_iters": 20000, "eps": 1e-6}
    """

    name: str = "scs"
    options: Mapping[str, Any] | None = None


# ---------------------------
# Full config
# ---------------------------
@dataclass(frozen=True)
class RobotArmDemoConfig:
    """
    Optional aggregator for convenience when loading from YAML.
    This mirrors the example YAML structure show earlier.
    """

    plant: ArmParams
    twin: ArmParams
    discretization: DiscretizationConfig
    X: ConstraintBox
    U: ConstraintBox
    funnel: FunnelConfig
    segmentation: SegmentationConfig
    solver: SolverConfig

    def validate(self) -> None:
        self.discretization.validate()
        self.X.validate()
        self.U.validate()
        self.funnel.validate()
        self.segmentation.validate()


__all__ = [
    "ArmParams",
    "ConstraintBox",
    "DiscretizationConfig",
    "FunnelConfig",
    "RobotArmDemoConfig",
    "SegmentationConfig",
    "SolverConfig",
    "armparams_from_dict",
    "make_twin_from_plant",
]
