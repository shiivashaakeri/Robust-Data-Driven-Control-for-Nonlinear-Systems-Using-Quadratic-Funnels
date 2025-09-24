# src/robust_ddfunnel/models/robot_arm_2dof.py
"""
2-DoF planar robot arm (Spong form), integrated via models/discretization.py.

State: x = [q1, q2, dq1, dq2]^T      (n = 4)
Input: u = [tau1, tau2]^T            (m = 2)

Continuous dynamics:
    qd   = dq
    ddq  = M(q)^{-1} [ tau - C(q, dq) dq - G(q) - B dq ]

Discretization (sample-and-hold over [t, t+dt]):
    Provided by discretization.make_stepper(f, dt, method="rk4"/"euler")

Notes:
- Instantiate two copies (plant & twin) with slightly mismatched parameters
  to realize the imperfect digital twin assumption from the paper.
- This module implements only the dynamics and a discrete step wrapper.
  Controller logic, data stacks (H_i, H_i^+, Xi_i), LMIs, etc. live elsewhere.
"""

from __future__ import annotations

from typing import Literal, Tuple

import numpy as np

from models.discretization import discrete_jacobians_fd, make_stepper
from models.parameters import ArmParams

Array = np.ndarray
IntegratorName = Literal["rk4", "euler"]


class RobotArm2DOF:
    """
    2-DoF planar arm wrapper using shared discretization utilities.

    Example
    -------
    >>> plant_params = ArmParams(m1=1.0, m2=0.8, l1=0.7, l2=0.6, I1=0.05, I2=0.04,
    ...                          b1=0.02, b2=0.02).with_defaults()
    >>> twin_params  = ArmParams(m1=0.95, m2=0.84, l1=0.73, l2=0.58, I1=0.055, I2=0.038,
    ...                          b1=0.018, b2=0.022).with_defaults()
    >>> plant = RobotArm2DOF(plant_params, dt=0.01, integrator="rk4")
    >>> twin  = RobotArm2DOF(twin_params,  dt=0.01, integrator="rk4")
    >>> x = np.zeros(4); u = np.zeros(2)
    >>> x_next = plant.step(x, u)
    """

    def __init__(
        self,
        params: ArmParams,
        dt: float,
        integrator: IntegratorName = "rk4",
    ) -> None:
        self.p = params.with_defaults()
        self.dt = float(dt)
        if integrator not in ("rk4", "euler"):
            raise ValueError("integrator must be 'rk4' or 'euler'")
        self._integrator_name: IntegratorName = integrator

        # Build a discrete-time stepper from the continuous dynamics f(x,u)
        self._step = make_stepper(self.f, self.dt, method=self._integrator_name)

    # ---------------------------
    # Public API
    # ---------------------------

    @property
    def n(self) -> int:
        return 4

    @property
    def m(self) -> int:
        return 2

    def step(self, x: Array, u: Array) -> Array:
        """
        Advance one discrete step with the chosen integrator.

        Parameters
        ----------
        x : (4,) array_like
        u : (2,) array_like

        Returns
        -------
        x_next : (4,) ndarray
        """
        x = np.asarray(x, dtype=float).reshape(self.n)
        u = np.asarray(u, dtype=float).reshape(self.m)
        return self._step(x, u)

    def set_integrator(self, method: IntegratorName) -> None:
        """
        Switch integrator at runtime ('rk4' or 'euler'), preserving dt.
        """
        if method not in ("rk4", "euler"):
            raise ValueError("method must be 'rk4' or 'euler'")
        self._integrator_name = method
        self._step = make_stepper(self.f, self.dt, method=method)

    # ---------------------------
    # Continuous dynamics f(x, u)
    # ---------------------------

    def f(self, x: Array, u: Array) -> Array:
        """
        Continuous-time state derivative xdot = f(x, u).
        """
        q, dq = x[:2], x[2:]
        tau = u

        M = self._mass_matrix(q)
        C = self._coriolis_matrix(q, dq)
        G = self._gravity(q)
        B = self._viscous()

        # qdd = M^{-1} (tau - C*dq - G - B*dq)
        rhs = tau - C @ dq - G - B @ dq
        ddq = np.linalg.solve(M, rhs)

        return np.hstack((dq, ddq))

    # ---------------------------
    # Mechanics: M, C, G, B
    # ---------------------------

    def _mass_matrix(self, q: Array) -> Array:
        """
        M(q) for 2-link planar arm (Spong, standard form).
        """
        p = self.p
        _, q2 = float(q[0]), float(q[1])

        a = p.I1 + p.I2 + p.m1 * p.lc1**2 + p.m2 * (p.l1**2 + p.lc2**2)
        b = p.m2 * p.l1 * p.lc2
        d = p.I2 + p.m2 * p.lc2**2

        c2 = np.cos(q2)

        M11 = a + 2.0 * b * c2
        M12 = d + b * c2
        M22 = d

        M = np.array([[M11, M12], [M12, M22]], dtype=float)
        return M

    def _coriolis_matrix(self, q: Array, dq: Array) -> Array:
        """
        C(q, dq) such that C(q, dq) * dq gives Coriolis/centrifugal terms.
        """
        p = self.p
        _, q2 = float(q[0]), float(q[1])
        dq1, dq2 = float(dq[0]), float(dq[1])

        s2 = np.sin(q2)
        b = p.m2 * p.l1 * p.lc2

        c11 = -2.0 * b * s2 * dq2
        c12 = -1.0 * b * s2 * dq2
        c21 = 1.0 * b * s2 * dq1
        c22 = 0.0

        C = np.array([[c11, c12], [c21, c22]], dtype=float)
        return C

    def _gravity(self, q: Array) -> Array:
        """
        Gravity vector G(q).
        """
        p = self.p
        q1, q2 = float(q[0]), float(q[1])

        g1 = (p.m1 * p.lc1 + p.m2 * p.l1) * p.g * np.cos(q1) + p.m2 * p.lc2 * p.g * np.cos(q1 + q2)
        g2 = p.m2 * p.lc2 * p.g * np.cos(q1 + q2)

        return np.array([g1, g2], dtype=float)

    def _viscous(self) -> Array:
        """
        Joint-space viscous damping matrix B (diagonal).
        """
        return np.array([[self.p.b1, 0.0], [0.0, self.p.b2]], dtype=float)

    # ---------------------------
    # Diagnostics (optional)
    # ---------------------------

    def jacobians_fd(self, x: Array, u: Array, eps: float = 1e-6) -> Tuple[Array, Array]:
        """
        Finite-difference Jacobians of the *discrete* map x^+ = step(x, u):
            A = ∂step/∂x,  B = ∂step/∂u

        Delegates to models.discretization.discrete_jacobians_fd.
        """
        x = np.asarray(x, dtype=float).reshape(self.n)
        u = np.asarray(u, dtype=float).reshape(self.m)
        return discrete_jacobians_fd(self._step, x, u, eps=eps)
