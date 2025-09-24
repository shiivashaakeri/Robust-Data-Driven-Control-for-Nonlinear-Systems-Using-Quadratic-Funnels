# src/core/bounds.py

from __future__ import annotations

from typing import Callable, Literal, Optional, Tuple, Union

import numpy as np

from models.discretization import discrete_jacobians_fd

Array = np.ndarray
StepFn = Callable[[Array, Array], Array]


def gamma_pointwise_on_nominal(
    step_plant: StepFn,
    step_twin: StepFn,
    X_nom: Array,
    U_nom: Array,
    *,
    norm: Literal["l2", "linf"] = "l2",
    abs_margin: float = 1e-6,
    rel_margin: float = 0.0,
    return_per_step: bool = False,
) -> Union[float, Tuple[float, Array]]:
    """
    Pointwise mismatch bound gamma on the nominal trajectory (NO rollout).

    For each k = 0..N-1, evaluate both discrete one-step maps at the same
    nominal pair (x_k, u_k):
        x⁺_plant = step_plant(x_k, u_k)
        x⁺_twin  = step_twin (x_k, u_k)
        e_k      = x⁺_plant - x⁺_twin
    Then gamma = max_k ||e_k||, with an optional safety margin:
        gamma := (max_k ||e_k||) * (1 + rel_margin) + abs_margin

    Notes
    -----
    - This intentionally does NOT feed back the plant's next state; each step
      is evaluated at the nominal (x_k, u_k) only (no error accumulation).
    - `step_plant` and `step_twin` should be *discrete* steppers, e.g. from
      models.discretization.make_stepper(f, dt, method).

    Parameters
    ----------
    step_plant : callable
        Discrete step function for the physical plant: x⁺ = step(x, u).
    step_twin : callable
        Discrete step function for the digital twin:  x⁺ = step(x, u).
    X_nom : (N+1, n) array
        Nominal states along the horizon (includes terminal state).
    U_nom : (N, m) array
        Nominal inputs along the horizon.
    norm : {"l2","linf"}, optional
        Vector norm used for ||e_k||. Default "l2".
    abs_margin : float, optional
        Small absolute safety addition to the max norm. Default 1e-6.
    rel_margin : float, optional
        Relative safety factor (fraction of the max). Default 0.0.
    return_per_step : bool, optional
        If True, also return the per-step mismatch norms as an array of length N.

    Returns
    -------
    gamma : float
        Bound on the one-step mismatch magnitude over the nominal path.
    ek_norms : (N,) array, optional
        Per-step mismatch norms (returned only if return_per_step=True).

    Raises
    ------
    ValueError
        If X_nom / U_nom shapes are inconsistent
    """

    X_nom = np.asarray(X_nom, dtype=float)
    U_nom = np.asarray(U_nom, dtype=float)

    if X_nom.ndim != 2 or U_nom.ndim != 2:
        raise ValueError("X_nom and U_nom must be 2D arrays")
    if X_nom.shape[0] != U_nom.shape[0] + 1:
        raise ValueError("X_nom and U_nom must have consistent shapes")

    N = U_nom.shape[0]
    ek_norms = np.zeros(N, dtype=float)

    for k in range(N):
        xk = X_nom[k]
        uk = U_nom[k]
        x_plant = step_plant(xk, uk)
        x_twin = step_twin(xk, uk)
        e = x_plant - x_twin
        if norm == "l2":
            ek_norms[k] = float(np.linalg.norm(e, ord=2))
        elif norm == "linf":
            ek_norms[k] = float(np.linalg.norm(e, ord=np.inf))
        else:
            raise ValueError(f"Invalid norm: {norm}")

    gamma_raw = float(np.max(ek_norms)) if N > 0 else 0.0
    gamma = gamma_raw * (1.0 + float(rel_margin)) + float(abs_margin)

    if return_per_step:
        return gamma, ek_norms
    else:
        return gamma


def estimate_LJ_discrete(
    step: StepFn,
    X_box: Tuple[Array, Array],
    U_box: Tuple[Array, Array],
    *,
    samples: int = 200,
    h_rel: float = 1e-4,
    fd_eps: float = 1e-6,
    rng: Optional[np.random.Generator] = None,
    norm: Literal["2", "fro"] = "2",
    abs_margin: float = 0.0,
    rel_margin: float = 0.0,
    use_random_dirs: bool = False,
    dirs_per_point: int = 6,
) -> float:
    """
    Estimate L_J for the *discrete* map x⁺ = step(x,u) over the box XxU.

    L_J is a Lipschitz constant of the Jacobian:
        || J(z1) - J(z2) || <= L_J || z1 - z2 ||,   z=[x;u]
    which equals sup_z ||D^2 F(z)|| in operator norm. We approximate via
    symmetric directional FD on J:

        along direction d (||d||=1):
            L_local ≈ || J(z + h d) - J(z - h d) || / (2h)

    We take the maximum over sampled points z and directions d.

    Parameters
    ----------
    step : callable
        Discrete step function: x⁺ = step(x, u).
    X_box : (low, high)
        State bounds (n,). Each is a length-n array.
    U_box : (low, high)
        Input bounds (m,). Each is a length-m array.
    samples : int
        Number of (x,u) samples in the box.
    h_rel : float
        Relative step size h as a fraction of box width per coordinate (typ. 1e-4).
    fd_eps : float
        Inner FD epsilon passed to discrete_jacobians_fd.
    rng : np.random.Generator, optional
        RNG for sampling. If None, uses np.random.default_rng(0).
    norm : {"2","fro"}
        Matrix norm for ||·|| on (A|B). "2" = spectral norm, "fro" = Frobenius.
    abs_margin, rel_margin : float
        Safety cushions applied at the end: L_J ← L_J*(1+rel) + abs.
    use_random_dirs : bool
        If True, use random unit directions in R^{n+m}; else use coordinate axes.
    dirs_per_point : int
        Number of random directions per point if use_random_dirs=True.

    Returns
    -------
    LJ : float
        Estimated Lipschitz constant of the Jacobian over XxU
    """
    X_low, X_high = (np.asarray(X_box[0], float), np.asarray(X_box[1], float))
    U_low, U_high = (np.asarray(U_box[0], float), np.asarray(U_box[1], float))

    n = X_low.size
    m = U_low.size
    d = n + m

    if rng is None:
        rng = np.random.default_rng(0)

    width = np.concatenate([X_high - X_low, U_high - U_low])
    h_vec = np.maximum(h_rel * width, 1e-12)

    def _J_at(x: Array, u: Array) -> Array:
        A, B = discrete_jacobians_fd(step, x, u, eps=fd_eps)
        return np.concatenate([A, B], axis=1)

    Z_samples = rng.random((samples, d))
    Z_samples = np.concatenate(
        [
            X_low + Z_samples[:, :n] * (X_high - X_low),
            U_low + Z_samples[:, n:] * (U_high - U_low),
        ],
        axis=1,
    )
    if use_random_dirs:
        D_list = []
        for _ in range(samples):
            D = rng.random((dirs_per_point, d))
            D /= np.linalg.norm(D, axis=1, keepdims=True) + 1e-15
            D_list.append(D)
    else:
        D_coord = np.eye(d)
        D_list = [D_coord] * samples

    best = 0.0
    for s in range(samples):
        z = Z_samples[s]

        for dir_vec in D_list[s]:
            # Directional step sizes: scale by per-coordinate h_vec
            # Effective h = ||h_vec ∘ dir_vec||_2 to keep a meaningful magnitude.
            step_vec = h_vec * dir_vec
            h_eff = float(np.linalg.norm(step_vec))
            if h_eff < 1e-15:
                continue
            # z_plus / z_minus and clip to the box
            z_plus = np.clip(z + step_vec, np.concatenate([X_low, U_low]), np.concatenate([X_high, U_high]))
            z_minus = np.clip(z - step_vec, np.concatenate([X_low, U_low]), np.concatenate([X_high, U_high]))

            x_p, u_p = z_plus[:n], z_plus[n:]
            x_m, u_m = z_minus[:n], z_minus[n:]

            Jp = _J_at(x_p, u_p)
            Jm = _J_at(x_m, u_m)
            Jdiff = (Jp - Jm) / (2.0 * h_eff)

            if norm == "2":
                val = float(np.linalg.norm(Jdiff, 2))
            elif norm == "fro":
                val = float(np.linalg.norm(Jdiff, "fro"))
            else:
                raise ValueError("norm must be '2' or 'fro'.")

            best = max(best, val)

    return best * (1.0 + float(rel_margin)) + float(abs_margin)

def estimate_Lr_discrete(  # noqa: PLR0915
    step: StepFn,
    X_nom: Array,
    U_nom: Array,
    X_box: Tuple[Array, Array],
    U_box: Tuple[Array, Array],
    *,
    use_hessian_bound: bool = True,
    use_remainder_ratio: bool = True,
    # Hessian-bound knobs
    h_rel: float = 1e-4,
    dirs_per_point_hess: int = 4,
    # Remainder-ratio knobs
    delta_rel: float = 1e-2,
    dirs_per_point_rem: int = 8,
    # Common knobs
    fd_eps: float = 1e-6,
    norm: Literal["2", "fro"] = "2",
    rng: Optional[np.random.Generator] = None,
    rel_margin: float = 0.0,
    abs_margin: float = 0.0,
) -> float:
    """
    Estimate L_r for the *discrete* plant map F(x,u)=step(x,u) along a given nominal
    trajectory (X_nom, U_nom), with bounds enforced on XxU.

    We find a scalar L_r so that, for perturbations δ=[η;ξ] near each nominal (x*,u*),
        || F(x*+η, u*+ξ) - ( F(x*,u*) + A*η + B*ξ ) || <= L_r ||δ||^2,
    where [A*,B*] = Jacobian of F at (x*,u*).

    Two estimators (take the maximum):
      (1) Hessian bound:    L_r ≈ 0.5 * sup || D^2 F ||   via symmetric FD on J.
      (2) Remainder ratio:  sup_δ  || r || / ||δ||^2      with random small δ.

    Parameters
    ----------
    step : callable, x⁺ = step(x,u).
    X_nom, U_nom : arrays
        Nominal trajectory (T+1,n), (T,m).
    X_box, U_box : (low, high)
        State and input bounds (length-n / length-m vectors).
    use_hessian_bound, use_remainder_ratio : bool
        Enable/disable each estimator.
    h_rel : float
        Relative step size for Hessian FD (scale per-coordinate by box width).
    dirs_per_point_hess : int
        Random directions per nominal point for Hessian estimator.
    delta_rel : float
        Relative perturbation radius for remainder estimator (fraction of box width).
    dirs_per_point_rem : int
        Random directions per nominal point for remainder estimator.
    fd_eps : float
        Epsilon for inner Jacobian finite differences.
    norm : {"2","fro"}
        Matrix norm for Jacobian differences and vector 2-norm for residuals.
    rng : np.random.Generator
        RNG; default uses np.random.default_rng(0).
    rel_margin, abs_margin : float
        Safety margins applied at the end: L_r ← L_r*(1+rel) + abs.

    Returns
    -------
    Lr_hat : float
        Conservative estimate of L_r for use in the paper's bound.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    X_nom = np.asarray(X_nom, float)
    U_nom = np.asarray(U_nom, float)
    assert X_nom.ndim == 2 and U_nom.ndim == 2 and X_nom.shape[0] == U_nom.shape[0] + 1
    T = U_nom.shape[0]
    n = X_nom.shape[1]
    m = U_nom.shape[1]
    d = n + m

    X_low, X_high = (np.asarray(X_box[0], float), np.asarray(X_box[1], float))
    U_low, U_high = (np.asarray(U_box[0], float), np.asarray(U_box[1], float))
    z_low = np.concatenate([X_low, U_low])
    z_high = np.concatenate([X_high, U_high])
    box_width = z_high - z_low

    def _J_at(x: Array, u: Array) -> Array:
        A, B = discrete_jacobians_fd(step, x, u, eps=fd_eps)  # (n,n),(n,m)
        return np.concatenate([A, B], axis=1)  # (n, n+m)

    # -------------------------------
    # (1) Hessian-based upper bound
    # -------------------------------
    Lr_hess = 0.0
    if use_hessian_bound:
        # per-coordinate step magnitudes (avoid zero)
        h_vec = np.maximum(h_rel * box_width, 1e-12)
        for k in range(T):
            z0 = np.concatenate([X_nom[k], U_nom[k]])
            # random unit directions in R^{n+m}
            D = rng.normal(size=(dirs_per_point_hess, d))
            D /= np.linalg.norm(D, axis=1, keepdims=True) + 1e-15
            for dir_vec in D:
                step_vec = h_vec * dir_vec
                h_eff = float(np.linalg.norm(step_vec))
                if h_eff < 1e-15:
                    continue
                z_plus = np.clip(z0 + step_vec, z_low, z_high)
                z_minus = np.clip(z0 - step_vec, z_low, z_high)
                x_p, u_p = z_plus[:n], z_plus[n:]
                x_m, u_m = z_minus[:n], z_minus[n:]

                Jp = _J_at(x_p, u_p)
                Jm = _J_at(x_m, u_m)
                Jdiff = (Jp - Jm) / (2.0 * h_eff)

                val = float(np.linalg.norm(Jdiff, 2 if norm == "2" else "fro"))
                # Taylor: r ≈ 0.5 * D^2F[z](δ,δ) ⇒ ||r|| ≤ 0.5 ||D^2F|| ||δ||^2
                Lr_hess = max(Lr_hess, 0.5 * val)

    # -------------------------------------
    # (2) Empirical remainder-ratio bound
    # -------------------------------------
    Lr_rem = 0.0
    if use_remainder_ratio:
        # perturbation magnitude per coordinate
        d_vec = np.maximum(delta_rel * box_width, 1e-12)
        for k in range(T):
            x0 = X_nom[k]
            u0 = U_nom[k]
            x1_nom = step(x0, u0)  # F(x*,u*)
            J0 = _J_at(x0, u0)     # [A|B] at (x*,u*)

            # random directions in R^{n+m}
            D = rng.normal(size=(dirs_per_point_rem, d))
            D /= np.linalg.norm(D, axis=1, keepdims=True) + 1e-15
            for dir_vec in D:
                delta = d_vec * dir_vec  # [η; ξ]
                eta = delta[:n]
                xi = delta[n:]

                x_plus = np.clip(x0 + eta, X_low, X_high)
                u_plus = np.clip(u0 + xi, U_low, U_high)

                x1_plus = step(x_plus, u_plus)
                # linear prediction about (x0,u0): x1_lin = x1_nom + A*eta + B*xi = x1_nom + J0 @ [eta;xi]
                x1_lin = x1_nom + J0 @ delta
                r = x1_plus - x1_lin

                denom = float(np.linalg.norm(delta) ** 2 + 1e-18)
                val = float(np.linalg.norm(r) / denom)
                Lr_rem = max(Lr_rem, val)

    Lr = max(Lr_hess, Lr_rem)
    Lr = Lr * (1.0 + float(rel_margin)) + float(abs_margin)
    return float(Lr)
