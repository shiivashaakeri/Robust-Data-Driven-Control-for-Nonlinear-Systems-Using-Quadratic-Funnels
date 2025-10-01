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


def estimate_Lr_discrete(  # noqa: PLR0915, C901, PLR0912
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
    # Local estimation (recommended)
    local_only: bool = False,
    eps_x: float = 1e-3,
    eps_u: float = 1e-2,
    # NEW: control clipping & denominator
    clip_to_box: bool = False,
) -> float:
    """
    Estimate L_r along (X_nom,U_nom). If local_only=True, use fixed small radii
    eps_x/eps_u around each nominal point and (by default) DO NOT clip.
    The remainder-ratio denominator uses the *actual* applied perturbation:
        delta_applied = [x_plus-x0; u_plus-u0].
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

    def _J_at(x: Array, u: Array) -> Array:
        A, B = discrete_jacobians_fd(step, x, u, eps=fd_eps)  # (n,n),(n,m)
        return np.concatenate([A, B], axis=1)  # (n, n+m)

    # -------------------------------
    # (1) Hessian-based upper bound
    # -------------------------------
    Lr_hess = 0.0
    if use_hessian_bound:
        if local_only:
            h_vec = np.concatenate([np.full(n, float(eps_x)), np.full(m, float(eps_u))])
        else:
            box_width = np.concatenate([X_high - X_low, U_high - U_low])
            h_vec = np.maximum(h_rel * box_width, 1e-12)

        for k in range(T):
            z0 = np.concatenate([X_nom[k], U_nom[k]])
            D = rng.normal(size=(dirs_per_point_hess, d))
            D /= np.linalg.norm(D, axis=1, keepdims=True) + 1e-15
            for dir_vec in D:
                step_vec = h_vec * dir_vec  # desired displacement from z0
                z_plus = z0 + step_vec
                z_minus = z0 - step_vec
                if clip_to_box:
                    z_plus = np.clip(z_plus, z_low, z_high)
                    z_minus = np.clip(z_minus, z_low, z_high)
                # effective step actually applied
                step_eff = 0.5 * (np.linalg.norm(z_plus - z0) + np.linalg.norm(z0 - z_minus))
                if step_eff < 1e-15:
                    continue

                x_p, u_p = z_plus[:n], z_plus[n:]
                x_m, u_m = z_minus[:n], z_minus[n:]

                Jp = _J_at(x_p, u_p)
                Jm = _J_at(x_m, u_m)
                Jdiff = (Jp - Jm) / (2.0 * step_eff)

                val = float(np.linalg.norm(Jdiff, 2 if norm == "2" else "fro"))
                Lr_hess = max(Lr_hess, 0.5 * val)  # 0.5 * ||D^2F||

    # -------------------------------------
    # (2) Empirical remainder-ratio bound
    # -------------------------------------
    Lr_rem = 0.0
    if use_remainder_ratio:
        if local_only:
            d_vec = np.concatenate([np.full(n, float(eps_x)), np.full(m, float(eps_u))])
        else:
            box_width = np.concatenate([X_high - X_low, U_high - U_low])
            d_vec = np.maximum(delta_rel * box_width, 1e-12)

        for k in range(T):
            x0 = X_nom[k]
            u0 = U_nom[k]
            z0 = np.concatenate([x0, u0])
            x1_nom = step(x0, u0)  # F(x*,u*)
            J0 = _J_at(x0, u0)  # [A|B] at (x*,u*)

            D = rng.normal(size=(dirs_per_point_rem, d))
            D /= np.linalg.norm(D, axis=1, keepdims=True) + 1e-15
            for dir_vec in D:
                target = d_vec * dir_vec
                z_plus = z0 + target
                if clip_to_box:
                    z_plus = np.clip(z_plus, z_low, z_high)
                # actual applied perturbation:
                delta_applied = z_plus - z0
                if np.linalg.norm(delta_applied) < 1e-15:
                    continue

                eta = delta_applied[:n]
                xi = delta_applied[n:]
                x1_plus = step(x0 + eta, u0 + xi)
                x1_lin = x1_nom + J0 @ delta_applied  # affine linearization
                r = x1_plus - x1_lin

                denom = float(np.linalg.norm(delta_applied) ** 2)
                val = float(np.linalg.norm(r) / (denom + 1e-18))
                Lr_rem = max(Lr_rem, val)

    Lr = max(Lr_hess, Lr_rem)
    Lr = Lr * (1.0 + float(rel_margin)) + float(abs_margin)
    return float(Lr)


# def beta_from_segment(
#     etas,
#     xis,
#     k_i: int,
#     C: float,
#     gamma: float,
#     L_r: float,
#     *,
#     ks: Optional[Sequence[int]] = None,
#     k_start: Optional[int] = None,
# ) -> float:
#     """
#     Aggregate β over a data window (eq. β_def in the paper):

#         β_i = Σ_k ( C*|k - k_i| * ||[η(k); ξ(k)]||_2 + gamma + L_r * ||[η(k); ξ(k)]||_2^2 )^2

#     Parameters
#     ----------
#     etas : sequence of (n,) arrays
#         Deviation states η(k) collected over the data window (in order).
#     xis : sequence of (m,) arrays
#         Input deviations ξ(k) collected over the data window (in order).
#     k_i : int
#         Start index of segment i.
#     C : float
#         The linear-in-time bound coefficient (C = L_J * v).
#     gamma : float
#         Uniform mismatch bound.
#     L_r : float
#         Quadratic remainder coefficient for linearization error.
#     ks : sequence of int, optional
#         Absolute time indices for each sample in `etas`/`xis` (same length). If provided,
#         |k - k_i| is computed exactly as abs(ks[t] - k_i).
#     k_start : int, optional
#         Absolute index of the first sample in this window (k_i^D). If provided (and `ks` is not),
#         distances are abs((k_start + t) - k_i), t=0..L-1.

#     Returns
#     -------
#     float
#         β_i value.

#     Notes
#     -----
#     - If neither `ks` nor `k_start` is provided, we fallback to using the local window
#       index t=0..L-1 as |k - k_i|; this is OK only if your window starts at k_i
#       (i.e., k_start == k_i). Prefer passing `k_start` (k_i^D) or `ks`.
#     """
#     L = min(len(etas), len(xis))
#     if L == 0:
#         return 0.0

#     # Build ||[η; ξ]||_2 per sample
#     z_norms = np.empty(L, dtype=float)
#     for t in range(L):
#         e = np.asarray(etas[t], dtype=float).ravel()
#         u = np.asarray(xis[t], dtype=float).ravel()
#         z_norms[t] = float(np.linalg.norm(np.concatenate([e, u]), ord=2))

#     # Distances |k - k_i| in "steps"
#     if ks is not None:
#         ks_arr = np.asarray(ks, dtype=int)
#         if ks_arr.size != L:
#             raise ValueError("len(ks) must match number of samples in etas/xis")
#         dists = np.abs(ks_arr - int(k_i)).astype(float)
#     elif k_start is not None:
#         # samples correspond to k = k_start + t
#         dists = np.abs((int(k_start) + np.arange(L)) - int(k_i)).astype(float)
#     else:
#         # Fallback: treat local index as distance (only correct if k_start == k_i)
#         dists = np.arange(L, dtype=float)

#     # term_k = C*|k-k_i|*||z|| + gamma + L_r*||z||^2
#     terms = C * dists * z_norms + gamma + L_r * (z_norms**2)
#     beta = float(np.sum(terms**2))
#     return beta


def beta_from_segment(
    etas,
    xis,
    k_i: int,
    C: float,
    gamma: float,
    L_r: float,
    k_start: int | None = None,
    z_clip: float | None = None,
) -> float:
    """
    Implements paper Equation 42: β_i = Σ_{k∈T_i^D} (C|k-k_i| ||[η(k); ξ(k)]|| + gamma + L_r ||[η(k); ξ(k)]||²)²

    Parameters
    ----------
    etas : sequence of (n,) arrays
        Deviation states η(k) collected over the data window (in order).
    xis : sequence of (m,) arrays
        Input deviations ξ(k) collected over the data window (in order).
    k_i : int
        Start index of segment i.
    C : float
        The linear-in-time bound coefficient (C = L_J * v).
    gamma : float
        Uniform mismatch bound.
    L_r : float
        Quadratic remainder coefficient for linearization error.
    k_start : int, optional
        Absolute index of the first sample in this window (k_i^D). If provided,
        distances are abs((k_start + t) - k_i), t=0..L-1.
    z_clip : float, optional
        Optional clipping to avoid crazy β from a few outliers.
    """
    L = min(len(etas), len(xis))
    if L == 0:
        return 0.0

    s = 0.0
    for idx, (e, xi) in enumerate(zip(etas, xis)):
        z = np.concatenate([e, xi])
        if z_clip is not None:
            z = np.clip(z, -z_clip, z_clip)
        z_norm = float(np.linalg.norm(z))

        # Calculate |k - k_i| as in paper
        if k_start is not None:
            k = k_start + idx  # absolute time index
            dist = abs(k - k_i)
        else:
            # Fallback: treat local index as distance (only correct if k_start == k_i)
            dist = idx

        # Paper Equation 42: term_k = C*|k-k_i|*||z|| + gamma + L_r*||z||^2
        term = C * dist * z_norm + gamma + L_r * (z_norm**2)
        s += term * term

    return float(s)
