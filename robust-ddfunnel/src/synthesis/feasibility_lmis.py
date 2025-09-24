# src/synthesis/feasibility_lmis.py
from typing import Dict, List, Tuple

import numpy as np


def bounds_at_k(k: int, Q_bounds: Dict, R_bounds: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fetch time-varying caps at step k.

    Q_bounds / R_bounds formats supported:
      - time-varying (diagonal): {"diag": np.ndarray of shape (T, n) or (T+1, n)}
      - time-invariant (full):   {"full": np.ndarray of shape (n,n) or (m,m)}
    """
    Qk = np.diag(np.asarray(Q_bounds["diag"][k], float)) if "diag" in Q_bounds else np.asarray(Q_bounds["full"], float)

    Rk = np.diag(np.asarray(R_bounds["diag"][k], float)) if "diag" in R_bounds else np.asarray(R_bounds["full"], float)

    return Qk, Rk


def per_step_feasibility_specs(
    Q_bounds: Dict,
    R_bounds: Dict,
    k_states: List[int],
    k_inputs: List[int],
) -> Dict[str, List[np.ndarray]]:
    """
    Build the list of *per-step* caps to enforce in the SDP for a given segment.

    Returns:
      {
        "Qk_list": [Q_k for k in k_states],                      # each (n,n)
        "Rk_list": [R_k for k in k_inputs],                      # each (m,m)
      }

    SDP layer then enforces:
      - for each Qk in Qk_list:            Q << Qk
      - for each Rk in Rk_list:            [[Rk, Y], [Y.T, Q]] >> 0
    """
    Qk_list: List[np.ndarray] = []
    for ks in k_states:
        Qk, _ = bounds_at_k(ks, Q_bounds, {"full": np.eye(1)})  # dummy R
        Qk_list.append(Qk)

    Rk_list: List[np.ndarray] = []
    for ku in k_inputs:
        _, Rk = bounds_at_k(ku, {"full": np.eye(Qk_list[0].shape[0])}, R_bounds)  # dummy Q
        Rk_list.append(Rk)

    return {"Qk_list": Qk_list, "Rk_list": Rk_list}
