# src/core/constants_io.py
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml


@dataclass
class NominalConstants:
    # required at this stage
    dt: float
    N: int
    v: float

    # optional; to be filled later
    gamma: Optional[float] = None
    L_J: Optional[float] = None
    L_r: Optional[float] = None
    C: Optional[float] = None         # convenience: C = L_J * v
    T_tilde: Optional[int] = None     # e.g., 2*T-1 for your segments

    # context (nice to have in file)
    x_goal: Optional[np.ndarray] = None
    u_goal: Optional[np.ndarray] = None

def save_constants_yaml(path: Path, consts: NominalConstants) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(consts)
    # ensure numpy arrays (x_goal, u_goal) serialize nicely
    for k in ("x_goal", "u_goal"):
        if isinstance(data.get(k), np.ndarray):
            data[k] = data[k].tolist()
    with path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)

def load_constants_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f)
