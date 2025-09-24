# scripts/run_nominal_on_plant.py
from __future__ import annotations

import pathlib
import sys

# add ./src to sys.path
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from nominal.plant_rollout import main  # pyright: ignore[reportMissingImports]

if __name__ == "__main__":
    main()
