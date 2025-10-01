# Robust Data-Driven Control for Nonlinear Systems Using Quadratic Funnels

This repository hosts a paper-faithful reference implementation of the data-driven quadratic funnel approach for robust control of nonlinear systems. The current demo focuses on a two-degree-of-freedom robot arm, including nominal trajectory generation, uncertainty modeling, controller synthesis, and visualization utilities.

## Highlights
- End-to-end pipeline that reproduces the funnel-based robust controller from the paper.
- Robot-arm twin model that separates true plant dynamics from the model used for synthesis.
- Ready-to-run scripts for generating nominal trajectories, solving the funnel SDPs, and plotting state/input funnels over time.
- Lightweight utilities for inspecting feasible ellipsoids, Lyapunov certificates, and reproducing paper-quality figures.

## Getting Started
1. Ensure Python 3.10 or newer is available.
2. (Optional) create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. Install the package and dependencies:
   ```bash
   pip install -e .
   # or include development tooling
   pip install -e .[dev]
   ```

> Commercial solvers (e.g. MOSEK) are not required but can speed up the SDP solves if installed separately.

## Quickstart Pipeline
The `scripts/run_all.sh` helper executes the full demonstration: nominal rollout, bound tightening, funnel synthesis, and plot generation.
```bash
bash scripts/run_all.sh
```
Artifacts are dropped under `results/online_runs/` and `results/nominal_plots/`.

To run an individual step, prepend `PYTHONPATH=src` so the package is importable, for example:
```bash
PYTHONPATH=src python scripts/run_online_ddfunnel.py \
  --cfg configs/robot_arm_2dof.yaml \
  --nom data/nominal_trajectory/nominal_q1_+4.00_q2_-1.00_N600.npz \
  --constants data/nominal_constants/constants_q1_+4.00_q2_-1.00_N600.yaml
```
Run `--help` on any script to inspect optional flags.


## Repository Layout
- `configs/`: YAML configuration files describing plant/twin parameters, constraints, and solver settings.
- `data/`: Nominal trajectories and constants generated during the pipeline.
- `scripts/`: End-to-end and diagnostic utilities (nominal rollouts, synthesis, plotting).
- `src/robust_ddfunnel/`: Core library with dynamics models, data stacking, SDP assembly, and visualization helpers.
- `tests/`: Pytest suite covering key numerical building blocks.

## Testing
Run the unit tests with:
```bash
pytest
```
