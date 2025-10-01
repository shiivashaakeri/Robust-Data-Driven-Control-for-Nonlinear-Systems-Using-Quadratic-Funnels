#!/usr/bin/env bash
set -euo pipefail

CFG="configs/robot_arm_2dof.yaml"
SLUG="q1_+4.00_q2_-1.00_N600"
NOM="data/nominal_trajectory/nominal_${SLUG}.npz"
CONST="data/nominal_constants/constants_${SLUG}.yaml"
RUN_ROOT="results/online_runs"
RUN_DIR="${RUN_ROOT}/${SLUG}"
PLOT_DIR="results/nominal_plots"
FPS=30

mkdir -p "$(dirname "$NOM")" "$(dirname "$CONST")" "$RUN_ROOT" "$PLOT_DIR"

echo "1) Nominal (debug)…"
PYTHONPATH=src python scripts/run_nominal.py --debug

echo "2) Nominal rolled on plant…"
PYTHONPATH=src python scripts/run_nominal_on_plant.py \
  --cfg "$CFG" \
  --nom "$NOM" \
  --fps "$FPS" \
  --no-anim \
  --no-save-anim \
  --no-show

echo "3) Compute bounds (debug)…"
PYTHONPATH=src python scripts/compute_bounds.py \
  --cfg "$CFG" \
  --nom "$NOM" \
  --debug

echo "4) Feasible ellipsoids (plots)…"
PYTHONPATH=src python scripts/compute_feasible_ellipsoids.py \
  --cfg "$CFG" \
  --nom "$NOM" \
  --plot --save-plots "$PLOT_DIR"

echo "5) Online DD-Funnel synthesis on plant…"
PYTHONPATH=src python scripts/run_online_ddfunnel.py \
  --cfg "$CFG" \
  --nom "$NOM" \
  --constants "$CONST" \
  --solver SCS \
  --out "$RUN_ROOT" \
  --save-plots \
  --alpha 0.995 \
  --ignore-k0

echo "6) Plot funnels over trajectories…"
PYTHONPATH=src python scripts/plot_funnels.py \
  --cfg "$CFG" \
  --run "${RUN_DIR}/online_run_${SLUG}.npz" \
  --save \
  --no-show

echo "7) Plot Lyapunov along time…"
PYTHONPATH=src python scripts/plot_lyapunov.py \
  --cfg "$CFG" \
  --run "${RUN_DIR}/online_run_${SLUG}.npz" \
  --save \
  --no-show

echo "Done. Outputs:"
echo "  - Online run: ${RUN_DIR}/online_run_${SLUG}.npz"
echo "  - Plots: ${RUN_DIR}/ and ${PLOT_DIR}/"
