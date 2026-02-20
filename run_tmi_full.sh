#!/usr/bin/env bash
# =============================================================================
# TMI Full CEC 2022 Benchmark — Mac / Linux runner (tmux-friendly)
#
# Usage:
#   1. Clone the repo and cd into it
#   2. Run:  bash run_tmi_full.sh
#
# The script will:
#   - Create/activate a Python venv
#   - Install all dependencies
#   - Run the full 4-variant benchmark (30 seeds × 12 functions × 200k FEs)
#   - Save results to results/tmi_cec2022_<timestamp>.csv
#
# Recommended: run inside a tmux session so SSH disconnects don't kill it.
#   tmux new -s tmi
#   bash run_tmi_full.sh
#   (detach with Ctrl+B then D)
#   (reattach with: tmux attach -t tmi)
# =============================================================================

set -euo pipefail

# ---------- configuration (edit here if needed) ----------
VENV_DIR=".venv"
SEEDS=30
DIM=10
MAX_FE=200000
FUNCS="1 2 3 4 5 6 7 8 9 10 11 12"
# E = NL-SHADE + random injection (same trigger/cap as D but no ORC).
# This is the critical ablation: if D >> E, ORC geometry is proven useful.
VARIANTS="A B C D E"
WORKERS=$(python3 -c "import os; print(max(1, os.cpu_count() - 1))" 2>/dev/null || echo 4)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUT="results/tmi_cec2022_${TIMESTAMP}.csv"
LOG="results/tmi_cec2022_${TIMESTAMP}.log"
# ---------------------------------------------------------

echo "=============================================="
echo " TMI - Topological Manifold Injection"
echo " CEC 2022 Full Benchmark"
echo "=============================================="
echo " Seeds    : ${SEEDS}"
echo " Dim      : ${DIM}"
echo " max_FE   : ${MAX_FE}"
echo " Variants : ${VARIANTS}"
echo " Workers  : ${WORKERS}"
echo " Output   : ${OUT}"
echo " Log      : ${LOG}"
echo "=============================================="

# ---------- venv setup ----------
if [ ! -d "${VENV_DIR}" ]; then
    echo "[setup] Creating virtual environment..."
    python3 -m venv "${VENV_DIR}"
fi

echo "[setup] Activating venv..."
source "${VENV_DIR}/bin/activate"

echo "[setup] Installing / upgrading dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

echo "[setup] Done. Python: $(python --version)"

# ---------- create results dir ----------
mkdir -p results

# ---------- run benchmark ----------
echo ""
echo "[run] Starting benchmark at $(date)"
echo ""

# ---------- D=10 run (CEC 2022 primary) ----------
echo "[run] D=10 benchmark starting..."
python benchmarks/run_tmi_benchmark.py \
    --seeds    "${SEEDS}" \
    --dim      "${DIM}" \
    --max-fe   "${MAX_FE}" \
    --funcs    ${FUNCS} \
    --variants ${VARIANTS} \
    --workers  "${WORKERS}" \
    --out      "${OUT}" \
    2>&1 | tee "${LOG}"

# ---------- D=20 run (curse-of-dimensionality validation) ----------
OUT20="results/tmi_cec2022_D20_${TIMESTAMP}.csv"
LOG20="results/tmi_cec2022_D20_${TIMESTAMP}.log"
echo ""
echo "[run] D=20 benchmark starting (max_FE=1,000,000)..."
python benchmarks/run_tmi_benchmark.py \
    --seeds    "${SEEDS}" \
    --dim      20 \
    --max-fe   1000000 \
    --funcs    ${FUNCS} \
    --variants ${VARIANTS} \
    --workers  "${WORKERS}" \
    --out      "${OUT20}" \
    2>&1 | tee "${LOG20}"

echo ""
echo "[done] Benchmark complete at $(date)"
echo "Results : ${OUT}"
echo "Log     : ${LOG}"

# ---------- auto-analyze ----------
echo ""
echo "[analyze] Analyzing D=10 results..."
python benchmarks/analyze_results.py "${OUT}" --latex \
    2>&1 | tee -a "${LOG}"

echo ""
echo "[analyze] Analyzing D=20 results..."
python benchmarks/analyze_results.py "${OUT20}" --latex \
    2>&1 | tee -a "${LOG20}"

echo ""
echo "========================================"
echo "All done."
echo "D=10 CSV     : ${OUT}"
echo "D=20 CSV     : ${OUT20}"
echo "Figures      : results/figures/"
echo "========================================"
