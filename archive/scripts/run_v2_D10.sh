#!/usr/bin/env bash
# =============================================================================
# TMI v2 Full D=10 Benchmark
# Runs the newly updated v2 architecture (Reversed Descent + Convergence Gate)
# =============================================================================

set -euo pipefail

VENV_DIR=".venv"
SEEDS=30
DIM=10
MAX_FE=200000
FUNCS="1 2 3 4 5 6 7 8 9 10 11 12"
VARIANTS="A B C D E"
WORKERS=$(python3 -c "import os; print(max(1, os.cpu_count() - 1))" 2>/dev/null || echo 4)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUT="results/tmi_v2_D10_${TIMESTAMP}.csv"
LOG="results/tmi_v2_D10_${TIMESTAMP}.log"

echo "=============================================="
echo " TMI v2 - Topological Manifold Injection"
echo " CEC 2022 D=10 Benchmark"
echo "=============================================="
echo " Output : ${OUT}"
echo " Workers: ${WORKERS}"
echo "=============================================="

if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

mkdir -p results

python benchmarks/run_tmi_benchmark.py \
    --seeds    "${SEEDS}" \
    --dim      "${DIM}" \
    --max-fe   "${MAX_FE}" \
    --funcs    ${FUNCS} \
    --variants ${VARIANTS} \
    --workers  "${WORKERS}" \
    --out      "${OUT}" \
    2>&1 | tee "${LOG}"

echo ""
echo "[analyze] Analyzing v2 results..."
python benchmarks/analyze_results.py "${OUT}" --latex \
    2>&1 | tee -a "${LOG}"

echo ""
echo "Done. Results saved to: ${OUT}"
