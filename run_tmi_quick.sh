#!/usr/bin/env bash
# =============================================================================
# TMI Quick Smoke Test — run this first to verify everything works on the Mac.
#
# Runs 3 seeds × 3 functions (F4, F8, F12) × 4 variants at 30k FEs.
# Should complete in ~5-10 minutes depending on the machine.
#
# Usage:
#   bash run_tmi_quick.sh
# =============================================================================

set -euo pipefail

VENV_DIR=".venv"
WORKERS=$(python3 -c "import os; print(max(1, os.cpu_count() - 1))" 2>/dev/null || echo 4)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUT="results/tmi_quick_${TIMESTAMP}.csv"

echo "=== TMI Quick Smoke Test ==="
echo " Workers: ${WORKERS}"
echo "============================"

# venv
if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

mkdir -p results

python benchmarks/run_tmi_benchmark.py \
    --seeds    3 \
    --dim      10 \
    --max-fe   30000 \
    --funcs    4 8 12 \
    --variants A B C D E \
    --workers  "${WORKERS}" \
    --out      "${OUT}"

echo ""
echo "=== Analysis ==="
python benchmarks/analyze_results.py "${OUT}"

echo ""
echo "Quick test done. Results: ${OUT}"
