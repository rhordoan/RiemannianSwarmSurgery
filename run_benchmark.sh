#!/bin/bash
# ==============================================================================
# ORC-SHADE v2: Full Overnight Benchmark
# ==============================================================================
# Single command to run the complete benchmark pipeline:
#   1. Creates Python virtual environment
#   2. Installs all dependencies
#   3. Runs 30-seed CEC 2022 benchmark (D=10 + D=20, all 12 functions)
#   4. Generates summary statistics with Wilcoxon tests
#
# Usage:
#   chmod +x run_benchmark.sh
#   ./run_benchmark.sh              # full overnight run
#   ./run_benchmark.sh --quick      # 3-seed quick test (~15 min)
#
# For tmux:
#   tmux new -s orc
#   ./run_benchmark.sh
#   # Ctrl-B D to detach, tmux attach -t orc to re-attach
# ==============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${PROJECT_ROOT}/venv_orc"
RESULTS_DIR="${PROJECT_ROOT}/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CSV_OUT="${RESULTS_DIR}/orc_shade_v2_${TIMESTAMP}.csv"
SUMMARY_OUT="${RESULTS_DIR}/orc_shade_v2_${TIMESTAMP}.summary.txt"

# Parse args
QUICK=0
for arg in "$@"; do
    case "$arg" in
        --quick) QUICK=1 ;;
    esac
done

mkdir -p "$RESULTS_DIR"

echo "=============================================================================="
echo " ORC-SHADE v2 Benchmark Pipeline"
echo " Started: $(date)"
echo " Project: ${PROJECT_ROOT}"
echo "=============================================================================="

# --------------------------------------------------------------------------
# 1. Environment setup
# --------------------------------------------------------------------------
echo ""
echo "[1/4] Setting up Python environment..."

if command -v python3 &>/dev/null; then
    PYTHON_BIN="python3"
elif command -v python &>/dev/null; then
    PYTHON_BIN="python"
else
    echo "ERROR: Python not found. Install Python 3.10+."
    exit 1
fi

echo "  Python: $($PYTHON_BIN --version 2>&1)"

if [ ! -d "$VENV_DIR" ]; then
    echo "  Creating virtual environment..."
    $PYTHON_BIN -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "  Installing dependencies..."
pip install --upgrade pip -q
pip install "setuptools<70.0.0" wheel -q
pip install numpy scipy opfunu matplotlib -q

python -c "import pkg_resources; import opfunu; import scipy; print('  Dependencies OK')"

# --------------------------------------------------------------------------
# 2. Quick smoke test
# --------------------------------------------------------------------------
echo ""
echo "[2/4] Smoke test (single seed, F1+F7, 30k FE)..."

python -c "
import sys, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
import numpy as np, opfunu
from benchmarks.nlshade import NLSHADE
from src.orc_shade import ORCSHADE

def prob(fn, d):
    cls = getattr(opfunu.cec_based, f'F{fn}2022')
    inner = cls(ndim=d)
    bias = inner.f_bias
    class P:
        bounds = [-100.0, 100.0]
        def evaluate(self, x): return max(0.0, float(inner.evaluate(x)) - bias)
    return P()

np.random.seed(0)
o = ORCSHADE(prob(1, 10), dim=10, max_fe=30000); o.run()
print(f'  F1 ORC: {o.best_fitness:.2e} (should be ~0)')
np.random.seed(0)
o = ORCSHADE(prob(7, 10), dim=10, max_fe=30000); o.run()
print(f'  F7 ORC: {o.best_fitness:.2e} (should be low)')
print('  Smoke test PASSED')
"

# --------------------------------------------------------------------------
# 3. Main benchmark
# --------------------------------------------------------------------------
echo ""
if [ "$QUICK" -eq 1 ]; then
    echo "[3/4] Running QUICK benchmark (3 seeds, D=10 only)..."
    SEEDS=3
    DIMS="10"
    BUDGET_D10=50000
    EXTRA_ARGS="--seeds $SEEDS --dims $DIMS --budget_d10 $BUDGET_D10"
else
    echo "[3/4] Running FULL benchmark (30 seeds, D=10 + D=20)..."
    SEEDS=30
    DIMS="10 20"
    EXTRA_ARGS="--seeds $SEEDS --dims $DIMS --ablation"
fi

# Detect CPU count for workers
NCPU=$(python -c "import os; print(max(1, os.cpu_count() - 2))")
echo "  Workers: $NCPU"
echo "  Seeds:   $SEEDS"
echo "  Dims:    $DIMS"
echo "  Output:  $CSV_OUT"
echo ""

python benchmarks/run_overnight.py \
    $EXTRA_ARGS \
    --workers "$NCPU" \
    --out "$CSV_OUT" \
    --resume

# --------------------------------------------------------------------------
# 4. Analysis
# --------------------------------------------------------------------------
echo ""
echo "[4/4] Generating analysis and summary..."

if [ -f "benchmarks/analyze_overnight.py" ]; then
    if [ "$QUICK" -eq 1 ]; then
        python benchmarks/analyze_overnight.py "$CSV_OUT" > "$SUMMARY_OUT" 2>&1 || true
    else
        python benchmarks/analyze_overnight.py "$CSV_OUT" --latex --ablation > "$SUMMARY_OUT" 2>&1 || true
    fi
    echo "  Summary saved to: $SUMMARY_OUT"
else
    echo "  (analyze_overnight.py not found, skipping)"
fi

echo ""
echo "=============================================================================="
echo " DONE"
echo " Finished: $(date)"
echo " Results:  $CSV_OUT"
echo " Summary:  $SUMMARY_OUT"
echo "=============================================================================="
