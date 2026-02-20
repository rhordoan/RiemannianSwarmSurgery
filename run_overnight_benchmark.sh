#!/bin/bash

# ==============================================================================
# ORC-SHADE A* Paper: FULL Overnight Pipeline
# ==============================================================================
# 1. Creates/Updates Python Virtual Environment
# 2. Installs SOTA Optimization Dependencies
# 3. Runs 30-seed CEC 2022 Benchmark (D=10, D=20)
# 4. Generates LaTeX Tables & Analysis
# ==============================================================================

set -e # Exit on error

PROJECT_ROOT=$(pwd)
VENV_DIR="${PROJECT_ROOT}/venv_orc"
RESULTS_DIR="${PROJECT_ROOT}/results"
CSV_OUT="${RESULTS_DIR}/orc_shade_cec2022.csv"
SUMMARY_OUT="${RESULTS_DIR}/orc_shade_cec2022.summary.txt"

mkdir -p "$RESULTS_DIR"

echo "=============================================================================="
echo "1. ENVIRONMENT SETUP"
echo "=============================================================================="

# Detect Python
if command -v python3 &>/dev/null; then
    PYTHON_BIN="python3"
elif command -v python &>/dev/null; then
    PYTHON_BIN="python"
else
    echo "Error: Python not found. Please install Python 3.10+."
    exit 1
fi

echo "Using: $($PYTHON_BIN --version)"

# Create VENV if missing
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    $PYTHON_BIN -m venv "$VENV_DIR"
fi

# Activate VENV
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Update tools
echo "Updating pip and installing dependencies..."
pip install --upgrade pip setuptools wheel
pip install numpy scipy opfunu matplotlib

echo "=============================================================================="
echo "2. EXECUTION: ORC-SHADE SOTA BENCHMARK"
echo "=============================================================================="
echo "Machine: M2 Ultra Detection"
echo "Workers: 20 (Optimized for performance/thermal headroom)"
echo "Mode: Full Ablation (Sensitivity Analysis Enabled)"
echo "Output: $CSV_OUT"
echo "------------------------------------------------------------------------------"

# Run the benchmark
# --resume allows you to stop and restart without losing data
python benchmarks/run_overnight.py \
    --dims 10 20 \
    --seeds 30 \
    --ablation \
    --workers 20 \
    --out "$CSV_OUT" \
    --resume

echo "=============================================================================="
echo "3. POST-PROCESSING: ANALYTICS & PAPER TABLES"
echo "=============================================================================="

python benchmarks/analyze_overnight.py "$CSV_OUT" --latex --ablation > "$SUMMARY_OUT"

echo "DONE."
echo "------------------------------------------------------------------------------"
echo "Results CSV     : $CSV_OUT"
echo "LaTeX & Summary : $SUMMARY_OUT"
echo "------------------------------------------------------------------------------"
echo "To view summary now: cat $SUMMARY_OUT"
echo "=============================================================================="
