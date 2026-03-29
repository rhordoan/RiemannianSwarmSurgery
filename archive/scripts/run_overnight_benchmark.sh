#!/bin/bash

# ==============================================================================
# ORC-SHADE A* Paper: FULL Overnight Pipeline (macOS Fix Edition)
# ==============================================================================
# 1. Creates/Updates Python Virtual Environment
# 2. Forces installation of setuptools<70.0 (pkg_resources fix)
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

# Create VENV if missing
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    $PYTHON_BIN -m venv "$VENV_DIR"
fi

# Activate VENV
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# CRITICAL FIX: Explicitly pin setuptools to <70.0.0 because opfunu relies on pkg_resources
# which was entirely removed in setuptools 70+
echo "Installing/Updating setuptools and core dependencies..."
pip install --upgrade pip
pip install "setuptools<70.0.0" wheel
pip install numpy scipy opfunu matplotlib

# Verify pkg_resources is available
python -c "import pkg_resources; print('Dependency check: pkg_resources OK')"

echo "=============================================================================="
echo "2. EXECUTION: ORC-SHADE SOTA BENCHMARK"
echo "=============================================================================="
echo "Machine: M2 Ultra / macOS detected"
echo "Mode: Full Ablation (Sensitivity Analysis Enabled)"
echo "------------------------------------------------------------------------------"

# Run the benchmark using the venv python explicitly to ensure workers use it
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
echo "Results CSV     : $CSV_OUT"
echo "LaTeX & Summary : $SUMMARY_OUT"
echo "=============================================================================="
