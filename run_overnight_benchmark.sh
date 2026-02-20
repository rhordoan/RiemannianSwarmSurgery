#!/bin/bash

# ==============================================================================
# ORC-SHADE A* Paper: Overnight Benchmark Runner
# ==============================================================================
# Designed for M2 Ultra (High Core Count)
# ==============================================================================

# 1. Setup
PROJECT_ROOT=$(pwd)
RESULTS_DIR="${PROJECT_ROOT}/results"
CSV_OUT="${RESULTS_DIR}/orc_shade_cec2022.csv"
SUMMARY_OUT="${RESULTS_DIR}/orc_shade_cec2022.summary.txt"

mkdir -p "$RESULTS_DIR"

# 2. Dependencies check
echo "Checking dependencies..."
python3 -c "import numpy, scipy, opfunu" || { echo "Error: Missing dependencies. Run: pip install numpy scipy opfunu"; exit 1; }

# 3. Execution
# Note: --workers 20 is a safe bet for M2 Ultra, leaving some headroom.
# --ablation runs all tau and explore_frac variants for the paper's sensitivity section.
echo "Starting ORC-SHADE benchmark..."
echo "Output will be saved to: $CSV_OUT"
echo "------------------------------------------------------------------------------"

python3 benchmarks/run_overnight.py \
    --dims 10 20 \
    --seeds 30 \
    --ablation \
    --workers 20 \
    --out "$CSV_OUT" \
    --resume

# 4. Analysis
echo "------------------------------------------------------------------------------"
echo "Benchmark complete. Generating summary and LaTeX tables..."

python3 benchmarks/analyze_overnight.py "$CSV_OUT" --latex --ablation > "$SUMMARY_OUT"

echo "------------------------------------------------------------------------------"
echo "DONE."
echo "Results CSV: $CSV_OUT"
echo "Summary & LaTeX: $SUMMARY_OUT"
echo "To view results now: cat $SUMMARY_OUT"
