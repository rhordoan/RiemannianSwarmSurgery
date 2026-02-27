#!/usr/bin/env bash
# run_benchmark.sh -- Full CEC 2017 benchmark for CARS
#
# Usage:
#   bash run_benchmark.sh              # Auto-detect cores
#   bash run_benchmark.sh 190          # Use 190 workers
#   bash run_benchmark.sh 190 quick    # Quick 3-run test

set -euo pipefail

WORKERS="${1:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"
MODE="${2:-full}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$SCRIPT_DIR/results/cec2017_${TIMESTAMP}"

echo "=============================================="
echo "  CARS CEC 2017 Benchmark"
echo "  Workers: $WORKERS"
echo "  Mode:    $MODE"
echo "  Output:  $OUTPUT_DIR"
echo "=============================================="

# --- Virtual environment setup ---
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "Installing dependencies..."
pip install --quiet --upgrade pip setuptools
pip install --quiet numpy scipy "opfunu>=1.0"

# --- Run benchmark ---
echo ""
echo "Starting benchmark at $(date)..."

if [ "$MODE" = "quick" ]; then
    python "$SCRIPT_DIR/benchmarks/run_cec2017.py" \
        --func 21 22 23 24 25 26 27 28 29 30 \
        --dims 10 30 \
        --runs 3 \
        --workers "$WORKERS" \
        --output "$OUTPUT_DIR" \
        2>&1 | tee "$OUTPUT_DIR.log"
else
    python "$SCRIPT_DIR/benchmarks/run_cec2017.py" \
        --full \
        --workers "$WORKERS" \
        --output "$OUTPUT_DIR" \
        2>&1 | tee "$OUTPUT_DIR.log"
fi

echo ""
echo "Benchmark completed at $(date)"
echo "Results: $OUTPUT_DIR"
echo "Summary: $OUTPUT_DIR/summary.csv"
echo "Stats:   $OUTPUT_DIR/statistics.txt"
echo "Log:     $OUTPUT_DIR.log"
