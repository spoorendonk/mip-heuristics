#!/usr/bin/env bash
# Run the PLATO mipfeas comparison: vanilla HiGHS vs patched HiGHS with
# preset=all_opp (FJ+FPR+LocalMIP, opportunistic) on the 233-instance
# MIPLIB2017 feasibility set.
#
# Reference: https://plato.asu.edu/ftp/mipfeas.html
# Metric: primal integral (area under primal-gap curve, 600s window, SGM shift=0.001)
#
# Usage:
#   bench/run_plato.sh [OUTPUT_DIR] [HIGHS_BINARY]
#
# Arguments:
#   OUTPUT_DIR    Where to write logs (default: bench/results/plato_<timestamp>)
#   HIGHS_BINARY  Path to HiGHS binary (default: ./build/bin/highs)
#
# Prerequisites:
#   1. Build the project:         cmake -B build && cmake --build build -j$(nproc)
#   2. Download MIPLIB instances: bash bench/download_miplib.sh
#   3. (~30 CPU-hours) This script runs 233 instances × 2 configs × 600s each.
#      Run on a machine with enough cores; HiGHS uses its default thread count.
#
# NOTE: Do NOT pass --threads or set threads= in options — forcing thread count
# collapses opportunistic parallelism to one worker per epoch (see CLAUDE.md).

set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${1:-bench/results/plato_${TIMESTAMP}}"
BINARY="${2:-./build/bin/highs}"
INSTANCES="bench/instances_plato.txt"
TIME_LIMIT=600

if [ ! -f "$BINARY" ]; then
    echo "ERROR: HiGHS binary not found: $BINARY" >&2
    echo "Build first: cmake -B build && cmake --build build -j\$(nproc)" >&2
    exit 1
fi

if [ ! -f "$INSTANCES" ]; then
    echo "ERROR: Instance list not found: $INSTANCES" >&2
    exit 1
fi

echo "==================================================================="
echo "PLATO mipfeas benchmark (233 instances, ${TIME_LIMIT}s per instance)"
echo "Output: $OUTPUT_DIR"
echo "Binary: $BINARY"
echo "==================================================================="

# --- Step 1: vanilla HiGHS (no custom heuristics, upstream effort budget) ---
echo ""
echo "Step 1/2: vanilla HiGHS (mip_heuristic_run_fpr=false etc.)"
python bench/run_benchmark.py \
    --instances "$INSTANCES" \
    --binary "$BINARY" \
    --time-limit "$TIME_LIMIT" \
    --output-dir "$OUTPUT_DIR" \
    --configs vanilla

# --- Step 2: patched HiGHS (all_opp: FJ + FPR + LocalMIP, opportunistic) ---
echo ""
echo "Step 2/2: patched HiGHS (preset=all_opp: FJ+FPR+LocalMIP, opportunistic, no portfolio, no Scylla)"
python bench/run_benchmark.py \
    --instances "$INSTANCES" \
    --binary "$BINARY" \
    --time-limit "$TIME_LIMIT" \
    --output-dir "$OUTPUT_DIR" \
    --configs patched

# --- Step 3: compare ---
echo ""
echo "==================================================================="
echo "Analysis"
echo "==================================================================="
python bench/analyze_results.py \
    "$OUTPUT_DIR" \
    --configs patched vanilla \
    --time-limit "$TIME_LIMIT" \
    --baseline

echo ""
echo "Done. Full logs in $OUTPUT_DIR/"
echo "Re-run analysis: python bench/analyze_results.py $OUTPUT_DIR --configs patched vanilla --time-limit $TIME_LIMIT --baseline"
