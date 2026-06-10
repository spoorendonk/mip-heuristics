#!/usr/bin/env bash
# Run the PLATO mipfeas comparison: vanilla HiGHS vs patched HiGHS with
# preset=all_opp (FJ+FPR+LocalMIP, opportunistic) on the 233-instance
# MIPLIB2017 feasibility set.
#
# Reference: https://plato.asu.edu/ftp/mipfeas.html
# Metric: primal integral (area under primal-gap curve, 600s window, SGM shift=0.001)
#
# CHUNKED USAGE (recommended — run one chunk per night):
#   bench/run_plato.sh <chunk> [total_chunks] [output_dir] [binary]
#
#   chunk         Which chunk to run, 1-based (e.g. 1, 2, ... 10)
#   total_chunks  How many chunks to split into (default: 10 → ~24 instances,
#                 ~8 hours per chunk at 600s × 2 configs sequential)
#   output_dir    Shared output dir across all chunks (default: bench/results/plato)
#   binary        Path to HiGHS binary (default: ./build/bin/highs)
#
#   Example — 10 overnight runs writing to the same output dir:
#     bench/run_plato.sh 1 10   # night 1
#     bench/run_plato.sh 2 10   # night 2
#     ...
#     bench/run_plato.sh 10 10  # night 10
#
#   After all chunks complete, run analysis:
#     python bench/analyze_results.py bench/results/plato --configs patched vanilla \
#       --time-limit 600 --baseline
#
# FULL RUN (single shot — ~40-70 hours):
#   bench/run_plato.sh all [output_dir] [binary]
#
# NOTE: Do NOT pass --threads — forces thread count and collapses opportunistic
#       parallelism to one worker per epoch (see CLAUDE.md benchmarking note).

set -euo pipefail

INSTANCES="bench/instances_plato.txt"
TOTAL_INSTANCES=233
TIME_LIMIT=600

# --- Parse arguments ---
CHUNK_ARG="${1:-all}"

if [ "$CHUNK_ARG" = "all" ]; then
    CHUNK=1
    TOTAL_CHUNKS=1
    OUTPUT_DIR="${2:-bench/results/plato}"
    BINARY="${3:-./build/bin/highs}"
    START=0
    COUNT=$TOTAL_INSTANCES
else
    CHUNK="$CHUNK_ARG"
    TOTAL_CHUNKS="${2:-10}"
    OUTPUT_DIR="${3:-bench/results/plato}"
    BINARY="${4:-./build/bin/highs}"

    # Compute instance slice for this chunk
    CHUNK_SIZE=$(( (TOTAL_INSTANCES + TOTAL_CHUNKS - 1) / TOTAL_CHUNKS ))
    START=$(( (CHUNK - 1) * CHUNK_SIZE ))
    COUNT=$(( CHUNK_SIZE < TOTAL_INSTANCES - START ? CHUNK_SIZE : TOTAL_INSTANCES - START ))
    if [ "$COUNT" -le 0 ]; then
        echo "Chunk $CHUNK of $TOTAL_CHUNKS: nothing to do (all instances covered by earlier chunks)."
        exit 0
    fi
fi

if [ ! -f "$BINARY" ]; then
    echo "ERROR: HiGHS binary not found: $BINARY" >&2
    echo "Build first: cmake -B build && cmake --build build -j\$(nproc)" >&2
    exit 1
fi
if [ ! -f "$INSTANCES" ]; then
    echo "ERROR: Instance list not found: $INSTANCES" >&2
    exit 1
fi

END=$(( START + COUNT ))
echo "==================================================================="
if [ "$CHUNK_ARG" = "all" ]; then
    echo "PLATO benchmark — full run ($TOTAL_INSTANCES instances, ${TIME_LIMIT}s each)"
else
    echo "PLATO benchmark — chunk $CHUNK of $TOTAL_CHUNKS"
    echo "  Instances $((START+1))–$END of $TOTAL_INSTANCES ($COUNT instances)"
    echo "  Estimated time: ~$(( COUNT * TIME_LIMIT * 2 / 3600 ))–$(( COUNT * TIME_LIMIT * 2 / 3600 + 2 )) hours"
fi
echo "  Output : $OUTPUT_DIR"
echo "  Binary : $BINARY"
echo "  Skipping already-completed runs (safe to resume interrupted chunks)"
echo "==================================================================="

# --- Step 1: vanilla HiGHS ---
echo ""
echo "Step 1/2: vanilla HiGHS (preset=off, effort=0.05)"
python bench/run_benchmark.py \
    --instances "$INSTANCES" \
    --binary "$BINARY" \
    --time-limit "$TIME_LIMIT" \
    --output "$OUTPUT_DIR" \
    --configs vanilla \
    --start "$START" \
    --count "$COUNT" \
    --skip-existing

# --- Step 2: patched HiGHS ---
echo ""
echo "Step 2/2: patched HiGHS (preset=all_opp: FJ+FPR+LocalMIP, opportunistic)"
python bench/run_benchmark.py \
    --instances "$INSTANCES" \
    --binary "$BINARY" \
    --time-limit "$TIME_LIMIT" \
    --output "$OUTPUT_DIR" \
    --configs patched \
    --start "$START" \
    --count "$COUNT" \
    --skip-existing

# --- Analysis (only runs after all chunks are done) ---
DONE_VANILLA=$(find "$OUTPUT_DIR/vanilla/seed0" -name "*.log" 2>/dev/null | wc -l)
DONE_PATCHED=$(find "$OUTPUT_DIR/patched/seed0" -name "*.log" 2>/dev/null | wc -l)
echo ""
echo "Progress: vanilla $DONE_VANILLA/$TOTAL_INSTANCES, patched $DONE_PATCHED/$TOTAL_INSTANCES"

if [ "$DONE_VANILLA" -ge "$TOTAL_INSTANCES" ] && [ "$DONE_PATCHED" -ge "$TOTAL_INSTANCES" ]; then
    echo ""
    echo "==================================================================="
    echo "All chunks complete — running full analysis"
    echo "==================================================================="
    python bench/analyze_results.py \
        "$OUTPUT_DIR" \
        --configs patched vanilla \
        --time-limit "$TIME_LIMIT" \
        --baseline
else
    REMAINING=$(( TOTAL_INSTANCES - (DONE_VANILLA < DONE_PATCHED ? DONE_VANILLA : DONE_PATCHED) ))
    echo "Run remaining chunks to complete. ~$REMAINING instances left."
    echo "When all done: python bench/analyze_results.py $OUTPUT_DIR --configs patched vanilla --time-limit $TIME_LIMIT --baseline"
fi
