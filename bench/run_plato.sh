#!/usr/bin/env bash
# PLATO mipfeas benchmark runner.
#
# Usage:
#   bench/run_plato.sh next [hours]    Run for up to HOURS (default 1). Resume safely.
#   bench/run_plato.sh status          Show progress and estimated time remaining.
#
# All results go to bench/results/plato (persistent across sessions).
# Instances run interleaved (vanilla + patched per instance) so partial
# results are always paired and comparable.
#
# Example workflow:
#   bench/run_plato.sh next 8     # run overnight
#   bench/run_plato.sh status     # check in the morning
#   bench/run_plato.sh next 8     # run again next night
#   ...until status shows 233/233
#
# NOTE: Do NOT set threads= — HiGHS uses its default (all cores).
#       Forcing a thread count collapses opportunistic parallelism.

set -euo pipefail

INSTANCES="bench/instances_plato.txt"
TOTAL=233
TIME_LIMIT=600
OUTPUT="bench/results/plato"
BINARY="${PLATO_BINARY:-./build/bin/highs}"
# Vanilla binary: prefer system HiGHS (unpatched), fall back to patched build.
# Override with PLATO_VANILLA_BINARY env var if needed.
VANILLA_BINARY="${PLATO_VANILLA_BINARY:-$(which highs 2>/dev/null || echo "$BINARY")}"

# ── helpers ──────────────────────────────────────────────────────────────────

count_done() {
    local config=$1
    local dir="$OUTPUT/$config/seed0"
    [ -d "$dir" ] || { echo 0; return 0; }
    find "$dir" -name "*.log" -size +0c 2>/dev/null | wc -l
}

estimate_hours() {
    local remaining=$1
    # Each instance: TIME_LIMIT seconds × 2 configs, but interleaved so wall
    # time per "pair" is 2×TIME_LIMIT sequential.
    echo $(( remaining * TIME_LIMIT * 2 / 3600 ))
}

# ── subcommands ───────────────────────────────────────────────────────────────

cmd_status() {
    local v p paired remaining
    v=$(count_done vanilla)
    p=$(count_done patched)
    paired=$(( v < p ? v : p )) || true
    remaining=$(( TOTAL - paired )) || true

    echo "PLATO mipfeas progress  ($OUTPUT)"
    echo "  vanilla : $v / $TOTAL"
    echo "  patched : $p / $TOTAL"
    echo "  paired  : $paired / $TOTAL  (both configs done)"
    if [ "$paired" -ge "$TOTAL" ]; then
        echo "  STATUS  : COMPLETE"
        echo ""
        echo "Run analysis:"
        echo "  python bench/analyze_results.py $OUTPUT --configs patched vanilla --time-limit $TIME_LIMIT --baseline"
    else
        local est
        est=$(estimate_hours "$remaining") || true
        echo "  remaining : ~$remaining instances  (~${est}h at 600s×2 sequential)"
    fi
}

cmd_next() {
    local hours=${1:-1}
    local budget_secs=$(( hours * 3600 )) || true

    if [ ! -f "$BINARY" ]; then
        echo "ERROR: binary not found: $BINARY" >&2
        echo "Build: cmake -B build && cmake --build build -j\$(nproc)" >&2
        echo "Or set: export PLATO_BINARY=/path/to/highs" >&2
        exit 1
    fi

    local v p
    v=$(count_done vanilla)
    p=$(count_done patched)
    echo "================================================================"
    echo "PLATO benchmark — running for up to ${hours}h"
    echo "  Progress before : vanilla $v/$TOTAL, patched $p/$TOTAL"
    echo "  Vanilla binary  : $VANILLA_BINARY"
    echo "  Patched binary  : $BINARY"
    echo "  Output          : $OUTPUT"
    echo "  (Skipping already-completed instances)"
    echo "================================================================"

    python bench/run_benchmark.py \
        --instances "$INSTANCES" \
        --binary "$BINARY" \
        --vanilla-binary "$VANILLA_BINARY" \
        --time-limit "$TIME_LIMIT" \
        --output "$OUTPUT" \
        --configs vanilla patched \
        --skip-existing \
        --interleave \
        --wall-time-budget "$budget_secs"

    echo ""
    cmd_status

    local paired
    paired=$(( $(count_done vanilla) < $(count_done patched) ? $(count_done vanilla) : $(count_done patched) ))
    if [ "$paired" -ge "$TOTAL" ]; then
        echo ""
        echo "All instances complete — running analysis..."
        python bench/analyze_results.py \
            "$OUTPUT" \
            --configs patched vanilla \
            --time-limit "$TIME_LIMIT" \
            --baseline
    fi
}

# ── dispatch ──────────────────────────────────────────────────────────────────

CMD="${1:-status}"
shift || true

case "$CMD" in
    next)   cmd_next "$@" ;;
    status) cmd_status ;;
    *)
        echo "Usage: bench/run_plato.sh next [hours] | status" >&2
        exit 1
        ;;
esac
