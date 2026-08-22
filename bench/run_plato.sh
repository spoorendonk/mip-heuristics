#!/usr/bin/env bash
# PLATO mipfeas benchmark runner.
#
# Usage:
#   bench/run_plato.sh next [hours]    Run within a HOURS window (default 1). Resume safely.
#   bench/run_plato.sh status          Show progress and estimated time remaining.
#
# Results go to bench/results/plato (persistent across sessions).  Instances
# run interleaved (every config per instance) so partial results are always
# paired and comparable.
#
# Example workflow:
#   bench/run_plato.sh next 8     # run overnight
#   bench/run_plato.sh status     # check in the morning
#   bench/run_plato.sh next 8     # run again next night
#   ...until status shows 233/233
#
# The campaign's four stages differ only in what they run, so each is this
# script with a different environment rather than a different launcher
# (issue #109):
#
#   PLATO_CONFIGS    configs to run       (default "vanilla all")
#   PLATO_SEEDS      seeds per config     (default "0")
#   PLATO_INSTANCES  instance list        (default bench/instances_plato.txt)
#   PLATO_OUTPUT     results tree         (default bench/results/plato)
#   PLATO_TIME_LIMIT seconds per solve    (default 600, the PLATO limit)
#   PLATO_BINARY / PLATO_VANILLA_BINARY   the two binaries
#
# A config name may carry an effort suffix (`fpr@e0.0125`), which is how a
# budget ladder is expressed: one config per ladder point, one directory each.
#
# So the #105 baseline is the default, and the #108 headline is
#
#   PLATO_CONFIGS="fj+fpr+local_mip vanilla" PLATO_SEEDS="0 1 2" \
#     bench/run_plato.sh next 10
#
# NOTE: Do NOT set threads= — HiGHS uses its default (all cores).
#       Forcing a thread count collapses opportunistic parallelism.

set -euo pipefail
shopt -s nullglob

INSTANCES="${PLATO_INSTANCES:-bench/instances_plato.txt}"
# 600 s is the PLATO limit and the headline stages' limit.  The tuning stages
# run at a reduced one; the budget arithmetic below reads this, so a chunk
# stays sized correctly either way.
TIME_LIMIT="${PLATO_TIME_LIMIT:-600}"
OUTPUT="${PLATO_OUTPUT:-bench/results/plato}"
BINARY="${PLATO_BINARY:-./build/bin/highs}"
# Vanilla binary: prefer system HiGHS (unpatched), fall back to patched build.
# Override with PLATO_VANILLA_BINARY env var if needed.
VANILLA_BINARY="${PLATO_VANILLA_BINARY:-$(which highs 2>/dev/null || echo "$BINARY")}"
# Word-split on purpose: both are lists.
# shellcheck disable=SC2206
CONFIGS=(${PLATO_CONFIGS:-vanilla all})
# shellcheck disable=SC2206
SEEDS=(${PLATO_SEEDS:-0})

# Derived from the list rather than hardcoded: the script is pointed at the
# tuning subset as often as at the full 233, and a stale constant would report
# a tuning run as 11% done forever.
TOTAL=$(grep -c '^[[:space:]]*[^[:space:]#]' "$INSTANCES" 2>/dev/null || true)
TOTAL=${TOTAL:-0}
if [ "$TOTAL" -eq 0 ]; then
	# Otherwise every count trivially reaches TOTAL and the run reports
	# COMPLETE having solved nothing.
	echo "ERROR: $INSTANCES names no instances (missing, empty, or all comments)" >&2
	exit 1
fi

# ── helpers ──────────────────────────────────────────────────────────────────

count_done() {
	# Instances finished for a config, counted only where *every* seed has a
	# log: a resume is per (config, instance, seed), so an instance with one
	# of three seeds done is not done.
	local config=$1 seed dir f
	{
		for seed in "${SEEDS[@]}"; do
			dir="$OUTPUT/$config/seed$seed"
			for f in "$dir"/*.log; do
				if [ -s "$f" ]; then
					echo "${f##*/}"
				fi
			done
		done
	} | sort | uniq -c | awk -v n="${#SEEDS[@]}" '$1 == n' | wc -l
}

paired_done() {
	# The campaign is only as complete as its least complete config.
	local config least=$TOTAL n
	for config in "${CONFIGS[@]}"; do
		n=$(count_done "$config")
		if [ "$n" -lt "$least" ]; then least=$n; fi
	done
	echo "$least"
}

estimate_hours() {
	local remaining=$1
	# Runs are sequential, so each remaining instance costs one time limit per
	# config per seed.
	echo $((remaining * TIME_LIMIT * ${#CONFIGS[@]} * ${#SEEDS[@]} / 3600))
}

analysis_configs() {
	# Baseline last: analyze_results.py reports the SGM ratio as first/second
	# and names the winner, which reads as "patched vs vanilla" only in that
	# order.
	local config
	for config in "${CONFIGS[@]}"; do
		case "$config" in vanilla | off) ;; *) printf '%s ' "$config" ;; esac
	done
	for config in "${CONFIGS[@]}"; do
		case "$config" in vanilla | off) printf '%s ' "$config" ;; esac
	done
}

# ── subcommands ───────────────────────────────────────────────────────────────

cmd_status() {
	local config paired remaining
	paired=$(paired_done)
	remaining=$((TOTAL - paired)) || true

	echo "PLATO mipfeas progress  ($OUTPUT)"
	echo "  instances : $INSTANCES ($TOTAL)"
	echo "  seeds     : ${SEEDS[*]}"
	for config in "${CONFIGS[@]}"; do
		printf '  %-10s: %s / %s\n' "$config" "$(count_done "$config")" "$TOTAL"
	done
	echo "  paired  : $paired / $TOTAL  (every config, every seed)"
	if [ "$paired" -ge "$TOTAL" ]; then
		echo "  STATUS  : COMPLETE"
		echo ""
		echo "Run analysis:"
		echo "  python3 bench/analyze_results.py $OUTPUT --configs $(analysis_configs)--time-limit $TIME_LIMIT --baseline"
	else
		local est
		est=$(estimate_hours "$remaining") || true
		echo "  remaining : ~$remaining instances  (~${est}h at ${TIME_LIMIT}s × ${#CONFIGS[@]} configs × ${#SEEDS[@]} seeds)"
	fi
}

cmd_next() {
	local hours=${1:-1}
	# --wall-time-budget stops *launching* new instances; the one already
	# running still gets its full limit, so a chunk can overrun its budget by
	# up to TIME_LIMIT.  Size it as `window - time_limit` (the campaign rule in
	# issue #109) so an overnight window is actually free by morning, rather
	# than 10 minutes into the working day.
	local budget_secs=$((hours * 3600 - TIME_LIMIT)) || true
	if [ "$budget_secs" -le 0 ]; then
		echo "ERROR: a ${hours}h window is not longer than the ${TIME_LIMIT}s time limit" >&2
		echo "       (the budget has to leave room for one instance to finish)" >&2
		exit 1
	fi

	if [ ! -f "$BINARY" ]; then
		echo "ERROR: binary not found: $BINARY" >&2
		echo "Build: cmake -B build && cmake --build build -j\$(nproc)" >&2
		echo "Or set: export PLATO_BINARY=/path/to/highs" >&2
		exit 1
	fi

	echo "================================================================"
	echo "PLATO benchmark — ${hours}h window (launching for ${budget_secs}s)"
	echo "  Progress before : $(paired_done)/$TOTAL paired"
	echo "  Configs         : ${CONFIGS[*]}  (seeds ${SEEDS[*]})"
	echo "  Vanilla binary  : $VANILLA_BINARY"
	echo "  Patched binary  : $BINARY"
	echo "  Output          : $OUTPUT"
	echo "  (Skipping already-completed instances)"
	echo "================================================================"

	python3 bench/run_benchmark.py \
		--instances "$INSTANCES" \
		--binary "$BINARY" \
		--vanilla-binary "$VANILLA_BINARY" \
		--time-limit "$TIME_LIMIT" \
		--output "$OUTPUT" \
		--configs "${CONFIGS[@]}" \
		--seeds "${SEEDS[@]}" \
		--skip-existing \
		--interleave \
		--wall-time-budget "$budget_secs"

	echo ""
	cmd_status

	if [ "$(paired_done)" -ge "$TOTAL" ]; then
		echo ""
		echo "All instances complete — running analysis..."
		# shellcheck disable=SC2046
		python3 bench/analyze_results.py \
			"$OUTPUT" \
			--configs $(analysis_configs) \
			--time-limit "$TIME_LIMIT" \
			--baseline
	fi
}

# ── dispatch ──────────────────────────────────────────────────────────────────

CMD="${1:-status}"
shift || true

case "$CMD" in
next) cmd_next "$@" ;;
status) cmd_status ;;
*)
	echo "Usage: bench/run_plato.sh next [hours] | status" >&2
	exit 1
	;;
esac
