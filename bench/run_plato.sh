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
#   PLATO_BINARY / PLATO_VANILLA_BINARY   the two binaries.  The vanilla one
#                        must be a separately built UNPATCHED HiGHS of the
#                        same tag, and it has no default: it is named by you
#                        or it is absent.  Neither a PATH search nor a
#                        fallback to the patched build, because which binary
#                        produced a baseline is the one fact the whole
#                        comparison rests on and it should not depend on what
#                        happens to be installed.  A `vanilla` config without
#                        it is an error, not a quietly substituted run.
#   PLATO_EXTRA_OPTIONS  HiGHS options    (default none), e.g.
#                        "mip_heuristic_fpr_effort=1.0 mip_heuristic_fpr_patience=0"
#                        These apply to *every* config, and the vanilla one is
#                        an unpatched binary that has none of the ten options
#                        the patch adds — so pair a patched-only option with a
#                        PLATO_CONFIGS that omits `vanilla`.  The runner probes
#                        every binary it will use for each key and refuses
#                        before the first solve, rather than failing every
#                        instance of the affected arm at solve time; that
#                        covers a typo on the patched arm too.  (Seven of the
#                        seventeen mip_heuristic_* names are upstream's own —
#                        mip_heuristic_effort and the six mip_heuristic_run_*
#                        switches — and are legal on both binaries.)
#   PLATO_DEV_LOG    1 for log_dev_level=3 (default 0; attribution runs only)
#   PLATO_THREADS    pin the solver thread count (default: unset — see below)
#   PLATO_COUNT      run at most N *pending* instances, then stop.  The
#                    count-based chunk: `next` bounds a window in hours,
#                    this bounds it in work, and a campaign that has to give
#                    the machine back uses whichever is easier to predict.
#   PLATO_ANALYZE    0 to skip the end-of-tree analyze_results.py call
#                    (default 1; a presolve-only tree has no dual side, so the
#                    probe stage turns it off and reads the tree with
#                    bench/analyze_presolve_probe.py instead)
#
# Per-run option overrides go through run_benchmark.py's --extra-options;
# a config name is exactly a `mip_heuristic_suite` value.
#
# So the #105 baseline is the default, and the #108 headline is
#
#   PLATO_CONFIGS="fj+fpr+local_mip vanilla" PLATO_SEEDS="0 1 2" \
#     bench/run_plato.sh next 10
#
# The #113 probe is the same script with the generous presolve-only
# environment, which bench/run_presolve_probe.sh sets.
#
# NOTE: Do NOT set PLATO_THREADS — HiGHS uses its default (all cores).
#       Forcing a thread count collapses opportunistic parallelism, and for a
#       tuning run it *moves* the objective's distribution rather than
#       narrowing it (docs/REPRODUCIBILITY.md).  The one legitimate use is a
#       trajectory trace, where a reproducible effort timeline is the point.

set -euo pipefail
shopt -s nullglob

INSTANCES="${PLATO_INSTANCES:-bench/instances_plato.txt}"
# 600 s is the PLATO limit and the headline stages' limit.  The tuning stages
# run at a reduced one; the budget arithmetic below reads this, so a chunk
# stays sized correctly either way.
TIME_LIMIT="${PLATO_TIME_LIMIT:-600}"
OUTPUT="${PLATO_OUTPUT:-bench/results/plato}"
BINARY="${PLATO_BINARY:-./build/bin/highs}"
# Vanilla binary: exactly what PLATO_VANILLA_BINARY says, or nothing.  No PATH
# search and no fallback to $BINARY — a baseline whose binary was discovered
# rather than named is a baseline nobody chose, and the patched build is not a
# vanilla baseline in any configuration (#147).  An empty value is passed
# through to the runner, which reads it as "not given"; `cmd_next` refuses
# outright when the config list asks for `vanilla`.
VANILLA_BINARY="${PLATO_VANILLA_BINARY:-}"
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

wants_vanilla() {
	# Whether the config list asks for the separately built unpatched binary.
	local config
	for config in "${CONFIGS[@]}"; do
		if [ "$config" = "vanilla" ]; then return 0; fi
	done
	return 1
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
		if [ "${PLATO_ANALYZE:-1}" = "1" ]; then
			echo ""
			echo "Run analysis:"
			echo "  python3 bench/analyze_results.py $OUTPUT --configs $(analysis_configs)--time-limit $TIME_LIMIT --baseline"
		fi
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

	# The `vanilla` config is a second *binary*, not a setting on the first
	# one, and there is no longer a fallback to $BINARY: that fallback used to
	# turn a missing system HiGHS into a `vanilla/` tree holding the patched
	# binary at mip_heuristic_suite=off, which is an ablation of our four
	# presolve heuristics rather than a baseline (issue #147).  Fail here,
	# where the message can name the fix, rather than in the runner.
	if wants_vanilla && [ -z "$VANILLA_BINARY" ]; then
		echo "ERROR: config 'vanilla' needs a separately built UNPATCHED HiGHS," >&2
		echo "       and PLATO_VANILLA_BINARY does not name one.  There is no" >&2
		echo "       PATH search: the baseline binary is always named, never" >&2
		echo "       discovered." >&2
		echo "Build one from the same tag as cmake/FetchHiGHS.cmake and set:" >&2
		echo "  export PLATO_VANILLA_BINARY=/path/to/unpatched/highs" >&2
		echo "Or drop 'vanilla' from PLATO_CONFIGS — the 'off' config is the" >&2
		echo "ablation with our heuristics disabled, on the patched binary." >&2
		exit 1
	fi

	# Held in an array rather than spelled into the invocation: each of the
	# three is absent by default, and a flag that is only sometimes passed
	# cannot be written inline.
	local extra_args=()
	if [ -n "${PLATO_EXTRA_OPTIONS:-}" ]; then
		# Word-split on purpose: it is a list of key=value pairs.
		# shellcheck disable=SC2206
		local extra_opts=(${PLATO_EXTRA_OPTIONS})
		extra_args+=(--extra-options "${extra_opts[@]}")
	fi
	if [ "${PLATO_DEV_LOG:-0}" = "1" ]; then
		extra_args+=(--dev-log)
	fi
	if [ -n "${PLATO_THREADS:-}" ]; then
		extra_args+=(--threads "$PLATO_THREADS")
	fi
	if [ -n "${PLATO_COUNT:-}" ]; then
		extra_args+=(--count "$PLATO_COUNT")
	fi

	echo "================================================================"
	echo "PLATO benchmark — ${hours}h window (launching for ${budget_secs}s)"
	echo "  Progress before : $(paired_done)/$TOTAL paired"
	echo "  Configs         : ${CONFIGS[*]}  (seeds ${SEEDS[*]})"
	echo "  Vanilla binary  : ${VANILLA_BINARY:-(none — no vanilla config requested)}"
	echo "  Patched binary  : $BINARY"
	echo "  Output          : $OUTPUT"
	echo "  Time limit      : ${TIME_LIMIT}s"
	if [ ${#extra_args[@]} -gt 0 ]; then
		echo "  Extra runner args: ${extra_args[*]}"
	fi
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
		--wall-time-budget "$budget_secs" \
		${extra_args[@]+"${extra_args[@]}"}

	echo ""
	cmd_status

	if [ "$(paired_done)" -ge "$TOTAL" ] && [ "${PLATO_ANALYZE:-1}" = "1" ]; then
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
