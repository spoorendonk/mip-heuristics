#!/usr/bin/env bash
# Presolve-only probe launcher (issue #113).
#
# One deliberately generous configuration over the PLATO list, run
# presolve-only, to answer the two questions the tuning search cannot start
# without: which instances a presolve screen can see at all, and what each
# heuristic's effort trajectory looks like when nothing stops it.
#
# Usage:
#   bench/run_presolve_probe.sh filter    next [hours] | status
#   bench/run_presolve_probe.sh trace     next [hours] | status
#   bench/run_presolve_probe.sh trace-low next [hours] | status
#
# This is bench/run_plato.sh with the probe environment — the chunking,
# resume and progress accounting are that script's, and this file is only the
# configuration.  Chunk boundaries and `--skip-existing` therefore work the
# same way: `next 8` overnight, `status` in the morning, repeat.
#
# ── the generous configuration ───────────────────────────────────────────────
#
#   * every presolve heuristic at effort 1.0, the top of its range;
#   * every stall gate at 0, which means *no gate*, so the run measures what
#     the heuristic does when nothing stops it — that is what makes the
#     trajectory usable to calibrate a threshold;
#   * `mip_heuristic_presolve_only`, so the run exits before the root LP;
#   * a 60 s per-run cap, enforced by the harness as a wall-clock kill rather
#     than by `time_limit` alone: HiGHS checks its clock between work units
#     and an instance that does not return from its own presolve never looks
#     at it.  A truncated log is *evidence* that the instance is not
#     screenable, not a lost run, and the analyser reads it as such.
#
# ── why the probe is four singles plus the chain, not one chained run ────────
#
# `run_sequential` runs FJ → FPR → LocalMIP → Scylla in order, so a wall-clock
# cap truncates the chain's *tail*.  At effort 1.0 with the gates off, FJ's
# budget is enormous, and on a 12-instance pilot the full chain ran on only 5
# of 12 — three of four heuristics never executed on the rest.  Filtering on
# that run encodes "what FeasibilityJump can do in 60 seconds", which is the
# config-dependent filter #113 exists to avoid, and it is biased against
# exactly the case the campaign wants to detect: an instance only a different
# heuristic can crack.  Measured on that pilot, the singles union saw 10/12
# where the chained run saw 7/12, and the three it gained were cracked by
# heuristics the chain never reached.
#
# So the informative set is the union over `(config, seed)` of the four
# singles, and the chained run is kept alongside because it is the only thing
# that measures the chain interaction — and it is what the campaign deploys.
#
# ── the three passes ─────────────────────────────────────────────────────────
#
#   filter     the instance screen.  Two seeds, no developer logging, HiGHS's
#              own thread default — that is the regime the search runs in.
#   trace      the trajectory characterisation.  `log_dev_level=3` for the
#              [HeurSol] trace, and `threads=1` with a fixed seed, the
#              project's reproducible configuration: multi-worker interleaving
#              makes the effort timeline non-reproducible and lets the
#              solution pool confound attribution.  Runs over the informative
#              set the filter pass emits, since a trajectory on an instance no
#              configuration can crack characterises nothing.
#   trace-low  the same trace one decade down in effort.  The per-attempt
#              slice is derived from the total budget
#              (`attempt_cap = max(total / (10 N), 1)`), so a trajectory taken
#              at a large budget does not exactly reproduce a small one; the
#              two passes are compared over the effort range they share and
#              the discrepancy is reported.
#
# Read a finished tree with bench/analyze_presolve_probe.py, never with
# analyze_results.py: a presolve-only run has no dual side, so its gap is
# meaningless (docs/REPRODUCIBILITY.md).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The four presolve heuristics, one config each, plus the chain.  Singles
# first: the union over them is the informative set, and `all` is the
# interaction control.
PROBE_CONFIGS="${PROBE_CONFIGS:-fj fpr local_mip scylla all}"
PROBE_OUTPUT_ROOT="${PROBE_OUTPUT_ROOT:-bench/results/probe}"
# 60 s truncates 6 of 233 instances on HiGHS's own presolve time alone and
# holds the probe's floor at ~18 min/seed; 30 s truncates 4 more to save 3
# minutes, and 300 s buys 4 instances for 15 minutes each (issue #113).
PROBE_TIME_LIMIT="${PROBE_TIME_LIMIT:-60}"
PROBE_SEEDS="${PROBE_SEEDS:-0 1}"
# The trace passes are reproducible runs, so one seed and one worker.
PROBE_TRACE_SEEDS="${PROBE_TRACE_SEEDS:-0}"
PROBE_INFORMATIVE="${PROBE_INFORMATIVE:-$PROBE_OUTPUT_ROOT/informative.txt}"

probe_options() {
	# $1 — the effort every heuristic runs at.
	local effort=$1 heur
	for heur in fj fpr local_mip scylla; do
		printf 'mip_heuristic_%s_effort=%s ' "$heur" "$effort"
		# 0 is *no gate*, not "give up immediately".
		printf 'mip_heuristic_%s_stall=0 ' "$heur"
	done
	printf 'mip_heuristic_presolve_only=true'
}

MODE="${1:-}"
shift || true
case "$MODE" in
filter)
	export PLATO_OUTPUT="$PROBE_OUTPUT_ROOT/filter"
	export PLATO_SEEDS="$PROBE_SEEDS"
	export PLATO_INSTANCES="${PLATO_INSTANCES:-bench/instances_plato.txt}"
	PLATO_EXTRA_OPTIONS="$(probe_options 1.0)"
	export PLATO_EXTRA_OPTIONS
	;;
trace | trace-low)
	effort=1.0
	suffix=e100
	if [ "$MODE" = "trace-low" ]; then
		effort=0.1
		suffix=e010
	fi
	export PLATO_OUTPUT="$PROBE_OUTPUT_ROOT/trace-$suffix"
	export PLATO_SEEDS="$PROBE_TRACE_SEEDS"
	export PLATO_INSTANCES="${PLATO_INSTANCES:-$PROBE_INFORMATIVE}"
	PLATO_EXTRA_OPTIONS="$(probe_options "$effort")"
	export PLATO_EXTRA_OPTIONS
	export PLATO_DEV_LOG=1
	export PLATO_THREADS=1
	if [ ! -f "$PLATO_INSTANCES" ]; then
		echo "ERROR: no instance list at $PLATO_INSTANCES" >&2
		echo "       The trace passes run over the informative set, which the" >&2
		echo "       filter pass produces:" >&2
		echo "         bench/run_presolve_probe.sh filter next 8" >&2
		echo "         python3 bench/analyze_presolve_probe.py $PROBE_OUTPUT_ROOT/filter \\" >&2
		echo "           --informative-output $PROBE_INFORMATIVE" >&2
		exit 1
	fi
	;;
*)
	echo "Usage: bench/run_presolve_probe.sh {filter|trace|trace-low} next [hours] | status" >&2
	exit 1
	;;
esac

export PLATO_CONFIGS="$PROBE_CONFIGS"
export PLATO_TIME_LIMIT="$PROBE_TIME_LIMIT"
# A presolve-only tree has no dual side; analyze_presolve_probe.py reads it.
export PLATO_ANALYZE=0

exec "$HERE/run_plato.sh" "$@"
