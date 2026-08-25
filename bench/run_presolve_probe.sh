#!/usr/bin/env bash
# Presolve-only calibration probe (issue #113).
#
# Usage:
#   bench/run_presolve_probe.sh preprobe next [hours]   the experiment
#   bench/run_presolve_probe.sh budget   next [hours]   bounded-budget control
#   bench/run_presolve_probe.sh serial   next [hours]   single-worker control
#   bench/run_presolve_probe.sh <mode>   status
#
#   PROBE_COUNT=20 bench/run_presolve_probe.sh preprobe next 4
#     ...stops after 20 *pending* instances even if the window is longer.
#     Use it to borrow the machine for a bounded amount of work rather than
#     for a whole tree; every chunk resumes exactly where the last stopped.
#
# This is bench/run_plato.sh with the probe environment, so the chunking,
# resume and progress accounting are that script's and a stage is an
# environment rather than a launcher (#109).
#
# ── the experiment ───────────────────────────────────────────────────────────
#
# One question: **what does each heuristic do with 30 seconds of presolve,
# alone, when nothing but the clock stops it?**
#
#   * effort `$PROBE_EFFORT` (1e4, the option's ceiling since #113) — the
#     budget is then `8.2e8` effort units per matrix nonzero, which no run
#     inside the cap can reach;
#   * every stall gate at 0, which means *no gate*;
#   * `mip_heuristic_presolve_only`, so the run exits before the root LP;
#   * a 30 s cap, enforced by the harness as a wall-clock kill as well as by
#     `time_limit`, since HiGHS checks its clock between work units and an
#     instance that does not return from its own presolve never looks at it;
#   * `log_dev_level=3`, so every run carries the [HeurSol] trace.
#
# `WorkerBudgetState` retires a worker only when it is `exhausted()` (total
# budget) or `stale()`.  Disable both and no worker ever retires, so the
# wall clock is the single stopping rule — the same one for all four
# heuristics on every instance.  That is the whole point of the design: at
# any binding budget the trace measures the setting we are trying to derive
# from it, and each heuristic's budget binds at a different model size (FJ
# exhausts its effort-1.0 budget in 7.8 s on a 41 k-nonzero model while
# Scylla is already clock-bound there).
#
# What comes off one such tree:
#   * the informative set — instances where the chain produced the reported
#     incumbent — and its complement, the retained hard tier;
#   * the tuning set, stratified out of the informative set;
#   * per-heuristic productive vs stale effort, and the inter-acceptance
#     effort-gap quantiles that are literally the unit
#     `mip_heuristic_<name>_stall` is denominated in;
#   * effort at last acceptance — the yield knee — which is a *measured*
#     initial effort vector rather than the inherited one;
#   * charged effort per millisecond, which converts an effort vector into
#     seconds and back.
#
# ── why singles, and why `all` alongside ─────────────────────────────────────
#
# `run_sequential` runs FJ → FPR → LocalMIP → Scylla in order, so a
# wall-clock cap truncates the chain's *tail*, and with no budget and no
# gate the first heuristic takes the entire cap on every instance.  A
# chained probe would therefore report "produced nothing here" for
# instances where three of the four never executed — the config-dependent
# filter #113 exists to avoid, biased against exactly the case the campaign
# wants to find.  Membership is the union over the four singles; `all` is
# run because it is the only arm that measures what the deployed chain
# actually does, and it is held out of the union.
#
# ── the two controls ─────────────────────────────────────────────────────────
#
#   budget  The same thing at effort 1.0, where the budget binds on small
#           models.  `attempt_cap` is derived from the total budget, so a
#           trace at one budget does not exactly reproduce another; this is
#           the second budget level that measurement needs.  It also says
#           whether membership moved.
#   serial  The same thing at `threads=1`, the project's reproducible
#           configuration.  The multi-worker regime is the one the search
#           runs in, so it is what the experiment uses; this control says
#           whether its quantiles are an artifact of worker interleaving.
#
# Both controls run over a subset — they answer a question about the
# experiment, not about the instances.
#
# Read a finished tree with bench/analyze_presolve_probe.py, never with
# analyze_results.py: a presolve-only run computes no dual bound, so its
# gap is meaningless (docs/REPRODUCIBILITY.md).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROBE_CONFIGS="${PROBE_CONFIGS:-fj fpr local_mip scylla all}"
PROBE_OUTPUT_ROOT="${PROBE_OUTPUT_ROOT:-bench/results/probe}"
# 30 s, decided on the measured shape of the cost: HiGHS's own root presolve
# is the floor under any presolve-only run (median 0.45 s, mean 9.98 s over
# the #105 tree, six instances carrying the whole tail), and with the budget
# no longer binding every run now spends its whole cap.
PROBE_TIME_LIMIT="${PROBE_TIME_LIMIT:-30}"
PROBE_SEEDS="${PROBE_SEEDS:-0 1}"
# The controls answer a question about the experiment, so one seed each.
PROBE_CONTROL_SEEDS="${PROBE_CONTROL_SEEDS:-0}"
# The option's ceiling.  Not "infinity": the analysis checks that no run was
# budget-bound rather than assuming it, and a finite ceiling keeps the
# `nnz << 12` product far from overflowing a size_t on the largest model.
PROBE_EFFORT="${PROBE_EFFORT:-1e4}"
# The bounded-budget control's effort: the top of the range everything else
# ships and tunes at.
PROBE_BUDGET_EFFORT="${PROBE_BUDGET_EFFORT:-1.0}"
PROBE_CONTROL_INSTANCES="${PROBE_CONTROL_INSTANCES:-bench/instances_tuning.txt}"

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
preprobe)
	export PLATO_OUTPUT="$PROBE_OUTPUT_ROOT/preprobe"
	export PLATO_SEEDS="$PROBE_SEEDS"
	export PLATO_INSTANCES="${PLATO_INSTANCES:-bench/instances_plato.txt}"
	PLATO_EXTRA_OPTIONS="$(probe_options "$PROBE_EFFORT")"
	;;
budget)
	export PLATO_OUTPUT="$PROBE_OUTPUT_ROOT/control-budget"
	export PLATO_SEEDS="$PROBE_CONTROL_SEEDS"
	export PLATO_INSTANCES="${PLATO_INSTANCES:-$PROBE_CONTROL_INSTANCES}"
	PLATO_EXTRA_OPTIONS="$(probe_options "$PROBE_BUDGET_EFFORT")"
	;;
serial)
	export PLATO_OUTPUT="$PROBE_OUTPUT_ROOT/control-serial"
	export PLATO_SEEDS="$PROBE_CONTROL_SEEDS"
	export PLATO_INSTANCES="${PLATO_INSTANCES:-$PROBE_CONTROL_INSTANCES}"
	PLATO_EXTRA_OPTIONS="$(probe_options "$PROBE_EFFORT")"
	export PLATO_THREADS=1
	;;
*)
	echo "Usage: bench/run_presolve_probe.sh {preprobe|budget|serial} next [hours] | status" >&2
	exit 1
	;;
esac
export PLATO_EXTRA_OPTIONS

export PLATO_CONFIGS="$PROBE_CONFIGS"
export PLATO_TIME_LIMIT="$PROBE_TIME_LIMIT"
# Every pass is a trace: the yield curve, the gap quantiles and the effort
# rate all come off [HeurSol], and membership is instrumentation-independent
# by construction (it reads display rows), so there is no reason to run a
# pass that cannot answer the calibration questions.
export PLATO_DEV_LOG=1
# A presolve-only tree has no dual side; analyze_presolve_probe.py reads it.
export PLATO_ANALYZE=0
if [ -n "${PROBE_COUNT:-}" ]; then
	export PLATO_COUNT="$PROBE_COUNT"
fi

exec "$HERE/run_plato.sh" "$@"
