#!/usr/bin/env bash
# The #107 joint search: irace over the eight presolve parameters.
#
# Usage:
#   bench/run_irace.sh all              every pre-registered lambda, in order
#   bench/run_irace.sh <lambda-tag>     one of them (e.g. 1_600)
#   bench/run_irace.sh status           what has finished
#
# The objective, the lambda sweep, the selection rule and the limitations are
# fixed in bench/irace/PREREGISTRATION.md and were signed off before the first
# experiment ran.  This script only executes what that file commits to; it
# decides nothing.  If you find yourself wanting to change a setting here,
# change the pre-registration first and re-sign it, or the run is no longer
# the one that was registered.
#
# Resumability: irace writes its state to `irace.Rdata` in the run's own
# directory and `--recovery-file` resumes from it, so a killed chunk continues
# rather than restarting.  `status` reports which lambdas are done.
#
# Artifacts per lambda, under bench/results/irace/<tag>/:
#   irace.Rdata   irace's own state and full history — the run log #107 asks
#                 to be kept with the results tree, and what the selection is
#                 re-derived from rather than re-run
#   irace.log     everything irace printed
#   target-runs/  one directory per experiment, written by run_target.py
#
# NOTE: irace resolves relative paths in a scenario file against *that file's*
# directory, so the scenario is invoked from bench/irace/ and the paths in it
# stay relative.  Learned by copying it elsewhere and watching irace refuse.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
IRACE_DIR="$HERE/irace"
RESULTS="${IRACE_RESULTS:-$REPO/bench/results/irace}"

# The three pre-registered cost weights: the derived g(0)/T at the campaign's
# 600 s limit, and one octave either side.  The *family* of resulting
# configurations is the deliverable, not the middle one alone.
LAMBDAS=("1_1200:0.00083333333" "1_600:0.0016666667" "1_300:0.0033333333")

IRACE_BIN="${IRACE_BIN:-$HOME/R/x86_64-pc-linux-gnu-library/4.3/irace/bin/irace}"
export R_LIBS_USER="${R_LIBS_USER:-$HOME/R/x86_64-pc-linux-gnu-library/4.3}"

# Deployment settings.  `threads` is deliberately NOT pinned: the four budgets
# are not worker-count invariant (FJ's is per worker, the other three per
# dispatch), so the search has to run at the count #108 will use, which is
# HiGHS's own default.  See the pre-registration's limitations.
export MIP_HEURISTICS_BINARY="${MIP_HEURISTICS_BINARY:-$REPO/build/bin/highs}"
export RUN_TARGET_TIME_LIMIT="${RUN_TARGET_TIME_LIMIT:-60}"
export RUN_TARGET_NO_SOLUTION_PENALTY="${RUN_TARGET_NO_SOLUTION_PENALTY:-2.0}"
export RUN_TARGET_PYTHON="${RUN_TARGET_PYTHON:-python3}"

run_one() {
	local tag=$1 lambda=$2
	local dir="$RESULTS/$tag"
	mkdir -p "$dir/target-runs"

	if [ -s "$dir/best.txt" ]; then
		echo "== lambda=$lambda ($tag): already complete, skipping"
		return 0
	fi

	local recover=()
	if [ -s "$dir/irace.Rdata" ]; then
		echo "== lambda=$lambda ($tag): resuming from irace.Rdata"
		recover=(--recovery-file "$dir/irace.Rdata")
	fi

	echo "================================================================"
	echo "irace #107 joint search"
	echo "  lambda      : $lambda   (tag $tag)"
	echo "  binary      : $MIP_HEURISTICS_BINARY"
	echo "  per-run cap : ${RUN_TARGET_TIME_LIMIT}s"
	echo "  output      : $dir"
	echo "================================================================"

	RUN_TARGET_LAMBDA="$lambda" \
	RUN_TARGET_RUN_DIR="$dir/target-runs" \
		"$IRACE_BIN" \
		--scenario "$IRACE_DIR/scenario.txt" \
		--exec-dir "$dir" \
		--log-file "$dir/irace.Rdata" \
		"${recover[@]}" 2>&1 | tee "$dir/irace.log"

	# The elites irace selected, extracted from its own state so the record
	# and the report cannot disagree.
	Rscript -e "suppressMessages(library(irace));
	            l <- read_logfile('$dir/irace.Rdata');
	            print(removeConfigurationsMetaData(getFinalElites(l)))" \
		> "$dir/best.txt" 2>&1 || true
	echo "Wrote $dir/best.txt"
}

cmd_status() {
	echo "irace #107 search  ($RESULTS)"
	for entry in "${LAMBDAS[@]}"; do
		local tag=${entry%%:*} lambda=${entry##*:}
		local dir="$RESULTS/$tag" state="not started"
		if [ -s "$dir/best.txt" ]; then
			state="COMPLETE"
		elif [ -s "$dir/irace.Rdata" ]; then
			state="in progress / resumable"
		fi
		printf '  %-8s lambda=%-14s %s\n' "$tag" "$lambda" "$state"
	done
}

CMD="${1:-status}"
case "$CMD" in
status) cmd_status ;;
all)
	for entry in "${LAMBDAS[@]}"; do
		run_one "${entry%%:*}" "${entry##*:}"
	done
	echo ""
	cmd_status
	;;
*)
	found=0
	for entry in "${LAMBDAS[@]}"; do
		if [ "${entry%%:*}" = "$CMD" ]; then
			run_one "${entry%%:*}" "${entry##*:}"
			found=1
		fi
	done
	if [ "$found" -eq 0 ]; then
		echo "Unknown lambda tag '$CMD'. Known: $(for e in "${LAMBDAS[@]}"; do printf '%s ' "${e%%:*}"; done)" >&2
		exit 1
	fi
	;;
esac
