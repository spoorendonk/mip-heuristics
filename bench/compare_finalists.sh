#!/usr/bin/env bash
# Score the #107 finalists on the campaign metric.
#
# Usage:
#   bench/compare_finalists.sh [confirm|heldout] [--time-limit N]
#
# `run_finalists.sh` writes one tree per arm — results/finalists/<stage>/<arm>/
# <config>/seed0 — because the harness names a tree after the *suite*, and two
# finalists can share a suite (B and C are both fj+fpr+local_mip).  The arm is
# the experimental unit, so this assembles the view `analyze_results.py` wants,
# <tree>/<arm>/seed0, by symlink.  Symlinks rather than copies: the logs are
# the record and there should be exactly one of each.
#
# The comparison that matters is **presolve-exit ranking versus campaign
# ranking**.  The search optimises `gap + lambda*tau` at presolve exit; #108
# scores primal-integral SGM over 600 s.  A configuration that spends more
# presolve time to exit with a better incumbent can still win or lose on the
# integral, because that time comes out of the B&B search where most of the
# integral is earned — and no presolve-only screen can see it.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
STAGE="${1:-confirm}"; shift || true
ROOT="$REPO/bench/results/finalists"
VIEW="$ROOT/compare-$STAGE"

rm -rf "$VIEW"; mkdir -p "$VIEW"
arms=()
for d in "$ROOT/$STAGE"/*/; do
	[ -d "$d" ] || continue
	arm=$(basename "$d")
	cfg=$(find "$d" -mindepth 2 -maxdepth 2 -type d -name seed0 | head -1)
	[ -n "$cfg" ] || continue
	mkdir -p "$VIEW/$arm"
	ln -sfn "$cfg" "$VIEW/$arm/seed0"
	arms+=("$arm")
done
[ ${#arms[@]} -gt 0 ] || { echo "no arms with runs under $ROOT/$STAGE" >&2; exit 1; }

echo "arms: ${arms[*]}"
for a in "${arms[@]}"; do
	printf '  %-18s %s runs\n' "$a" "$(ls "$VIEW/$a"/seed0/*.log 2>/dev/null | wc -l)"
done
echo
exec python3 "$REPO/bench/analyze_results.py" "$VIEW" --configs "${arms[@]}" --baseline "$@"
