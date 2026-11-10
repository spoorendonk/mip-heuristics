#!/usr/bin/env bash
# #237: main (FPR with WalkSAT) vs cdcl (FJ's CDCL lane) on the 42 pure-binary
# mipfeas instances, 3 seeds, 60 s.
#
#     bench/run_cdcl42.sh size | run | report [cuopt_smoke.py options]
#
# Later options win, so a smoke can name its own --instances and --seeds; its own
# tree goes in CDCL42_OUTPUT, never --output, and under systemd-run through
# --setenv, since a unit does not inherit the shell's environment.  The binary is
# build/bin/highs, never --binary, which the record below would not follow.  Each
# solve holds the machine lock (macOS: the whole build budget) unless SMOKE_WRAP
# says otherwise.  Stopped or cut short, `run` again resumes.  The tree pins the
# binary and records, until it is pinned, the checkout's commit and that
# binary's build time: the time is what shows a binary older than the commit.
set -euo pipefail

command=${1:?usage: run_cdcl42.sh size|run|report [options]}
shift
bench=$(cd "$(dirname "$0")" && pwd)
out=${CDCL42_OUTPUT:-$bench/results/cdcl42/$(hostname -s)}

if [ "$command" = run ]; then
  if ! git -C "$bench" diff --quiet HEAD; then
    echo "run_cdcl42: uncommitted changes; the tree records a commit" >&2
    exit 2
  fi
  # xk's machine lock does not run on macOS; there a solve charged all the
  # build budget keeps builds off it instead (cuopt_smoke.py's docstring).
  if [ "$(uname -s)" = Darwin ]; then
    wrap="wcp build --here --jobs 8 --"
  else
    wrap="xk lock --"
  fi
  export SMOKE_WRAP=${SMOKE_WRAP-$wrap}
  binary=$bench/../build/bin/highs
  if [ ! -e "$out/bin/highs" ] && [ -e "$binary" ]; then
    mkdir -p "$out"
    {
      git -C "$bench" rev-parse HEAD
      date -r "$binary" '+%Y-%m-%d %H:%M:%S %z' # GNU and BSD date alike
    } >"$out/commit"
  fi
fi

exec python3 "$bench/cuopt_smoke.py" "$command" \
  --instances "$bench/instances_mipfeas_binary.txt" \
  --seeds 0 1 2 \
  --arms main cdcl \
  --output "$out" \
  "$@"
