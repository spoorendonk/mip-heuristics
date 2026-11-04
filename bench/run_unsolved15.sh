#!/usr/bin/env bash
# #244: main, ruin, cut, recombine on the 15 unsolved instances, 3 seeds, 60 s.
#
#     bench/run_unsolved15.sh size | run | report [cuopt_smoke.py options]
#
# Later options win, so a smoke can name its own --instances and --seeds; its own
# tree goes in UNSOLVED15_OUTPUT, never --output, and under systemd-run through
# --setenv, since a unit does not inherit the shell's environment.  The binary is
# build/bin/highs, never --binary, which the record below would not follow.  Each
# solve holds the machine lock unless SMOKE_WRAP says otherwise.  Stopped or cut
# short, `run` again resumes.  The tree pins the binary and records, until it is
# pinned, the checkout's commit and that binary's build time: the time is what
# shows a binary older than the commit.
set -euo pipefail

command=${1:?usage: run_unsolved15.sh size|run|report [options]}
shift
bench=$(cd "$(dirname "$0")" && pwd)
out=${UNSOLVED15_OUTPUT:-$bench/results/unsolved15/$(hostname -s)}

if [ "$command" = run ]; then
  if ! git -C "$bench" diff --quiet HEAD; then
    echo "run_unsolved15: uncommitted changes; the tree records a commit" >&2
    exit 2
  fi
  export SMOKE_WRAP=${SMOKE_WRAP-xk lock --}
  binary=$bench/../build/bin/highs
  if [ ! -e "$out/bin/highs" ] && [ -e "$binary" ]; then
    mkdir -p "$out"
    {
      git -C "$bench" rev-parse HEAD
      stat -c '%y' "$binary"
    } >"$out/commit"
  fi
fi

exec python3 "$bench/cuopt_smoke.py" "$command" \
  --instances "$bench/instances_unsolved15.txt" \
  --seeds 0 1 2 \
  --arms main ruin cut recombine \
  --output "$out" \
  "$@"
