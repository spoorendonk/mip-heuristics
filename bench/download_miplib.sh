#!/usr/bin/env bash
# Download the MIPLIB 2017 collection, unless a copy is already on this machine.
# Source: https://miplib.zib.de/
#
# The full collection (collection.zip) covers all 240 MIPLIB2017 benchmark
# instances including all 233 needed for the PLATO mipfeas benchmark.
# See bench/instances_plato.txt and bench/run_plato.sh for the PLATO workflow.
#
# A 3.5 GB download is worth going out of its way to avoid, so an existing
# copy is *searched for* rather than assumed at one path: the candidates below
# are probed in order and the first populated one wins.  `~/data/miplib` comes
# before `/tmp/miplib` because /tmp does not survive a reboot on most distros
# and re-downloading is the expensive failure.  /tmp stays in the list so the
# checkouts that already populated it keep working.
#
# Usage:
#   bash bench/download_miplib.sh [DEST_DIR]
#   MIPLIB_DIR=/path/to/miplib bash bench/download_miplib.sh
#
# With no DEST_DIR the search order is:
#   $MIPLIB_DIR, ~/data/miplib, /tmp/miplib
# and a download that finds none of them lands in the first of those that is
# set — i.e. ~/data/miplib unless MIPLIB_DIR overrides it.
set -euo pipefail

URL="https://miplib.zib.de/downloads/collection.zip"

# An extracted collection is "present" above this many instances.  The full
# collection is ~1065; a partial extract or an unrelated directory holding a
# handful of .mps.gz files should not satisfy the check.
MIN_INSTANCES=200

count_instances() { # dir -> instance count on stdout (0 when absent/unreadable)
	# An unreadable candidate counts as absent rather than aborting the run.
	# /tmp/miplib is probed for every user now, so on a shared box it may be
	# someone else's mode-700 directory.  GNU find exits 1 on a permission
	# error, which `pipefail` would turn into a `set -e` abort of exactly the
	# run that needed to download.  Test readability explicitly instead of
	# relying on the exit status: find implementations disagree about it
	# (bfs 4.1 exits 0 where GNU findutils exits 1).
	#
	# `-r` only, deliberately not `-r && -x`: a mode-444 directory still
	# enumerates (measured: find and os.scandir both count 205 of 205), so
	# adding `-x` would make this report 0 while run_benchmark.py reports a
	# full collection -- the two artifacts resolving to different directories
	# is the one failure this shared search path exists to prevent.
	{ [ -d "$1" ] && [ -r "$1" ]; } || {
		echo 0
		return
	}
	find "$1" -maxdepth 1 -name '*.mps.gz' 2>/dev/null | wc -l
}

# Candidate locations, most-preferred first.  An explicit DEST_DIR argument
# short-circuits the search entirely: asking for a specific directory should
# not silently resolve to a different one.
CANDIDATES=()
if [ "$#" -gt 0 ] && [ -n "$1" ]; then
	CANDIDATES=("$1")
else
	[ -n "${MIPLIB_DIR:-}" ] && CANDIDATES+=("$MIPLIB_DIR")
	CANDIDATES+=("$HOME/data/miplib" "/tmp/miplib")
fi

for dir in "${CANDIDATES[@]}"; do
	COUNT=$(count_instances "$dir")
	if [ "$COUNT" -gt "$MIN_INSTANCES" ]; then
		echo "MIPLIB data already present at $dir ($COUNT instances)" >&2
		# Reusing a /tmp copy avoids today's 3.5 GB download but not tomorrow's:
		# the point of the search path is that a restart stops costing a refetch,
		# and a collection under /tmp still evaporates.  Say so rather than
		# moving 7.3 GB across filesystems as a silent side effect.
		case "$dir" in
		/tmp/*)
			echo "Note: $dir is under /tmp and will not survive a reboot." >&2
			echo "      To keep it: mkdir -p $HOME/data && mv $dir $HOME/data/miplib" >&2
			;;
		esac
		echo "$dir"
		exit 0
	fi
done

DEST="${CANDIDATES[0]}"
echo "No MIPLIB collection found in: ${CANDIDATES[*]}" >&2
echo "Downloading MIPLIB 2017 collection to $DEST ..." >&2
mkdir -p "$DEST"

# Stage the archive beside the extracted data, not in /tmp: the zip is 3.5 GB
# and a tmpfs /tmp will not hold it even when the real destination has room.
# Deliberately not dot-prefixed.  A retained part-file is something the user
# may have to delete by hand, and a hidden one leaves `ls` showing an empty
# directory with 3.5 GB in it.  The extraction glob is `*.mps.gz`, so a
# visible name here cannot be mistaken for an instance.
ZIP="$DEST/miplib_collection.zip.part"
# The trap covers the download only.  A partial transfer must not be left
# looking like a valid archive -- but once curl has succeeded, those 3.5 GB are
# the expensive thing on this machine, and a failed *extraction* must leave
# them for a retry instead of forcing a refetch.  Disk-full is the realistic
# extraction failure now that the archive and the extracted data share a
# filesystem (~10.8 GB peak), which is exactly when the retry matters most.
trap 'rm -f "$ZIP"' EXIT
curl -L --fail -C - -o "$ZIP" "$URL"
trap - EXIT

# `curl -C -` resumes onto whatever bytes are already there, and it cannot tell
# a genuine partial transfer from an unrelated one: if the local prefix does
# not match the server's, it appends the tail and exits 0.  The part-file
# survives a Ctrl-C (bash runs an EXIT trap on SIGTERM but not on SIGINT), so
# the mismatched case is reachable -- an interrupted download through a captive
# portal that served an HTML error page, resumed later against the real file.
# Without this test the retained-archive behaviour above turns that into a loop
# that repeats forever, since the corrupt archive is kept for the next retry.
echo "Verifying archive..." >&2
if ! unzip -t -q "$ZIP" >&2; then
	echo "ERROR: $ZIP is corrupt (a resumed download onto unrelated bytes does" >&2
	echo "       this).  Delete it and rerun to fetch a fresh copy:" >&2
	echo "         rm -f $ZIP" >&2
	exit 1
fi

echo "Extracting to $DEST..." >&2
# unzip lists every extracted member on stdout, and stdout here carries the
# resolved directory and nothing else.
unzip -q -o -j "$ZIP" "*.mps.gz" -d "$DEST" >&2
rm -f "$ZIP"

FINAL_COUNT=$(count_instances "$DEST")
echo "Done: $FINAL_COUNT instances in $DEST" >&2

# Sanity-check a few PLATO instances that were missing from instances_bench.txt
# before the PLATO list was added. All 233 PLATO instances should be present.
MISSING=0
for inst in assign1-5-8 bab2 binkar10_1 chromaticindex512-7 eil33-2 \
	istanbul-no-cutoff leo1 map10 neos-631710 ns1644855 \
	pg5_34 rmatr200-p5 satellites2-40 snp-02-004-104 supportcase40; do
	if [ ! -f "$DEST/${inst}.mps.gz" ] && [ ! -f "$DEST/${inst}.mps" ]; then
		echo "WARNING: PLATO instance not found after download: $inst" >&2
		MISSING=$((MISSING + 1))
	fi
done
if [ "$MISSING" -gt 0 ]; then
	echo "WARNING: $MISSING PLATO instances missing — the collection.zip may be incomplete." >&2
	echo "Visit https://miplib.zib.de/ for individual instance downloads." >&2
fi

# stdout carries the resolved directory and nothing else, so callers can do
# `DATA_DIR=$(bash bench/download_miplib.sh)`.  All progress goes to stderr.
echo "$DEST"
