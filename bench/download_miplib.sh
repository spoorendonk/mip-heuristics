#!/usr/bin/env bash
# Download MIPLIB 2017 collection to /tmp/miplib/ if not already present.
# Source: https://miplib.zib.de/
#
# The full collection (collection.zip) covers all 240 MIPLIB2017 benchmark
# instances including all 233 needed for the PLATO mipfeas benchmark.
# See bench/instances_plato.txt and bench/run_plato.sh for the PLATO workflow.
#
# Usage:
#   bash bench/download_miplib.sh [DEST_DIR]
#   DEST_DIR defaults to /tmp/miplib
set -euo pipefail

DEST="${1:-/tmp/miplib}"
URL="https://miplib.zib.de/downloads/collection.zip"
ZIP="/tmp/miplib_collection.zip"

COUNT=0
[ -d "$DEST" ] && COUNT=$(find "$DEST" -maxdepth 1 -name "*.mps.gz" | wc -l)
if [ "$COUNT" -gt 200 ]; then
    echo "MIPLIB data already present at $DEST ($COUNT instances)"
    exit 0
fi

echo "Downloading MIPLIB 2017 collection..."
mkdir -p "$DEST"
curl -L -o "$ZIP" "$URL"

echo "Extracting to $DEST..."
unzip -o -j "$ZIP" "*.mps.gz" -d "$DEST"
rm -f "$ZIP"

FINAL_COUNT=$(find "$DEST" -maxdepth 1 -name '*.mps.gz' | wc -l) || FINAL_COUNT=0
echo "Done: $FINAL_COUNT instances in $DEST"

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
