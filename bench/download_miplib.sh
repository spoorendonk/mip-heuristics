#!/usr/bin/env bash
# Download MIPLIB 2017 collection to /tmp/miplib/ if not already present.
# Source: https://miplib.zib.de/
set -euo pipefail

DEST="${1:-/tmp/miplib}"
URL="https://miplib.zib.de/downloads/collection.zip"
ZIP="/tmp/miplib_collection.zip"

COUNT=$(find "$DEST" -maxdepth 1 -name "*.mps.gz" 2>/dev/null | wc -l)
if [ -d "$DEST" ] && [ "$COUNT" -gt 100 ]; then
    echo "MIPLIB data already present at $DEST ($COUNT instances)"
    exit 0
fi

echo "Downloading MIPLIB 2017 collection..."
mkdir -p "$DEST"
curl -L -o "$ZIP" "$URL"

echo "Extracting to $DEST..."
unzip -o -j "$ZIP" "*.mps.gz" -d "$DEST"
rm -f "$ZIP"

echo "Done: $(find "$DEST" -maxdepth 1 -name '*.mps.gz' | wc -l) instances in $DEST"
