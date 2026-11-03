#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$ROOT/data/era5_240/full"
DOWNLOAD_SCRIPT="$ROOT/geoarches/geoarches/download/dl_era.py"
PYTHON="$ROOT/data-venv/bin/python"

mkdir -p "$DATA_DIR"
mapfile -t YEARS < <(seq 1998 2020)

"$PYTHON" "$DOWNLOAD_SCRIPT" \
  --folder "$DATA_DIR" \
  --years "${YEARS[@]}" \
  --hours 0 6 12 18 \
  --max_workers 4
