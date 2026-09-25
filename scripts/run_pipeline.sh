#!/usr/bin/env bash
# Rebuild everything from raw PDFs to a queryable index, then run the eval.
set -euo pipefail
cd "$(dirname "$0")/.."

python -m src.ingest
python -m src.chunk
python -m src.build_index
python -m eval.run_eval "$@"
