#!/usr/bin/env bash
set -euo pipefail
python -u scripts/build_epfl_return_bundle.py --repo-root . --output VISTA_EPFL_RETURN_v1 --archive
