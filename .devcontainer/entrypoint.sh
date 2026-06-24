#!/usr/bin/env bash
set -euo pipefail

if [ -f /opt/conda/etc/profile.d/micromamba.sh ]; then
  # shellcheck source=/dev/null
  source /opt/conda/etc/profile.d/micromamba.sh
  micromamba activate hovsg >/dev/null 2>&1 || true
fi

exec "$@"
