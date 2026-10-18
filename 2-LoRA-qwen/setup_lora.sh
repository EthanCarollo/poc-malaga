#!/usr/bin/env bash
set -euo pipefail
SHARM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SHARM_DIR/setup_sharm.sh" "$@"
