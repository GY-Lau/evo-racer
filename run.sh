#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
EVO_PORT="${PORT:-8765}"
EVO_BIND="${BIND:-127.0.0.1}"
EVO_PYTHON="${EVO_PYTHON:-python3}"
if [[ "$EVO_PYTHON" == python3 && -x .venv/bin/python ]]; then
  EVO_PYTHON="$PWD/.venv/bin/python"
fi
echo "EVO / 20 · http://${EVO_BIND}:${EVO_PORT}"
echo "按 Ctrl+C 停止；局域网访问：BIND=0.0.0.0 bash run.sh"
exec "$EVO_PYTHON" server.py --port "$EVO_PORT" --host "$EVO_BIND" "$@"
