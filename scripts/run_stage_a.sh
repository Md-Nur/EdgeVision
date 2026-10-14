#!/usr/bin/env bash
set -e

# Resolve repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

# Resolve Python binary
if [ -f "${ROOT_DIR}/.venv/bin/python" ]; then
    PYTHON="${ROOT_DIR}/.venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
else
    PYTHON="python"
fi

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

echo "================================================================================"
echo " Mukti Stage A: Model Training & Evaluation Orchestrator"
echo "================================================================================"
echo " Project Root : ${ROOT_DIR}"
echo " Python Bin   : ${PYTHON}"
echo " Arguments    : $@"
echo "================================================================================"

exec "${PYTHON}" -u src/orchestrator.py "$@"
