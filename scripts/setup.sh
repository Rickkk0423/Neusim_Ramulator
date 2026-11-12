#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
GIT="${GIT:-git}"
PYTHON="${PYTHON:-python3.12}"
"$GIT" submodule update --init --recursive
if [[ ! -x .venv/bin/python ]]; then
  "$PYTHON" -m venv .venv
fi
if ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
  .venv/bin/python -m ensurepip --upgrade
fi
if [[ -z "${CXX:-}" && -x /opt/rh/gcc-toolset-14/root/usr/bin/g++ ]]; then
  export CXX=/opt/rh/gcc-toolset-14/root/usr/bin/g++
fi
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGIT_EXECUTABLE="$(command -v "$GIT")"
.venv/bin/cmake --build build --parallel "${BUILD_JOBS:-2}"
.venv/bin/python -m pytest -q
