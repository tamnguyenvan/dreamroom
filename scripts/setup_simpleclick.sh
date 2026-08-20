#!/usr/bin/env bash
# Set up the local SimpleClick port: clone the repo, install deps, download weights.
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"
SIMPLECLICK_ROOT="third_party/SimpleClick"
CHECKPOINT="weights/cocolvis_vit_huge.pth"
CHECKPOINT_ID="1GXk6q5fwKo2twkY5ZZGjVKCgJv7XeLAW"

if [ ! -d "$SIMPLECLICK_ROOT" ]; then
    echo ">> cloning SimpleClick v1.0"
    git clone --depth 1 --branch v1.0 https://github.com/uncbiag/SimpleClick "$SIMPLECLICK_ROOT"
else
    echo ">> SimpleClick already cloned"
fi

echo ">> installing mmcv (pure-python build, no ops; no build isolation: setup.py imports torch/pkg_resources)"
MMCV_WITH_OPS=0 "$PYTHON" -m pip install --no-build-isolation 'mmcv==1.6.2'

if [ ! -f "$CHECKPOINT" ]; then
    echo ">> downloading checkpoint (~2.7 GB)"
    "$PYTHON" -m gdown "$CHECKPOINT_ID" -O "$CHECKPOINT"
else
    echo ">> checkpoint already present"
fi

echo ">> setup complete"
