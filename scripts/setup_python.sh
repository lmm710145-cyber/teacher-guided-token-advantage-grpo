#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Linux x86_64 + Python 3.10. Java 11 and C/C++ build tools must be available.
python3.10 -m venv envs/train
python3.10 -m venv envs/inference
python3.10 -m venv envs/bench
for name in train inference bench; do
  "envs/$name/bin/python" -m pip install --upgrade pip setuptools wheel
done
envs/train/bin/pip install -r requirements/train.txt
envs/inference/bin/pip install -r requirements/inference.txt
envs/bench/bin/pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
envs/bench/bin/pip install -r requirements/bench.txt
envs/bench/bin/pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.3.0/en_core_web_lg-3.3.0-py3-none-any.whl
