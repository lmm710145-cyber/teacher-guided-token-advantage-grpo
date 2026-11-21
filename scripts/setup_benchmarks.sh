#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p benchmarks
if [ ! -d benchmarks/AgentGym/.git ]; then
  git clone https://github.com/WooooDyy/AgentGym.git benchmarks/AgentGym
fi
if [ ! -d benchmarks/alfworld/.git ]; then
  git clone https://github.com/alfworld/alfworld.git benchmarks/alfworld
fi
git -C benchmarks/AgentGym checkout 3ef9235d23e68e7c2920c5422ad957dc8ced5c6c
git -C benchmarks/alfworld checkout aaba6870f86c5be6a08a491f32a50b906227bc3e
patch_file="$(pwd)/patches/webshop_option_reward.patch"
if git -C benchmarks/AgentGym apply --check "$patch_file" 2>/dev/null; then
  git -C benchmarks/AgentGym apply "$patch_file"
else
  git -C benchmarks/AgentGym apply --reverse --check "$patch_file"
fi
envs/bench/bin/pip install --no-deps -e benchmarks/alfworld
envs/bench/bin/python scripts/download_alfworld_assets.py
envs/bench/bin/python scripts/build_webshop_catalog.py
repo_root="$(pwd)"
cd benchmarks/AgentGym/agentenv-webshop/webshop/search_engine
mkdir -p resources resources_100 resources_1k resources_100k
"$repo_root/envs/bench/bin/python" convert_product_file_format.py
"$repo_root/envs/bench/bin/python" -m pyserini.index.lucene \
  --collection JsonCollection --input resources_1k --index indexes_1k \
  --generator DefaultLuceneDocumentGenerator --threads 4 \
  --storePositions --storeDocvectors --storeRaw
