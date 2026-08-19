#!/bin/bash
# Experiment 4 pipeline on the local GPU: zero-shot -> LoRA SFT -> re-eval -> linear probe.
set -e
cd "$(dirname "$0")/.."
export HF_HOME=/home/neurico/hfcache
export CC="$PWD/.toolbin/cc"
export PATH="$PWD/.toolbin:$PATH"
export PYTHONPATH=src
export TOKENIZERS_PARALLELISM=false
for stage in zeroshot probe train eval_trained; do
  echo "=== stage: $stage ==="
  .venv/bin/python -u src/carb/local_model.py --stage "$stage"
done
