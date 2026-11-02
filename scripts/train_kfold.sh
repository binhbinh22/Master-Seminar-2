#!/usr/bin/env bash
# Stage 1: screen methods with K-fold CV on all data.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
for argument in "$@"; do
    case "$argument" in
        --stage|--stage=*)
            printf '%s\n' 'train_kfold.sh always runs stage kfold.' >&2
            exit 2 ;;
        -h|--help)
            cat <<'EOF'
Usage: bash scripts/train_kfold.sh [options]

Run K-fold CV on data/k_fold_data (all rows), without final training/test.
Options:
  --experiments LIST  Numbers 1-17 separated by commas, or all (default all)
  --language LANG     python, java, pharo, or all (default all)
  --device DEVICE     auto (default), cpu, cuda:0, cuda:1, ...
  --seed INTEGER      Override the config seed (default 42)
  --python PATH       Python executable (default PYTHON_BIN or python3)
  --data-dir PATH     Override the K-fold dataset root
  --output-dir PATH   Override the results directory
  --dry-run           Print commands without training or creating output
  -h, --help          Show this help

After training, use src.evaluation.two_stage_results select to choose top 3-4.
EOF
            exit 0 ;;
    esac
done
exec bash "$script_dir/train_experiments.sh" --stage kfold "$@"
