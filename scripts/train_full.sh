#!/usr/bin/env bash
# Stage 2: fit selected methods with the fixed 80/10/10 split.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
for argument in "$@"; do
    case "$argument" in
        --stage|--stage=*)
            printf '%s\n' 'train_full.sh always runs stage final.' >&2
            exit 2 ;;
        -h|--help)
            cat <<'EOF'
Usage: bash scripts/train_full.sh --selection PATH [options]
   or: bash scripts/train_full.sh --experiments LIST [options]

Train on data/full_training_data: 80% train, 10% validation, 10% test.
Options:
  --selection PATH    Stage-1 selection.json; use selected methods/configs
  --experiments LIST  Explicit method numbers instead of --selection
  --language LANG     python, java, pharo, or all (default all)
  --device DEVICE     auto (default), cpu, cuda:0, cuda:1, ...
  --seed INTEGER      Override the config seed (default 42)
  --python PATH       Python executable (default PYTHON_BIN or python3)
  --data-dir PATH     Override the final dataset root
  --output-dir PATH   Override the results directory
  --dry-run           Print commands without training or creating output
  -h, --help          Show this help
EOF
            exit 0 ;;
    esac
done
exec bash "$script_dir/train_experiments.sh" --stage final "$@"
