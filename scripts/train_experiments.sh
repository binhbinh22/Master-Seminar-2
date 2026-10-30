#!/usr/bin/env bash
# Run the selected experiments sequentially with shared processed splits.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash scripts/train_experiments.sh [options]

Options:
  --experiments LIST  Comma-separated exp numbers (1-12), or all (default: all)
  --language LANG     python, java, pharo, or all (default: all)
  --data-dir PATH     Processed root (default: <repo>/data/processed)
  --output-dir PATH   Output root (default: <repo>/results/training/<timestamp>)
  --python PATH       Python executable (default: PYTHON_BIN or python3)
  --device DEVICE     auto, cpu, cuda:0, cuda:1, ... (default: auto)
  --dry-run           Print commands without training or creating output files
  -h, --help          Show this help

Examples:
  bash scripts/train_experiments.sh --language python
  bash scripts/train_experiments.sh --experiments 1,2,6 --language java
  bash scripts/train_experiments.sh --dry-run

Runs stop at the first failure. Each experiment/language gets a separate log.
After successful training, a comparison report is written under OUTPUT/model_zoo/.
Relative custom paths are resolved from the directory where you invoke the script.
EOF
}

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
experiments_arg=all
language_arg=all
data_dir="$repo_dir/data/processed"
output_dir="$repo_dir/results/training/$(date +%Y%m%d-%H%M%S)-$$"
python_bin="${PYTHON_BIN:-python3}"
device=auto
dry_run=false

while (( $# )); do
    case "$1" in
        --experiments|--language|--data-dir|--output-dir|--python|--device)
            if (( $# < 2 )) || [[ -z "$2" || "$2" == --* ]]; then
                printf 'Missing value for %s\n' "$1" >&2
                exit 2
            fi
            case "$1" in
                --experiments) experiments_arg="$2" ;;
                --language) language_arg="$2" ;;
                --data-dir) data_dir="$2" ;;
                --output-dir) output_dir="$2" ;;
                --python) python_bin="$2" ;;
                --device) device="$2" ;;
            esac
            shift 2
            ;;
        --dry-run) dry_run=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ ! "$device" =~ ^(auto|cpu|cuda(:[0-9]+)?)$ ]]; then
    printf 'Invalid device: %s (expected auto, cpu, or cuda:<index>)\n' "$device" >&2
    exit 2
fi

if [[ "$experiments_arg" == all ]]; then
    experiments=(1 2 3 4 5 6 7 8 9 10 11 12)
elif [[ "$experiments_arg" =~ ^([1-9]|1[0-2])(,([1-9]|1[0-2]))*$ ]]; then
    IFS=',' read -r -a experiments <<< "$experiments_arg"
else
    printf 'Invalid experiments: %s (expected all or e.g. 7,8,9,10,11,12)\n' "$experiments_arg" >&2
    exit 2
fi

case "$language_arg" in
    all) languages=(python java pharo) ;;
    python|java|pharo) languages=("$language_arg") ;;
    *) printf 'Invalid language: %s\n' "$language_arg" >&2; exit 2 ;;
esac

if ! command -v "$python_bin" >/dev/null 2>&1; then
    printf 'Python executable not found: %s\n' "$python_bin" >&2
    exit 2
fi
# Resolve explicit relative executable paths before changing directory.
if [[ "$python_bin" == */* && "$python_bin" != /* ]]; then
    python_bin="$PWD/$python_bin"
fi
[[ "$data_dir" == /* ]] || data_dir="$PWD/$data_dir"
[[ "$output_dir" == /* ]] || output_dir="$PWD/$output_dir"

for language in "${languages[@]}"; do
    for file in train.csv test.csv folds.json label_cols.json; do
        if [[ ! -f "$data_dir/$language/$file" ]]; then
            printf 'Missing input: %s\n' "$data_dir/$language/$file" >&2
            exit 2
        fi
    done
done

cd -- "$repo_dir"
# tee makes Python's stderr a pipe; preserve progress when invoked in a terminal.
progress_mode=auto
if [[ -t 1 ]]; then
    progress_mode=on
fi
export MODEL_ZOO_PROGRESS="${MODEL_ZOO_PROGRESS:-$progress_mode}"
printf 'Output root: %s\n' "$output_dir"
for experiment in "${experiments[@]}"; do
    for language in "${languages[@]}"; do
        run_dir="$output_dir/exp$experiment/$language"
        command_args=("$python_bin" -u -m "experiments.dl.train_exp$experiment"
                      --language "$language" --data-dir "$data_dir" --output-dir "$run_dir"
                      --device "$device")
        printf '\nE%s / %s:' "$experiment" "$language"
        printf ' %q' "${command_args[@]}"
        printf '\n'
        if [[ "$dry_run" == true ]]; then
            continue
        fi
        mkdir -p -- "$run_dir"
        if "${command_args[@]}" 2>&1 | tee "$run_dir/training.log"; then
            printf 'Completed E%s / %s\n' "$experiment" "$language"
        else
            status=$?
            printf 'Failed E%s / %s (exit %s). Log: %s/training.log\n' \
                "$experiment" "$language" "$status" "$run_dir" >&2
            exit "$status"
        fi
    done
done

analysis_args=("$python_bin" -m src.evaluation.analyze_results
               --results "$output_dir" --output-dir "$output_dir/model_zoo")
printf '\nAnalyze results:'
printf ' %q' "${analysis_args[@]}"
printf '\n'
if [[ "$dry_run" != true ]]; then
    "${analysis_args[@]}"
fi

