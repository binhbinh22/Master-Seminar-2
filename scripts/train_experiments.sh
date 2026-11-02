#!/usr/bin/env bash
# Run one explicit training stage with fixed, shared datasets.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash scripts/train_experiments.sh [options]

  --stage STAGE       kfold (default), final, or legacy
  --experiments LIST  Comma-separated numbers 1-17, or all (default all for kfold/legacy)
  --selection PATH    Stage-1 selection.json; final runs only the selected methods per language
  --language LANG     python, java, pharo, or all (default all)
  --data-dir PATH     Defaults: data/k_fold_data, data/full_training_data, or data/processed
  --output-dir PATH   Default: results/training/<stage>/<timestamp>-<pid>
  --python PATH       Python executable (default PYTHON_BIN or python3)
  --device DEVICE     auto, cpu, cuda:0, cuda:1, ... (default auto)
  --seed INTEGER      Override base seed (default config seed, 42)
  --dry-run           Print commands without training or creating output files
  -h, --help          Show this help

Stage final requires --selection or an explicit --experiments list.
Relative paths are resolved from the invoking directory. Existing completed staged
results are never overwritten. legacy preserves the old CV + full-train pipeline.
EOF
}

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
experiments_arg=
selection_path=
stage=kfold
language_arg=all
data_dir=
output_dir=
python_bin="${PYTHON_BIN:-python3}"
device=auto
seed_arg=
dry_run=false
while (( $# )); do
    case "$1" in
        --stage|--experiments|--selection|--language|--data-dir|--output-dir|--python|--device|--seed)
            if (( $# < 2 )) || [[ -z "$2" || "$2" == --* ]]; then
                printf 'Missing value for %s\n' "$1" >&2; exit 2
            fi
            case "$1" in
                --stage) stage="$2" ;;
                --experiments) experiments_arg="$2" ;;
                --selection) selection_path="$2" ;;
                --language) language_arg="$2" ;;
                --data-dir) data_dir="$2" ;;
                --output-dir) output_dir="$2" ;;
                --python) python_bin="$2" ;;
                --device) device="$2" ;;
                --seed) seed_arg="$2" ;;
            esac
            shift 2 ;;
        --dry-run) dry_run=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done
case "$stage" in
    kfold) default_data="$repo_dir/data/k_fold_data"; required_files=(all.csv folds.json label_cols.json manifest.json) ;;
    final) default_data="$repo_dir/data/full_training_data"; required_files=(train.csv val.csv test.csv label_cols.json manifest.json) ;;
    legacy) default_data="$repo_dir/data/processed"; required_files=(train.csv test.csv folds.json label_cols.json) ;;
    *) printf 'Invalid stage: %s\n' "$stage" >&2; exit 2 ;;
esac
if [[ ! "$device" =~ ^(auto|cpu|cuda(:[0-9]+)?)$ ]]; then
    printf 'Invalid device: %s\n' "$device" >&2; exit 2
fi
if [[ -n "$seed_arg" ]] && [[ ! "$seed_arg" =~ ^(0|[1-9][0-9]*)$ || ${#seed_arg} -gt 10 || "$seed_arg" -gt 4294967295 ]]; then
    printf 'Invalid seed: %s\n' "$seed_arg" >&2; exit 2
fi
if [[ -n "$selection_path" ]]; then
    if [[ "$stage" != final || -n "$experiments_arg" ]]; then
        printf '%s\n' '--selection requires stage final and cannot be combined with --experiments' >&2; exit 2
    fi
    [[ "$selection_path" == /* ]] || selection_path="$PWD/$selection_path"
    [[ -f "$selection_path" ]] || { printf 'Missing selection: %s\n' "$selection_path" >&2; exit 2; }
elif [[ "$stage" == final && ( -z "$experiments_arg" || "$experiments_arg" == all ) ]]; then
    printf '%s\n' 'Stage final requires --selection or an explicit candidate list in --experiments' >&2; exit 2
fi
experiments_arg="${experiments_arg:-all}"
if [[ "$experiments_arg" == all ]]; then
    experiments=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17)
elif [[ "$experiments_arg" =~ ^([1-9]|1[0-7])(,([1-9]|1[0-7]))*$ ]]; then
    IFS=',' read -r -a experiments <<< "$experiments_arg"
else
    printf 'Invalid experiments: %s\n' "$experiments_arg" >&2; exit 2
fi
case "$language_arg" in
    all) languages=(python java pharo) ;;
    python|java|pharo) languages=("$language_arg") ;;
    *) printf 'Invalid language: %s\n' "$language_arg" >&2; exit 2 ;;
esac
command -v "$python_bin" >/dev/null 2>&1 || { printf 'Python not found: %s\n' "$python_bin" >&2; exit 2; }
if [[ "$python_bin" == */* && "$python_bin" != /* ]]; then python_bin="$PWD/$python_bin"; fi
data_dir="${data_dir:-$default_data}"
output_dir="${output_dir:-$repo_dir/results/training/$stage/$(date +%Y%m%d-%H%M%S)-$$}"
[[ "$data_dir" == /* ]] || data_dir="$PWD/$data_dir"
[[ "$output_dir" == /* ]] || output_dir="$PWD/$output_dir"
for language in "${languages[@]}"; do
    for file in "${required_files[@]}"; do
        [[ -f "$data_dir/$language/$file" ]] || { printf 'Missing input: %s\n' "$data_dir/$language/$file" >&2; exit 2; }
    done
done
cd -- "$repo_dir"
progress_mode=auto
if [[ -t 1 ]]; then progress_mode=on; fi
export MODEL_ZOO_PROGRESS="${MODEL_ZOO_PROGRESS:-$progress_mode}"
printf 'Stage: %s\nOutput root: %s\n' "$stage" "$output_dir"
for language in "${languages[@]}"; do
    language_experiments=("${experiments[@]}")
    if [[ -n "$selection_path" ]]; then
        selected_numbers=$("$python_bin" -m src.evaluation.two_stage_results candidate --selection "$selection_path" --language "$language")
        IFS=',' read -r -a language_experiments <<< "$selected_numbers"
    fi
    for experiment in "${language_experiments[@]}"; do
        run_dir="$output_dir/exp$experiment/$language"
        command_args=("$python_bin" -u -m "experiments.dl.train_exp$experiment" --stage "$stage"
                      --language "$language" --data-dir "$data_dir" --output-dir "$run_dir" --device "$device")
        if [[ -n "$seed_arg" ]]; then command_args+=(--seed "$seed_arg"); fi
        if [[ -n "$selection_path" ]]; then
            config_path=$("$python_bin" -m src.evaluation.two_stage_results candidate --selection "$selection_path" --language "$language" --model "E$experiment")
            [[ -f "$config_path" ]] || { printf 'Missing selected config: %s\n' "$config_path" >&2; exit 2; }
            command_args+=(--selection "$selection_path" --config "$config_path")
        fi
        printf '\nE%s / %s:' "$experiment" "$language"
        printf ' %q' "${command_args[@]}"
        printf '\n'
        if [[ "$dry_run" == true ]]; then continue; fi
        mkdir -p -- "$run_dir"
        if "${command_args[@]}" 2>&1 | tee "$run_dir/training.log"; then
            printf 'Completed E%s / %s\n' "$experiment" "$language"
        else
            status=$?
            printf 'Failed E%s / %s (exit %s). Log: %s/training.log\n' "$experiment" "$language" "$status" "$run_dir" >&2
            exit "$status"
        fi
    done
done
if [[ "$stage" == final ]]; then
    analysis_args=("$python_bin" -m src.evaluation.two_stage_results report --results "$output_dir" --output-dir "$output_dir/report")
else
    analysis_args=("$python_bin" -m src.evaluation.analyze_results --results "$output_dir" --output-dir "$output_dir/model_zoo")
fi
printf '\nAnalyze results:'
printf ' %q' "${analysis_args[@]}"
printf '\n'
if [[ "$dry_run" != true ]]; then "${analysis_args[@]}"; fi
