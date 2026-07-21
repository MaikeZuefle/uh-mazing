#!/bin/bash

# Usage: ./run_icl.sh <GPU_ID> [--sections 1,2,3,4,5] [--num_icl 2]
# Example: ./run_icl.sh 0 --sections 3,4
#
# In-context learning sweep: 5 model + decoding combinations, "standard" prompt only.
#   1  Phi (audio)            sampling_t0.2
#   2  Phi (audio)            sampling_t0.4
#   3  Gold->Tower (text)     beam_search_n5
#   4  Gold->Llama (text)     beam_search_n10
#   5  Gold->Tower (text)     beam_sampling_n5
#
# Each test sample receives N (default 2) random in-context examples drawn from
# translation_data/references.csv, deterministically selected per sample and
# guaranteed not to include the current sample (no test-data leak).
# Luganda (lg) is skipped automatically because references.csv has no LG column.

if [ -z "$1" ]; then
    echo "Usage: $0 <GPU_ID> [--sections 1,2,3,4,5] [--num_icl N]"
    echo ""
    echo "Sections:"
    echo "  1  Phi (audio)        | sampling_t0.2"
    echo "  2  Phi (audio)        | sampling_t0.4"
    echo "  3  Gold->Tower (text) | beam_search_n5"
    echo "  4  Gold->Llama (text) | beam_search_n10"
    echo "  5  Gold->Tower (text) | beam_sampling_n5"
    exit 1
fi

GPU_ID=$1
export CUDA_VISIBLE_DEVICES=$GPU_ID

SECTIONS="1,2,3,4,5"
NUM_ICL=2
shift
while [[ $# -gt 0 ]]; do
    case $1 in
        --sections) SECTIONS="$2"; shift 2 ;;
        --num_icl)  NUM_ICL="$2";  shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

run_section() { [[ ",$SECTIONS," == *",$1,"* ]]; }

cd "$(dirname "$0")/.."

CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

ENV_TOWER="canary_env"
ENV_LLAMA="canary_env"
ENV_PHI="phi_env"

SRC_LANG="en"
PROMPT="standard"
ICL_REFS="../translation_data/references.csv"

# Languages with target-side translations in references.csv (lg has no column).
ALL_LANGS=("zh" "es" "hi" "fr" "de" "it" "sw" "cs" "ar")
declare -A LANG_NAMES=(
    ["zh"]="Mandarin" ["es"]="Spanish" ["hi"]="Hindi" ["fr"]="French"
    ["de"]="German"   ["it"]="Italian" ["sw"]="Swahili" ["cs"]="Czech"
    ["ar"]="Arabic"
)

activate_env() {
    local env_name=$1
    conda activate "$env_name"
    echo "[GPU $GPU_ID] Activated conda env: $env_name (python: $(which python))"
}

make_decode_label() {
    local strategy=$1 param=$2
    case $strategy in
        greedy)        echo "greedy" ;;
        sampling)      echo "sampling_t$param" ;;
        beam_search)   echo "beam_search_n${param%.*}" ;;
        beam_sampling) echo "beam_sampling_n${param%.*}" ;;
        typical)       echo "typical_p$param" ;;
    esac
}

TOTAL=0
CURRENT=0
FAILED=0
SKIPPED=0
FAIL_LOG=""

count_jobs() {
    local n=${#ALL_LANGS[@]}
    TOTAL=0
    run_section 1 && TOTAL=$(( TOTAL + n ))
    run_section 2 && TOTAL=$(( TOTAL + n ))
    run_section 3 && TOTAL=$(( TOTAL + n ))
    run_section 4 && TOTAL=$(( TOTAL + n ))
    run_section 5 && TOTAL=$(( TOTAL + n ))
}

run() {
    local section=$1 model=$2 modality=$3 in_folder=$4 out_folder=$5
    local lang=$6 strategy=$7 param=$8

    CURRENT=$((CURRENT + 1))
    local decode_label
    decode_label=$(make_decode_label "$strategy" "$param")
    local file_label="${decode_label}_icl${NUM_ICL}"
    local ts
    ts=$(date '+%Y-%m-%d %H:%M:%S')

    local output_xml="${out_folder}/${model}/${modality}_${SRC_LANG}_${lang}_${PROMPT}_${model}_${file_label}_output.xml"
    if [ -f "$output_xml" ]; then
        SKIPPED=$((SKIPPED + 1))
        echo ""
        echo "[$ts] [GPU $GPU_ID] [$CURRENT/$TOTAL] [$section] $model | $decode_label+ICL${NUM_ICL} | en->$lang (${LANG_NAMES[$lang]}) | SKIPPED (exists)"
        return
    fi

    echo ""
    echo "[$ts] [GPU $GPU_ID] [$CURRENT/$TOTAL] [$section] $model | $decode_label+ICL${NUM_ICL} | en->$lang (${LANG_NAMES[$lang]})"
    echo "------------------------------------------------------------"

    local decode_args="--decode_strategy $strategy"
    if [ "$param" != "." ]; then
        decode_args="$decode_args --decode_param $param"
    fi

    if ! python main.py \
        --src_lang "$SRC_LANG" \
        --trg_lang "$lang" \
        --modality "$modality" \
        --prompt "$PROMPT" \
        --model "$model" \
        --in_data_folder "$in_folder" \
        --out_folder "$out_folder" \
        --icl \
        --num_icl_examples "$NUM_ICL" \
        --icl_references_csv "$ICL_REFS" \
        $decode_args; then
        FAILED=$((FAILED + 1))
        FAIL_LOG+="  FAILED: [$section] $model | $decode_label+ICL${NUM_ICL} | en->$lang\n"
        echo "[GPU $GPU_ID] *** FAILED *** (continuing)"
    fi
}

count_jobs
echo "============================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [GPU $GPU_ID] ICL sweep ($TOTAL runs, sections: $SECTIONS, num_icl=$NUM_ICL)"
echo "============================================================"

# 1) Phi (audio) | sampling t=0.2
if run_section 1; then
    echo ""
    echo "============================================================"
    echo "[GPU $GPU_ID] SECTION 1/5: Phi (audio) | sampling_t0.2 | ICL"
    echo "============================================================"
    activate_env "$ENV_PHI"
    for lang in "${ALL_LANGS[@]}"; do
        run "speechllm/phi" "phi_multimodal" "audio" \
            "../translation_data/translation_dataset_audio_preprocessed" "generated_output/speechllm" \
            "$lang" "sampling" "0.2"
    done
fi

# 2) Phi (audio) | sampling t=0.4
if run_section 2; then
    echo ""
    echo "============================================================"
    echo "[GPU $GPU_ID] SECTION 2/5: Phi (audio) | sampling_t0.4 | ICL"
    echo "============================================================"
    activate_env "$ENV_PHI"
    for lang in "${ALL_LANGS[@]}"; do
        run "speechllm/phi" "phi_multimodal" "audio" \
            "../translation_data/translation_dataset_audio_preprocessed" "generated_output/speechllm" \
            "$lang" "sampling" "0.4"
    done
fi

# 3) Gold -> Tower (text) | beam_search n=5
if run_section 3; then
    echo ""
    echo "============================================================"
    echo "[GPU $GPU_ID] SECTION 3/5: Gold->Tower | beam_search_n5 | ICL"
    echo "============================================================"
    activate_env "$ENV_TOWER"
    for lang in "${ALL_LANGS[@]}"; do
        run "cascaded/gold-tower" "tower" "text" \
            "generated_output/transcript/gold" "generated_output/gold_cascaded" \
            "$lang" "beam_search" "5"
    done
fi

# 4) Gold -> Llama (text) | beam_search n=10
if run_section 4; then
    echo ""
    echo "============================================================"
    echo "[GPU $GPU_ID] SECTION 4/5: Gold->Llama | beam_search_n10 | ICL"
    echo "============================================================"
    activate_env "$ENV_LLAMA"
    for lang in "${ALL_LANGS[@]}"; do
        run "cascaded/gold-llama" "llama" "text" \
            "generated_output/transcript/gold" "generated_output/gold_cascaded" \
            "$lang" "beam_search" "10"
    done
fi

# 5) Gold -> Tower (text) | beam_sampling n=5
if run_section 5; then
    echo ""
    echo "============================================================"
    echo "[GPU $GPU_ID] SECTION 5/5: Gold->Tower | beam_sampling_n5 | ICL"
    echo "============================================================"
    activate_env "$ENV_TOWER"
    for lang in "${ALL_LANGS[@]}"; do
        run "cascaded/gold-tower" "tower" "text" \
            "generated_output/transcript/gold" "generated_output/gold_cascaded" \
            "$lang" "beam_sampling" "5"
    done
fi

echo ""
echo "============================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [GPU $GPU_ID] Finished: $((TOTAL - FAILED - SKIPPED))/$TOTAL succeeded, $FAILED failed, $SKIPPED skipped"
if [ $FAILED -gt 0 ]; then
    echo ""
    echo "Failed runs:"
    echo -e "$FAIL_LOG"
fi
echo "============================================================"
