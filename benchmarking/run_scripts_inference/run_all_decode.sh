#!/bin/bash

# Usage: ./run_all_decode.sh <GPU_ID> [--sections 3,4]
# Example: ./run_all_decode.sh 5 --sections 3,4

if [ -z "$1" ]; then
    echo "Usage: $0 <GPU_ID> [--sections 1,2,3,4,5,6]"
    echo ""
    echo "Sections:"
    echo "  1  Cascaded canary → tower (text)"
    echo "  2  Cascaded gold-transcript → tower (text)"
    echo "  3  End2end canary (audio)"
    echo "  4  SpeechLLM phi (audio)"
    echo "  5  Cascaded canary → llama (text)"
    echo "  6  Cascaded gold-transcript → llama (text)"
    echo ""
    echo "Example: $0 5 --sections 3,4"
    exit 1
fi

GPU_ID=$1
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Parse --sections (default: all)
SECTIONS="1,2,3,4,5,6"
shift
while [[ $# -gt 0 ]]; do
    case $1 in
        --sections) SECTIONS="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

run_section() { [[ ",$SECTIONS," == *",$1,"* ]]; }

# Run everything from the benchmarking directory
cd "$(dirname "$0")/.."

# --- Conda environments per model ----------------------------------------
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

ENV_TOWER="canary_env"
ENV_CANARY="canary_env"
ENV_PHI="phi_env"
ENV_LLAMA="canary_env"

SRC_LANG="en"
PROMPTS=("standard" "disfluency_aware")

# --- Languages -----------------------------------------------------------
ALL_LANGS=("zh" "es" "hi" "fr" "de" "it" "sw" "cs" "ar" "lg")
declare -A LANG_NAMES=(
    ["zh"]="Mandarin" ["es"]="Spanish" ["hi"]="Hindi" ["fr"]="French"
    ["de"]="German"   ["it"]="Italian" ["sw"]="Swahili" ["cs"]="Czech"
    ["ar"]="Arabic"   ["lg"]="Luganda"
)

# Canary does not support zh, hi, sw, ar
CANARY_LANGS=("es" "fr" "de" "it" "cs")

# --- Decode configs: "strategy param" (.=no param) -----------------------
ALL_DECODE_CONFIGS=(
    "greedy ."
    "sampling 0.2"
    "sampling 0.4"
    "sampling 0.6"
    "sampling 0.8"
    "sampling 1.0"
    "beam_search 5"
    "beam_search 10"
    "beam_search 20"
    "beam_sampling 5"
    "beam_sampling 10"
    "beam_sampling 20"
    "typical 0.2"
    "typical 0.5"
    "typical 0.9"
)

# NeMo (Canary) supports greedy, sampling, and beam_search (no beam_sampling or typical)
NEMO_DECODE_CONFIGS=(
    "greedy ."
    "sampling 0.2"
    "sampling 0.4"
    "sampling 0.6"
    "sampling 0.8"
    "sampling 1.0"
    "beam_search 5"
    "beam_search 10"
    "beam_search 20"
)

# --- Helpers --------------------------------------------------------------

make_label() {
    local strategy=$1 param=$2
    case $strategy in
        greedy)        echo "greedy" ;;
        sampling)      echo "sampling t=$param" ;;
        beam_search)   echo "beam_search n=$param" ;;
        beam_sampling) echo "beam_sampling n=$param" ;;
        typical)       echo "typical p=$param" ;;
    esac
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

activate_env() {
    local env_name=$1
    conda activate "$env_name"
    echo "[GPU $GPU_ID] Activated conda env: $env_name (python: $(which python))"
}

TOTAL=0
CURRENT=0
FAILED=0
SKIPPED=0
FAIL_LOG=""

count_jobs() {
    local n_prompts=${#PROMPTS[@]}
    TOTAL=0
    run_section 1 && TOTAL=$(( TOTAL + 15*10*n_prompts ))
    run_section 2 && TOTAL=$(( TOTAL + 15*10*n_prompts ))
    run_section 3 && TOTAL=$(( TOTAL + 9*5*n_prompts ))
    run_section 4 && TOTAL=$(( TOTAL + 15*10*n_prompts ))
    run_section 5 && TOTAL=$(( TOTAL + 15*10*n_prompts ))
    run_section 6 && TOTAL=$(( TOTAL + 15*10*n_prompts ))
}

run() {
    local section=$1 model=$2 modality=$3 in_folder=$4 out_folder=$5
    local lang=$6 strategy=$7 param=$8 prompt=$9

    CURRENT=$((CURRENT + 1))
    local label
    label=$(make_label "$strategy" "$param")
    local decode_label
    decode_label=$(make_decode_label "$strategy" "$param")
    local ts
    ts=$(date '+%Y-%m-%d %H:%M:%S')

    # Skip if output XML already exists
    local output_xml="${out_folder}/${model}/${modality}_${SRC_LANG}_${lang}_${prompt}_${model}_${decode_label}_output.xml"
    if [ -f "$output_xml" ]; then
        SKIPPED=$((SKIPPED + 1))
        echo ""
        echo "[$ts] [GPU $GPU_ID] [$CURRENT/$TOTAL] [$section] $model | $label | en→$lang (${LANG_NAMES[$lang]}) | $prompt | SKIPPED (exists)"
        return
    fi

    echo ""
    echo "[$ts] [GPU $GPU_ID] [$CURRENT/$TOTAL] [$section] $model | $label | en→$lang (${LANG_NAMES[$lang]}) | $prompt"
    echo "------------------------------------------------------------"

    local decode_args="--decode_strategy $strategy"
    if [ "$param" != "." ]; then
        decode_args="$decode_args --decode_param $param"
    fi

    if ! python main.py \
        --src_lang "$SRC_LANG" \
        --trg_lang "$lang" \
        --modality "$modality" \
        --prompt "$prompt" \
        --model "$model" \
        --in_data_folder "$in_folder" \
        --out_folder "$out_folder" \
        $decode_args; then
        FAILED=$((FAILED + 1))
        FAIL_LOG+="  FAILED: [$section] $model | $label | en→$lang | $prompt\n"
        echo "[GPU $GPU_ID] *** FAILED *** (continuing)"
    fi
}

# --- Main -----------------------------------------------------------------

count_jobs

echo "============================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [GPU $GPU_ID] Starting decode parameter sweep ($TOTAL runs, sections: $SECTIONS)"
echo "============================================================"

# --- 1) Cascaded canary-tower (text) -------------------------------------
if run_section 1; then
    echo ""
    echo "============================================================"
    echo "[GPU $GPU_ID] SECTION 1/6: Cascaded canary → tower"
    echo "============================================================"
    activate_env "$ENV_TOWER"
    for prompt in "${PROMPTS[@]}"; do
        for config in "${ALL_DECODE_CONFIGS[@]}"; do
            read -r strategy param <<< "$config"
            for lang in "${ALL_LANGS[@]}"; do
                run "cascaded/canary-tower" "tower" "text" \
                    "generated_output/transcript/canary" "generated_output/canary_cascaded" \
                    "$lang" "$strategy" "$param" "$prompt"
            done
        done
    done
fi

# --- 2) Cascaded gold-transcript → tower (text) --------------------------
if run_section 2; then
    echo ""
    echo "============================================================"
    echo "[GPU $GPU_ID] SECTION 2/6: Cascaded gold-transcript → tower"
    echo "============================================================"
    activate_env "$ENV_TOWER"
    for prompt in "${PROMPTS[@]}"; do
        for config in "${ALL_DECODE_CONFIGS[@]}"; do
            read -r strategy param <<< "$config"
            for lang in "${ALL_LANGS[@]}"; do
                run "cascaded/gold-tower" "tower" "text" \
                    "generated_output/transcript/gold" "generated_output/gold_cascaded" \
                    "$lang" "$strategy" "$param" "$prompt"
            done
        done
    done
fi

# --- 3) End2end canary (audio) --------------------------------------------
if run_section 3; then
    echo ""
    echo "============================================================"
    echo "[GPU $GPU_ID] SECTION 3/6: End2end canary"
    echo "============================================================"
    activate_env "$ENV_CANARY"
    for prompt in "${PROMPTS[@]}"; do
        for config in "${NEMO_DECODE_CONFIGS[@]}"; do
            read -r strategy param <<< "$config"
            for lang in "${CANARY_LANGS[@]}"; do
                run "end2end/canary" "canary" "audio" \
                    "../translation_data/translation_dataset_audio_preprocessed" "generated_output/end2end" \
                    "$lang" "$strategy" "$param" "$prompt"
            done
        done
    done
fi

# --- 4) SpeechLLM phi (audio) --------------------------------------------
if run_section 4; then
    echo ""
    echo "============================================================"
    echo "[GPU $GPU_ID] SECTION 4/6: SpeechLLM phi"
    echo "============================================================"
    activate_env "$ENV_PHI"
    for prompt in "${PROMPTS[@]}"; do
        for config in "${ALL_DECODE_CONFIGS[@]}"; do
            read -r strategy param <<< "$config"
            for lang in "${ALL_LANGS[@]}"; do
                run "speechllm/phi" "phi_multimodal" "audio" \
                    "../translation_data/translation_dataset_audio_preprocessed" "generated_output/speechllm" \
                    "$lang" "$strategy" "$param" "$prompt"
            done
        done
    done
fi

# --- 5) Cascaded canary-llama (text) -------------------------------------
if run_section 5; then
    echo ""
    echo "============================================================"
    echo "[GPU $GPU_ID] SECTION 5/6: Cascaded canary → llama"
    echo "============================================================"
    activate_env "$ENV_LLAMA"
    for prompt in "${PROMPTS[@]}"; do
        for config in "${ALL_DECODE_CONFIGS[@]}"; do
            read -r strategy param <<< "$config"
            for lang in "${ALL_LANGS[@]}"; do
                run "cascaded/canary-llama" "llama" "text" \
                    "generated_output/transcript/canary" "generated_output/canary_cascaded" \
                    "$lang" "$strategy" "$param" "$prompt"
            done
        done
    done
fi

# --- 6) Cascaded gold-transcript → llama (text) --------------------------
if run_section 6; then
    echo ""
    echo "============================================================"
    echo "[GPU $GPU_ID] SECTION 6/6: Cascaded gold-transcript → llama"
    echo "============================================================"
    activate_env "$ENV_LLAMA"
    for prompt in "${PROMPTS[@]}"; do
        for config in "${ALL_DECODE_CONFIGS[@]}"; do
            read -r strategy param <<< "$config"
            for lang in "${ALL_LANGS[@]}"; do
                run "cascaded/gold-llama" "llama" "text" \
                    "generated_output/transcript/gold" "generated_output/gold_cascaded" \
                    "$lang" "$strategy" "$param" "$prompt"
            done
        done
    done
fi

# --- Summary --------------------------------------------------------------
echo ""
echo "============================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [GPU $GPU_ID] Finished: $((TOTAL - FAILED - SKIPPED))/$TOTAL succeeded, $FAILED failed, $SKIPPED skipped"
if [ $FAILED -gt 0 ]; then
    echo ""
    echo "Failed runs:"
    echo -e "$FAIL_LOG"
fi
echo "============================================================"
