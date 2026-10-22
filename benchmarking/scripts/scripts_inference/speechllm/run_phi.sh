#!/bin/bash

cd "$(dirname "$0")/../../.." || exit 1

# Language dictionary
declare -A LANGUAGE_DICT=( 
    ["zh"]="Mandarin"
    ["es"]="Spanish"
    ["hi"]="Hindi"
    ["fr"]="French"
    ["de"]="German"
    ["it"]="Italian"
    ["sw"]="Swahili"
    ["cs"]="Czech"
    ["ar"]="Arabic"
    ["lg"]="Luganda"
)

SRC_LANG="en"

# Prompts
PROMPTS=("standard" "disfluency_aware")

# Models for audio modality
MODELS=("phi_multimodal")

IN_DATA_FOLDER="../translation_data/translation_dataset_audio_preprocessed"

# Loop over all combinations
for lang_code in "${!LANGUAGE_DICT[@]}"; do
    for prompt in "${PROMPTS[@]}"; do
        for model in "${MODELS[@]}"; do
            echo "Running model: $model | Modality: audio | Language: ${LANGUAGE_DICT[$lang_code]} ($lang_code) | Prompt: $prompt"
            python src/main.py --src_lang $SRC_LANG --trg_lang "$lang_code" --modality audio --prompt "$prompt" --model "$model" --in_data_folder $IN_DATA_FOLDER --out_folder generated_output/speechllm
            echo "----------------------------------------------------"
        done
    done
done
