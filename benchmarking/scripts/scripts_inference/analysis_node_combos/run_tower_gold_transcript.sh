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
    ["cs"]="Czech"
)

SRC_LANG="en"

# Prompts
PROMPTS=("standard")

# Models for audio modality
MODELS=("tower")

IN_DATA_FOLDER=../translation_data/

# Loop over all combinations
for lang_code in "${!LANGUAGE_DICT[@]}"; do
    for prompt in "${PROMPTS[@]}"; do
        for model in "${MODELS[@]}"; do
            echo "Running model: $model | Modality: text | Language: ${LANGUAGE_DICT[$lang_code]} ($lang_code) | Prompt: $prompt"
            python src/main.py --src_lang $SRC_LANG --trg_lang "$lang_code" --modality text --prompt "$prompt" --model "$model" --in_data_folder $IN_DATA_FOLDER --out_folder generated_output/analysis_node_combos --ablate_disfluencies
            echo "----------------------------------------------------"
        done
    done
done
