#!/bin/bash

# Language dictionary
declare -A LANGUAGE_DICT=( 
    ["zh"]="Mandarin"
    ["es"]="Spanish"
    ["hi"]="Hindi"
    ["fr"]="French"
    ["de"]="German"
    ["it"]="Italian"
    ["cs"]="Czech"
    ["ar"]="Arabic"
    ["sw"]="Swahili"
)

TRG_LANG="en"

# Prompts
PROMPTS=("standard" "disfluency_aware")

# Models for audio modality
MODELS=("llama")

cd "$(dirname "$0")/../../.." || exit 1

IN_DATA_FOLDER=generated_output/transcript/gold

# Loop over all combinations
for lang_code in "${!LANGUAGE_DICT[@]}"; do
    for prompt in "${PROMPTS[@]}"; do
        for model in "${MODELS[@]}"; do
            echo "Running model: $model | Modality: text | Language: ${LANGUAGE_DICT[$lang_code]} ($lang_code) | Prompt: $prompt"
            python main.py --src_lang "$lang_code" --trg_lang $TRG_LANG --modality text --prompt "$prompt" --model "$model" --in_data_folder $IN_DATA_FOLDER --out_folder generated_output/gold_cascaded
            echo "----------------------------------------------------"
        done
    done
done
