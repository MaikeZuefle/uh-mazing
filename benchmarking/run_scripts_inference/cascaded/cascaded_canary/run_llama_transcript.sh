#!/bin/bash

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
MODELS=("llama")

IN_DATA_FOLDER=generated_output/transcript/canary

# Loop over all combinations
for lang_code in "${!LANGUAGE_DICT[@]}"; do
    for prompt in "${PROMPTS[@]}"; do
        for model in "${MODELS[@]}"; do
            echo "Running model: $model | Modality: text | Language: ${LANGUAGE_DICT[$lang_code]} ($lang_code) | Prompt: $prompt"
            python main.py --src_lang $SRC_LANG --trg_lang "$lang_code" --modality text --prompt "$prompt" --model "$model" --in_data_folder $IN_DATA_FOLDER --out_folder generated_output/canary_cascaded
            echo "----------------------------------------------------"
        done
    done
done
