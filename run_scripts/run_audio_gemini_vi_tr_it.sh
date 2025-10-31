#!/bin/bash

# Language dictionary
declare -A LANGUAGE_DICT=(
    # ["zh"]="Mandarin"
    # ["es"]="Spanish"
    # ["hu"]="Hungarian"
    # ["ro"]="Romanian"
    # ["de"]="German"
    # ["cs"]="Czech"
    # ["vi"]="Vietnamese"
    ["tr"]="Turkish"
    # ["it"]="Italian"
)

# Prompts
PROMPTS=("standard") #"standard"  disfluency_aware

# Models for audio modality
MODELS=("gemini")

# Loop over all combinations
for lang_code in "${!LANGUAGE_DICT[@]}"; do
    for prompt in "${PROMPTS[@]}"; do
        for model in "${MODELS[@]}"; do
            echo "Running model: $model | Modality: audio | Language: ${LANGUAGE_DICT[$lang_code]} ($lang_code) | Prompt: $prompt"
            python main.py --lang "$lang_code" --modality audio --prompt "$prompt" --model "$model" --out_folder generated_output/end_to_end
            echo "----------------------------------------------------"
        done
    done
done
