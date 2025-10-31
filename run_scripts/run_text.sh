declare -A LANGUAGE_DICT=(
    ["zh"]="Mandarin"
    ["es"]="Spanish"
    ["hu"]="Hungarian"
    ["ro"]="Romanian"
    ["de"]="German"
    ["cs"]="Czech"
    ["vi"]="Vietnamese"
    ["tr"]="Turkish"
    ["it"]="Italian"
)

# Array of prompt types
PROMPTS=("standard" "disfluency_aware")

# Loop through each language and prompt
for lang_code in "${!LANGUAGE_DICT[@]}"; do
    for prompt in "${PROMPTS[@]}"; do
        echo "Running model for ${LANGUAGE_DICT[$lang_code]} (${lang_code}) with prompt: ${prompt}"
        python main.py --lang "$lang_code" --modality text --prompt "$prompt" --model llama --out_folder generated_output/cascaded_whisper
        echo "----------------------------------------------------"
    done
done