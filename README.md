# Uh-Mazing Translations

## Environment Stuff
```
conda create -n disfuent-mt python=3.10 pip ipykernel
conda env export -n disfuent-mt > disfuent-mt-environment.yml
``` 

## Run inference

Cd into the `benchmarking` folder.

Experiments can then be started using the following command:

```bash
python main.py \
  --src_lang [language_code] \ # supported languages below
  --trg_lang [language_code] \ # supported languages below
  --modality [modality] \ # text or audio
  --prompt [prompt_type] \ # standard or disfluency_aware
  --model [model_name] \ # phi_multimodal, qwen_omni, llama, tower, osm, canary
  --in_data_folder your_input_folder_name \ # input folder contains .wav files
  --out_folder generated_output/your_output_folder_name \
  --ablate_disfluencies # if set ablates over different disfluency types, only possible with English text as source
```


## Supported Languages

The script translates from English (`en`) to the following languages, or from the following languages to English.

| Language Code | Language Name |
|---------------|---------------|
| `zh` | Mandarin |
| `es` | Spanish |
| `hi` | Hindi |
| `fr` | French |
| `de` | German |
| `it` | Italian |
| `sw` | Swahili |
| `cs` | Czech |
| `ar` | Arabic |








