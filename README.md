# disfuent-mt

## env stuff
```
conda create -n disfuent-mt python=3.10 pip ipykernel
conda env export -n disfuent-mt > disfuent-mt-environment.yml
``` 

## how to run inference

Each experiment runs with the following parameters:
```bash
python main.py \
  --src_lang en \
  --trg_lang [language_code] \ # supported languages below
  --modality [modality] \ # text or audio
  --prompt [prompt_type] \ # standard or disfluency_aware
  --model [model_name] \ # phi_multimodal, qwen_omni, llama, tower, whisper, canary
  --in_data_folder your_input_folder_name \ # input folder contains .wav files
  --out_folder generated_output/your_output_folder_name
```


## Supported Languages

The script translates from English (`en`) to the following languages:

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
| `lg` | Luganda |







