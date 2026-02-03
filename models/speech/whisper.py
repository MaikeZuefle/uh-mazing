whisper_langs = {"en":"english", "de": "german", "zh": "mandarin", "es":"spanish", "hi": "hindi", "fr": "french", "it": "italian", "sw": "swahili", 
"cs": "czech", "ar": "arabic", "lg": "luganda"}


def load_model():
    from transformers import pipeline

    # Initialize the Whisper model (this may take a bit the first time)
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3",
        device_map="auto"  # uses GPU if available
    )
    return pipe

def generate(pipe, prompt, example_path, modality, src_lang, trg_lang):
    if modality != "audio":
        raise NotImplementedError("Phi only supports audio in this implementation!")
    
    if src_lang == trg_lang:
        response = pipe(example_path, return_timestamps=True)["text"]
    else:
        response = pipe(example_path,  return_timestamps=True, generate_kwargs={"language": whisper_langs[trg_lang], "task": "translate"})["text"]

    return response