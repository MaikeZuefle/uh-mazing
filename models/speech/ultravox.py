import transformers


def load_model():
    pipe = transformers.pipeline(model='fixie-ai/ultravox-v0_5-llama-3_2-1b', trust_remote_code=True)
    return pipe


def generate(model, prompt, example_path, modality):
    import librosa

    if modality != "audio":
        raise NotImplementedError("Ultravox only supports audio!")
    
    audio, sr = librosa.load(example_path, sr=16000)

    turns = [
    {"role": "system", "content": "You are a friendly and helpful character. You love to answer questions for people. Only return the answer requested. Do not include any explanation or introductions."},
    {"role": "user", "content": f"<|audio|>{prompt}"},
    ]

    response = model({'audio': audio, 'turns': turns, 'sampling_rate': sr}, max_new_tokens=4096)
    return response
