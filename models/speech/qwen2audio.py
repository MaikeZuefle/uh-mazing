

def load_model():
    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
    model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", device_map="auto")


    return model, processor


def generate(model_processor, prompt, example_path, modality):
    import librosa
    model, processor = model_processor
    if modality != "audio":
        raise NotImplementedError("Qwen2 only supports audio!")

    conversation =     [
        {"role": "user", "content": [
            {"type": "audio", "audio": example_path},
            {"type": "text", "text": prompt},
        ]},
    ]


    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios = []
    for message in conversation:
        if isinstance(message["content"], list):
            for ele in message["content"]:
                if ele["type"] == "audio":
                    audios.append(librosa.load(
                        example_path, 
                        sr=processor.feature_extractor.sampling_rate)[0]
                    )

    inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True).to(model.device)
    inputs.input_ids = inputs.input_ids.to("cuda")

    generate_ids = model.generate(**inputs, max_length=4096)
    generate_ids = generate_ids[:, inputs.input_ids.size(1):]

    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return response

