from utils import read_txt_file
#import torch

def load_model():
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration
    model_id = "google/gemma-3-12b-it"

    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id, device_map="auto"
    ).eval()

    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor

def generate(model_processor, prompt, example_path, modality):
    model, processor = model_processor
    if modality != "text":
        raise NotImplementedError("Gemma only supports text!")
    
    example = read_txt_file(example_path)
        
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant. Only return the answer requested. Do not include any explanation or introductions."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text":f"{example}\n{prompt}\n"}
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=4096, do_sample=False)
        generation = generation[0][input_len:]

    response = processor.decode(generation, skip_special_tokens=True)
    return response



