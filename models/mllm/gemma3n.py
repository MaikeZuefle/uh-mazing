import torch
torch._dynamo.config.cache_size_limit = 128  
torch._dynamo.disable()
from utils import read_txt_file

def load_model():
    from transformers import AutoProcessor, Gemma3nForConditionalGeneration
    model_id = "google/gemma-3n-e2b-it"

    model = Gemma3nForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16,).eval()
    model = model.to("cuda")

    processor = AutoProcessor.from_pretrained(model_id)

    return model, processor

def generate(model_processor, prompt, example_path, modality):
    model, processor = model_processor 

    if modality == "mllm":
        user_conv_content = [{"type": "video", "video": example_path}, {"type": "audio", "audio": example_path}, {"type": "text", "text": prompt}]

    elif modality == "video":
        user_conv_content = [{"type": "video", "video": example_path},  {"type": "text", "text": prompt}, ]

    elif modality == "audio":
        user_conv_content = [{"type": "audio", "audio": example_path},  {"type": "text", "text": prompt}, ]

    elif modality == "text":
        example = read_txt_file(example_path)
        user_conv_content = [{"type": "text", "text": f"{example}\n{prompt}\n"}, ]

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant.  Only return the answer requested. Do not include any explanation or introductions."}]
        },
        {
            "role": "user",
            "content": user_conv_content
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=4096, do_sample=False)
        generation = generation[0][input_len:]

    response = processor.decode(generation, skip_special_tokens=True)
    return response

