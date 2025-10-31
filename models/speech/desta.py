from utils import read_txt_file
from transformers import AutoModel
import torch

def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained("DeSTA-ntu/DeSTA2-8B-beta", trust_remote_code=True).to(device)
    return model


def generate(model, prompt, example_path, modality):

    if modality != "audio":
        raise NotImplementedError("Desta only supports audio!")
    
    messages = [
                {"role": "system", "content": "You are a helpful voice assistant. Only return the answer requested. Do not include any explanation or introductions."},
                {"role": "audio", "content": example_path},
                {"role": "user", "content": prompt}
            ]

    generated_ids = model.chat(
        messages, 
        max_new_tokens=4096, 
        do_sample=True, 
        temperature=0.6, 
        top_p=0.9
    )

    response = model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

