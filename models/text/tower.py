#from transformers import pipeline
from utils import read_txt_file


def load_model():
    pipe = pipeline("text-generation", model="Unbabel/Tower-Plus-9B", device_map="auto")
    return pipe

def generate(pipe, prompt, example_path, modality):

    if modality != "text":
        raise NotImplementedError("Tower only supports text!")
    
    example = read_txt_file(example_path)
        
    messages = [{"role": "user", "content": f"{example}\n{prompt}"}]
    outputs = pipe(messages, max_new_tokens=4096, do_sample=False)

    response = outputs[0]["generated_text"][-1]["content"].strip()

    return response



