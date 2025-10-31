#import transformers
from utils import read_txt_file

def load_model():
    pipeline = transformers.pipeline(
        "text-generation",
        model="microsoft/phi-4",
        model_kwargs={"torch_dtype": "auto"},
        device_map="auto",
    )
    return pipeline

def generate(pipeline, prompt, example_path, modality):

    if modality != "text":
        raise NotImplementedError("Phi4 only supports text!")
    
    example = read_txt_file(example_path)
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Only return the answer requested. Do not include any explanation or introductions."},
        {"role": "user", "content": f"{example}\n{prompt}"},
    ]
    
    response = pipeline(messages, max_new_tokens=4096)[0]["generated_text"][-1]["content"]
    return response

