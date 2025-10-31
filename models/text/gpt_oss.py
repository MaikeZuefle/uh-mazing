#from transformers import pipeline
from utils import read_txt_file


def load_model():
    model_id = "openai/gpt-oss-20b"
    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype="auto",
        device_map="auto",
    )

    return pipe

def generate(pipe, prompt, example_path, modality):

    if modality != "text":
        raise NotImplementedError("GPT-oss only supports text!")
    
    example = read_txt_file(example_path)
        
    messages = [{"role": "user", "content": f"{example}\n{prompt}"}]
    outputs = pipe(messages, max_new_tokens=4096)

    response = outputs[0]["generated_text"][-1]["content"].split("assistantfinal")[-1]

    return response





