from utils import read_txt_file

def load_model():
    from transformers import pipeline
    pipe = pipeline("text-generation", model="Unbabel/Tower-Plus-9B", device_map="auto")
    return pipe

def generate(pipe, prompt, example_path, modality, src_lang=None, trg_lang=None):

    if modality != "text":
        raise NotImplementedError("Tower only supports text!")
    
    example = example_path
        
    messages = [{"role": "user", "content": f"{example}\n{prompt}"}]
    outputs = pipe(messages, max_new_tokens=4096, do_sample=False)

    response = outputs[0]["generated_text"][-1]["content"].strip()

    return response



