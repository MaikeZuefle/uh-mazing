from utils import read_txt_file

def load_model():
    from transformers import pipeline
    pipe = pipeline("text-generation", model="Unbabel/Tower-Plus-9B", device_map="auto")
    return pipe

def generate(pipe, prompt, example_path, modality, src_lang=None, trg_lang=None, gen_kwargs=None):

    if modality != "text":
        raise NotImplementedError("Tower only supports text!")

    example = example_path

    messages = [{"role": "user", "content": f"{example}\n{prompt}"}]

    kwargs = {"max_new_tokens": 1024, "do_sample": False}
    if gen_kwargs:
        kwargs.update(gen_kwargs)

    outputs = pipe(messages, **kwargs)

    response = outputs[0]["generated_text"][-1]["content"].strip()

    return response



