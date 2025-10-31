#from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import read_txt_file

def load_model():
    model_id = "CohereLabs/aya-expanse-8b"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to("cuda")
    return model, tokenizer

def generate(model_tokenizer, prompt, example_path, modality):
    model, tokenizer = model_tokenizer
    if modality != "text":
        raise NotImplementedError("Aya only supports text!")
    
    example = read_txt_file(example_path)
        
    messages = [{"role": "user", "content": f"{example}\n{prompt}"}]
    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
    input_len = input_ids.shape[-1]
    gen_tokens = model.generate(
        input_ids, 
        max_new_tokens=4096, 
        do_sample=True, 
        temperature=0.3,
        )


    response = tokenizer.decode(gen_tokens[0][input_len:]).replace("<|END_OF_TURN_TOKEN|>", "")
    return response



