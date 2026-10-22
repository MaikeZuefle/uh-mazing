
import soundfile as sf
from utils import read_txt_file, format_icl_prefix

def load_model():
    from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
    model_path = "microsoft/Phi-4-multimodal-instruct"
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        device_map="cuda", 
        torch_dtype="auto", 
        trust_remote_code=True,
        _attn_implementation='flash_attention_2',
    ).cuda()

    generation_config = GenerationConfig.from_pretrained(model_path)

    return model, processor, generation_config

def generate(model_processor_config, prompt, example_path, modality, src_lang=None, trg_lang=None, gen_kwargs=None, icl_examples=None):
    if modality != "audio":
        raise NotImplementedError("Phi only supports audio in this implementation!")

    model, processor, generation_config = model_processor_config

    # prompts
    user_prompt = "<|user|>"
    assistant_prompt = "<|assistant|>"
    prompt_suffix = "<|end|>"

    icl_prefix = format_icl_prefix(icl_examples or [], src_lang, trg_lang)

    if example_path.endswith(".wav"):
        prompt = f"{user_prompt}{icl_prefix}<|audio_1|>{prompt}{prompt_suffix}{assistant_prompt}"

        # Open audio file
        audio, samplerate = sf.read(example_path)

        # Process with the model
        inputs = processor(text=prompt, audios=[(audio, samplerate)], return_tensors='pt').to('cuda:0')

    else:
        example = read_txt_file(example_path)
        prompt = f"{user_prompt}{icl_prefix}{example}\n{prompt}{prompt_suffix}{assistant_prompt}"
        inputs = processor(text=prompt, audios=None, return_tensors='pt').to('cuda:0')
        

    extra_kwargs = gen_kwargs if gen_kwargs else {}
    generate_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        generation_config=generation_config,
        **extra_kwargs,
    )
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return response