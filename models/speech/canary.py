def load_model():
    from nemo.collections.asr.models import ASRModel
    asr_ast_model = ASRModel.from_pretrained(model_name="nvidia/canary-1b-v2")
    return asr_ast_model

def generate(asr_ast_model, prompt, example_path, modality, src_lang, trg_lang):
    import torchaudio
    if modality != "audio":
        raise NotImplementedError("Canary only supports audio in this implementation!")
    
    response = asr_ast_model.transcribe(example_path, source_lang=src_lang, target_lang=trg_lang)[0].text
    return response