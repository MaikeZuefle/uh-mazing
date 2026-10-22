#Requirements are
#librosa
#torch
#espnet
#espnet_model_zoo




import logging

def load_model():
    logging.disable(logging.CRITICAL)
    from espnet2.bin.s2t_inference_ctc import Speech2TextGreedySearch
    import torch

    model = Speech2TextGreedySearch.from_pretrained(
        "espnet/owsm_ctc_v4_1B",
        device='cuda' if torch.cuda.is_available() else 'cpu',
        #dtype="bfloat16",
        use_flash_attn=False,
        generate_interctc_outputs=False,
    )
    return model

def generate(model, prompt, example_path, modality, src_lang, trg_lang, gen_kwargs=None):
    from langcodes import Language
    import librosa

    if modality != "audio":
        raise NotImplementedError("OWSM only supports audio in this implementation!")

    context_len_in_secs = 4   # left and right context when doing buffered inference, default parameter in HF repo

    src = Language.get(src_lang).to_alpha3()
    tgt = Language.get(trg_lang).to_alpha3()
    model.lang_sym=f'<{src}>'
    model.task_sym = "<asr>" if src == tgt else f"<st_{tgt}>"


    speech, rate = librosa.load(example_path, sr=16000)
    long_form_add_threshold = 0.2

    if (speech.shape[0] / rate) < (30 + long_form_add_threshold):
        speech = librosa.util.fix_length(speech, size=(16000 * 30))
        text = model(speech)[0][-2]
    else:
        text = model.decode_long_batched_buffered(
            speech,
            batch_size=1,
            context_len_in_secs=context_len_in_secs,
        )

    return text