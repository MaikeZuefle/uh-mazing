from omegaconf import OmegaConf

def load_model():
    from nemo.collections.asr.models import ASRModel
    asr_ast_model = ASRModel.from_pretrained(model_name="nvidia/canary-1b-v2")
    return asr_ast_model


def _apply_nemo_decoding(model, gen_kwargs):
    """Configure NeMo decoding strategy. Supports greedy, temperature sampling, and beam search."""

    num_beams = gen_kwargs.get("num_beams", 1)
    do_sample = gen_kwargs.get("do_sample", False)
    temperature = gen_kwargs.get("temperature")

    if num_beams > 1:
        if do_sample:
            raise NotImplementedError("Canary does not support beam sampling.")
        decoding_cfg = OmegaConf.create({
            "strategy": "beam",
            "beam": {"beam_size": num_beams, "len_pen": 1.0},
        })
    elif do_sample and temperature is not None:
        decoding_cfg = OmegaConf.create({
            "strategy": "greedy",
            "greedy": {"temperature": float(temperature), "n_samples": 1},
        })
    else:
        decoding_cfg = OmegaConf.create({"strategy": "greedy_batch"})
    model.change_decoding_strategy(decoding_cfg)


def generate(asr_ast_model, prompt, example_path, modality, src_lang, trg_lang, gen_kwargs=None):
    import torchaudio
    if modality != "audio":
        raise NotImplementedError("Canary only supports audio in this implementation!")

    if gen_kwargs:
        _apply_nemo_decoding(asr_ast_model, gen_kwargs)

    response = asr_ast_model.transcribe(example_path, source_lang=src_lang, target_lang=trg_lang)[0].text
    return response