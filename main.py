import argparse
import logging
import os
from tqdm import tqdm
from transformers import set_seed

# import multimodal models
from models.mllm.gemini import generate as generate_gemini
from models.mllm.gemini import load_model as load_gemini
from models.mllm.gemma3n import generate as generate_gemma3n
from models.mllm.gemma3n import load_model as load_gemma3n
from models.mllm.ming_lite_omni import generate as generate_ming_lite_omni
from models.mllm.ming_lite_omni import load_model as load_ming_lite_omni
from models.mllm.minicpm import generate as generate_minicpm
from models.mllm.minicpm import load_model as load_minicpm
from models.mllm.ola import generate as generate_ola
from models.mllm.ola import load_model as load_ola
from models.mllm.qwen_omni import generate as generate_qwen_omni
from models.mllm.qwen_omni import load_model as load_qwen_omni

# import speech models
from models.speech.desta import generate as generate_desta
from models.speech.desta import load_model as load_desta
from models.speech.granite import generate as generate_granite
from models.speech.granite import load_model as load_granite
from models.speech.phi_multimodal import generate as generate_phi_multimodal
from models.speech.phi_multimodal import load_model as load_phi_multimodal
from models.speech.qwen2audio import generate as generate_qwen2audio
from models.speech.qwen2audio import load_model as load_qwen2audio
from models.speech.ultravox import generate as generate_ultravox
from models.speech.ultravox import load_model as load_ultravox
from models.text.aya import generate as generate_aya
from models.text.aya import load_model as load_aya
from models.text.gemma import generate as generate_gemma
from models.text.gemma import load_model as load_gemma
from models.text.gpt_oss import generate as generate_gpt_oss
from models.text.gpt_oss import load_model as load_gpt_oss
from models.text.llama import generate as generate_llama
from models.text.llama import load_model as load_llama

# import text models
from models.text.phi4 import generate as generate_phi
from models.text.phi4 import load_model as load_phi
from models.text.qwen3 import generate as generate_qwen3
from models.text.qwen3 import load_model as load_qwen3
from models.text.tower import generate as generate_tower
from models.text.tower import load_model as load_tower


from utils import set_up_dirs, set_up_logging, write_to_xml, LANGUAGE_DICT

set_seed(42)


def load_model(model_name):
    # text models
    if model_name == "phi4":
        model = load_phi()
        generate_func = generate_phi
    elif model_name == "gemma":
        model = load_gemma()
        generate_func = generate_gemma
    elif model_name == "qwen3":
        model = load_qwen3()
        generate_func = generate_qwen3
    elif model_name == "llama":
        model = load_llama()
        generate_func = generate_llama
    elif model_name == "aya":
        model = load_aya()
        generate_func = generate_aya
    elif model_name == "qwen3":
        model = load_qwen3()
        generate_func = generate_qwen3
    elif model_name == "tower":
        model = load_tower()
        generate_func = generate_tower
    elif model_name == "gpt_oss":
        model = load_gpt_oss()
        generate_func = generate_gpt_oss

    # speech models
    elif model_name == "qwen2audio":
        model = load_qwen2audio()
        generate_func = generate_qwen2audio
    elif model_name == "phi_multimodal":
        model = load_phi_multimodal()
        generate_func = generate_phi_multimodal
    elif model_name == "desta":
        model = load_desta()
        generate_func = generate_desta
    elif model_name == "ultravox":
        model = load_ultravox()
        generate_func = generate_ultravox
    elif model_name == "granite":
        model = load_granite()
        generate_func = generate_granite

    # mllm models
    elif model_name == "qwen_omni":
        model = load_qwen_omni()
        generate_func = generate_qwen_omni
    elif model_name == "ming_lite_omni":
        model = load_ming_lite_omni()
        generate_func = generate_ming_lite_omni
    elif model_name == "minicpm":
        model = load_minicpm()
        generate_func = generate_minicpm
    elif model_name == "ola":
        model = load_ola()
        generate_func = generate_ola
    elif model_name == "gemma3n":
        model = load_gemma3n()
        generate_func = generate_gemma3n
    elif model_name == "gemini":
        model = load_gemini()
        generate_func = generate_gemini

    else:
        raise NotImplementedError(f"Model {model_name} currently not supported!")
    return model, generate_func


def main(in_data_folder, out_folder, model, lang, modality, prompt):

    # Setting up folders and logging
    output_file_path = set_up_dirs(
        in_data_folder, out_folder, model, lang, modality, prompt
    )
    set_up_logging(output_file_path)

    logging.info("Welcome!")
    logging.info(
        f"Modality: {modality}, Language: {lang}, Prompt: {prompt}, Model: {model}"
    )
    logging.info(f"Input folder: {in_data_folder}")
    logging.info(f"Output XML: {output_file_path}")

    logging.info(f"Loading Data.")
    wavs = []
    for root, dirs, files in os.walk(in_data_folder):
        for file in files:
            if modality == "audio":
                if file.endswith(".wav"):
                    wavs.append(os.path.join(root, file))
            elif modality == "text":
                if file.endswith(".txt"):
                    wavs.append(os.path.join(root, file))
            else:
                raise NotImplementedError
    logging.info(f"Loading Model.")
    model_instance, generate = load_model(model)

    logging.info(f"Starting Output Generation.")
    outputs = []
    for sample_id, example_path in enumerate(tqdm(wavs, desc="Generating Outputs")):
        if modality == "audio":
            if prompt == "standard":
                instruction = f"Listen to the following audio and translate the speech into {LANGUAGE_DICT[lang]} text."
            elif prompt == "disfluency_aware":
                instruction = f"Listen to the following audio and translate the speech into {LANGUAGE_DICT[lang]} text, " \
                "keeping any disfluencies (such as 'um', 'uh', repetitions, and hesitations) in the translation."
            else:
                raise NotImplementedError
        elif modality == "text":
            if prompt == "standard":
                instruction = f"Translate the transcribed speech into {LANGUAGE_DICT[lang]} text."
            elif prompt == "disfluency_aware":
                instruction = f"Translate the transcribed speech into {LANGUAGE_DICT[lang]} text, " \
                "keeping any disfluencies (such as 'um', 'uh', repetitions, and hesitations) in the translation."
            else:
                raise NotImplementedError
        else: 
            raise NotImplementedError
        
        output = generate(model_instance, instruction, example_path, modality).strip()
        outputs.append((sample_id, output))

    logging.info(f"Writing Outputs to XML file {output_file_path}.")
    write_to_xml(outputs, lang=lang, output_file=output_file_path)
    logging.info("XML written successfully")
    logging.info("All done.")


if __name__ == "__main__":
    LANGS = list(LANGUAGE_DICT.keys())
    MODALITIES = ["text", "audio", "mllm"]
    PROMPT = ["standard", "disfluency_aware"]
    MODELS = [
        "phi_multimodal",
        "desta",
        "ultravox",
        "phi4",
        "gemma",
        "aya",
        "llama",
        "qwen3",
        "tower",
        "qwen2audio",
        "minicpm",
        "gemma3n",
        "granite",
        "ming_lite_omni",
        "ola",
        "qwen_omni",
        "gpt_oss",
        "gemini",
    ]

    parser = argparse.ArgumentParser(description="Process MCIF data.")
    parser.add_argument(
        "--lang", choices=LANGS, default=LANGS[0], help="Language to process"
    )
    parser.add_argument(
        "--modality", choices=MODALITIES, default=MODALITIES[0], help="Modality type"
    )
    parser.add_argument(
        "--prompt", choices=PROMPT, default=PROMPT[0], help="Prompt type"
    )
    parser.add_argument("--model", choices=MODELS, default=MODELS[0], help="Model type")
    parser.add_argument(
        "--in_data_folder",
        default="sample_audio/wav_audio",
        help="Input data folder path",
    )
    parser.add_argument(
        "--out_folder", default="generated_output", help="Output data folder path"
    )

    args = parser.parse_args()

    main(
        in_data_folder=args.in_data_folder,
        out_folder=args.out_folder,
        model=args.model,
        lang=args.lang,
        modality=args.modality,
        prompt=args.prompt,
    )

    # python main.py --lang de --modality audio --prompt standard --model qwen_omni