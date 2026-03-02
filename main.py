import argparse
import logging
import os
from tqdm import tqdm
from transformers import set_seed

# import multimodal models
from models.mllm.qwen_omni import generate as generate_qwen_omni
from models.mllm.qwen_omni import load_model as load_qwen_omni
from models.mllm.phi_multimodal import generate as generate_phi_multimodal
from models.mllm.phi_multimodal import load_model as load_phi_multimodal

# import speech models
from models.speech.whisper import generate as generate_whisper
from models.speech.whisper import load_model as load_whisper
from models.speech.canary import generate as generate_canary
from models.speech.canary import load_model as load_canary
from models.speech.owsm import generate as generate_owsm
from models.speech.owsm import load_model as load_owsm


# import text models
from models.text.llama import generate as generate_llama
from models.text.llama import load_model as load_llama
from models.text.tower import generate as generate_tower
from models.text.tower import load_model as load_tower


from utils import set_up_dirs, set_up_logging, write_to_xml, read_transcript, read_disfluency_type_transcript
from utils import LANGUAGE_DICT, REQUIRED_DISFLUENCY_COLUMNS

set_seed(42)


def load_model(model_name):
    # text models
    if model_name == "llama":
        model = load_llama()
        generate_func = generate_llama
    elif model_name == "tower":
        model = load_tower()
        generate_func = generate_tower

    # mllm models
    elif model_name == "phi_multimodal":
        model = load_phi_multimodal()
        generate_func = generate_phi_multimodal
    elif model_name == "qwen_omni":
        model = load_qwen_omni()
        generate_func = generate_qwen_omni

    # end2end models
    elif model_name == "whisper":
        model = load_whisper()
        generate_func = generate_whisper
    elif model_name == "canary":
        model = load_canary()
        generate_func = generate_canary
    elif model_name == "owsm":
        model = load_owsm()
        generate_func = generate_owsm

    else:
        raise NotImplementedError(f"Model {model_name} currently not supported!")
    return model, generate_func


def main(in_data_folder, out_folder, model, src_lang, trg_lang, modality, prompt, ablate_disfluencies=False):




    # Setting up folders and logging
    output_file_path = set_up_dirs(
        in_data_folder, out_folder, model, src_lang, trg_lang, modality, prompt
    )
    set_up_logging(output_file_path)

    logging.info("Welcome!")
    logging.info(
        f"Modality: {modality}, Src Language: {src_lang}, Trg Language: {trg_lang} Prompt: {prompt}, Model: {model}"
    )

    if ablate_disfluencies:
        if src_lang != "en":
            raise ValueError("Disfluency ablation is only supported for source language 'en'.")

        if modality != "text":
            raise ValueError("Disfluency ablation requires modality='text'.")
        
        logging.info(
        f"Ablating over disfluency types!")

    logging.info(f"Input folder: {in_data_folder}")
    logging.info(f"Output XML: {output_file_path}")


    # Loading data
    logging.info(f"Loading Data.")
    input_data = []
    if modality == "audio":
        for root, dirs, files in os.walk(in_data_folder):
            for file in files:
                if file.endswith(".wav"):
                    input_data.append((file, os.path.join(root, file)))
    elif modality == "text":
        if ablate_disfluencies:
            csv_file = os.path.join(in_data_folder, "uh-mazing-with-node-combos.csv")
            input_data = read_disfluency_type_transcript(csv_file)
        else:
            asr_model = in_data_folder.split("/")[-1]
            xml_file = in_data_folder + f"/audio_{src_lang}_{src_lang}_standard_{asr_model}_output.xml"
            input_data = read_transcript(xml_file)
    else:
        raise NotImplementedError


    # Selecting Prompt
    if modality == "audio":
        if prompt == "standard":
            instruction = f"Listen to the following audio and translate the speech into {LANGUAGE_DICT[trg_lang]} text."
        elif prompt == "disfluency_aware":
            instruction = f"Listen to the following audio and translate the speech into {LANGUAGE_DICT[trg_lang]} text, " \
            "keeping any disfluencies (such as 'um', 'uh', repetitions, and hesitations) in the translation."
        else:
            raise NotImplementedError
    elif modality == "text":
        if prompt == "standard":
            instruction = f"Translate the transcribed speech into {LANGUAGE_DICT[trg_lang]} text."
        elif prompt == "disfluency_aware":
            instruction = f"Translate the transcribed speech into {LANGUAGE_DICT[trg_lang]} text, " \
            "keeping any disfluencies (such as 'um', 'uh', repetitions, and hesitations) in the translation."
        else:
            raise NotImplementedError
    else: 
        raise NotImplementedError


    # Loading Model
    logging.info(f"Loading Model.")
    model_instance, generate = load_model(model)

    logging.info(f"Starting Output Generation.")
    outputs = []

    
    for file_name, example_path in tqdm(input_data, desc="Generating Outputs"):
        output = generate(model_instance, instruction, example_path, modality, src_lang=src_lang, trg_lang=trg_lang).strip()
        outputs.append((file_name, output))

    logging.info(f"Writing Outputs to XML file {output_file_path}.")
    write_to_xml(outputs, src_lang=src_lang, trg_lang=trg_lang, output_file=output_file_path)
    logging.info("XML written successfully")
    logging.info("All done.")


if __name__ == "__main__":
    LANGS = list(LANGUAGE_DICT.keys())
    MODALITIES = ["text", "audio", "mllm"]
    TASKS = ["ASR", "ST"]
    PROMPT = ["standard", "disfluency_aware"]
    MODELS = [
        "phi_multimodal",
        "llama",
        "tower",
        "qwen_omni",
        "canary",
        "whisper",
        "owsm",
    ]

    parser = argparse.ArgumentParser(description="Process MCIF data.")

    parser.add_argument(
        "--src_lang", choices=LANGS, default=LANGS[0], help="Language to process"
    )

    parser.add_argument(
        "--trg_lang", choices=LANGS, default=LANGS[0], help="Language to process"
    )
    parser.add_argument(
        "--modality", choices=MODALITIES, default=MODALITIES[0], help="Modality type"
    )

    parser.add_argument(
        "--tast", choices=TASKS, default=TASKS[0], help="Task"
    )

    parser.add_argument(
        "--prompt", choices=PROMPT, default=PROMPT[0], help="Prompt type"
    )
    parser.add_argument("--model", choices=MODELS, default=MODELS[0], help="Model type")

    parser.add_argument(
        "--ablate_disfluencies",
        action="store_true",
        help="If set, will generate outputs for different disfluency inputs."
    )
    parser.add_argument(
        "--in_data_folder",
        default="debug_audio",
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
        src_lang=args.src_lang,
        trg_lang=args.trg_lang,
        modality=args.modality,
        prompt=args.prompt,
        ablate_disfluencies=args.ablate_disfluencies,
    )

    # python main.py --lang de --modality audio --prompt standard --model qwen_omni