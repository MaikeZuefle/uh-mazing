import xml.etree.ElementTree as ET
import logging
import os
import sys

TASK_ATTRIB = ["src_lang", "trg_lang"]
LANGUAGE_DICT = {
    "zh": "Mandarin",
    "es": "Spanish",
    "hu": "Hungarian",
    "ro": "Romanian",
    "de": "German",
    "cs": "Czech",
    "vi": "Vietnamese",
    "tr": "Turkish",
    "it": "Italian",
    "en": "English",
}

def set_up_logging(output_file_path):
    log_file = output_file_path.replace(".xml", ".log")

    # Clear existing handlers if rerunning in interactive environments
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Set a unified format without milliseconds
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to root logger
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])


def set_up_dirs(in_data_folder, out_folder, model, src_lang, trg_lang, modality, prompt):
    if not os.path.isdir(in_data_folder):
        raise FileNotFoundError(f"Input folder does not exist: {in_data_folder}")

    # Set up model-specific output directory
    model_out_folder = os.path.join(out_folder, model)
    os.makedirs(model_out_folder, exist_ok=True)

    # Define output file paths
    output_file_name = f"{modality}_{src_lang}_{trg_lang}_{prompt}_{model}_output.xml"
    output_file_path = os.path.join(model_out_folder, output_file_name)

    return output_file_path


def write_to_xml(outputs, src_lang, trg_lang, output_file):
    samples = [{"id": str(sample_id), "output": pred} for sample_id, pred in outputs]
    
    output_dict = {
        "src_lang": src_lang,
        "trg_lang": trg_lang,
        "samples": samples
    }

    xml = ET.Element("testset", attrib={'name': "Disfluency Pilot", "type": "output"})
    xml_track = ET.SubElement(xml, "task", attrib={key: output_dict[key] for key in TASK_ATTRIB})
    
    for sample in output_dict["samples"]:
        ET.SubElement(xml_track, "sample", attrib={'id': sample['id']}).text = sample['output']

    tree = ET.ElementTree(xml)
    ET.indent(tree)
    tree.write(sys.stdout.buffer, encoding="utf-8", xml_declaration=True)  # for stdout
    tree.write(output_file, encoding="utf-8",)

def read_txt_file(file_path):
    with open(file_path, "r", encoding="UTF8") as example:
        example = example.readlines()[0].strip()
    return example

def read_transcript(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    input_data = []
    for task in root.findall("task"):
        for sample in task.findall("sample"):
            sample_id = sample.get("id")
            text = sample.text.strip() if sample.text else ""
            input_data.append([sample_id, text])

    return input_data