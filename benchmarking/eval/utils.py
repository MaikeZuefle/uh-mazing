import os
import pandas as pd
import xml.etree.ElementTree as ET
import re
from jiwer import Compose, RemoveMultipleSpaces, Strip
from comet import download_model, load_from_checkpoint
normalizer = Compose([
    RemoveMultipleSpaces(),
    Strip()
])


LANG_COLUMN_MAP = {
    'ar': 'AR_disfluent',
    'cs': 'CS_disfluent',
    'de': 'DE_disfluent',
    'es': 'ES_disfluent',
    'fr': 'FR_disfluent',
    'hi': 'HI_disfluent',
    'it': 'IT_disfluent',
    'zh': 'ZH_disfluent',
    'sw': 'SW_disfluent',
    'en': 'EN_disfluent'  
}
MODEL_TYPES = ["canary_cascaded", "end2end", "gold_cascaded", "speechllm", "whisper_cascaded", "commercial_models"]

# Maps uppercase lang codes in commercial CSV hypothesis columns to their reference column lang codes
# (e.g. hypothesis column CZ_model → reference column CS_disfluent)
COMMERCIAL_REF_LANG_MAP = {"CZ": "CS"}

REQUIRED_DISFLUENCY_COLUMNS = [
    "ID",
    "EN_fluent",
    "EN_disfluent",
    "edited_nodes_only",
    "intj_only",
    "prn_only",
    "edited_intj",
    "intj_prn",
    "edited_prn",
    "edited_prn_intj",
]

# Commercial model CSV constants
COMMERCIAL_LANG_CODES = ["ZH", "ES", "HI", "FR", "DE", "IT", "SW", "CZ", "AR", "LG"]

COMMERCIAL_DISFLUENCY_TYPES = [
    "edited_nodes_only", "edited_prn_intj", "edited_intj",
    "edited_prn", "intj_prn", "intj_only", "prn_only",
]

COMMERCIAL_PROMPT_TYPES = ["standard"]

# Maps uppercase lang codes used in commercial CSV columns to ISO 639-1 codes
COMMERCIAL_LANG_CODE_MAP = {
    "ZH": "zh",
    "ES": "es",
    "HI": "hi",
    "FR": "fr",
    "DE": "de",
    "IT": "it",
    "CZ": "cs",  # Czech: column uses CZ, ISO 639-1 is cs
    "AR": "ar",
}

def load_comet_model(model_name="Unbabel/wmt22-cometkiwi-da"):
    model_path = download_model(model_name)
    return load_from_checkpoint(model_path).to("cuda")


def append_result(output_csv, result_dict):
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    row = pd.DataFrame([result_dict])
    if not os.path.exists(output_csv):
        row.to_csv(output_csv, mode='w', header=True, index=False)
    else:
        row.to_csv(output_csv, mode='a', header=False, index=False)


def parse_xml_filename(filename):
    """Returns (source_lang, target_lang, prompt_type) from a non-commercial XML filename."""
    source_lang, target_lang = extract_language_from_filename(filename)
    if source_lang is None or target_lang is None:
        raise ValueError(f"Could not extract language pair from {filename}")
    prompt_type = extract_prompt_type_from_filename(filename)
    if prompt_type is None:
        raise ValueError(f"Could not extract prompt type from {filename}")
    return source_lang, target_lang, prompt_type


def clean_disfluency_text(text: str) -> str:
    """
    Remove underscore-based disfluency markers.
    Example: _uh_ -> uh
    """
    return normalizer(text.replace("_", "").replace("ـ", ""))


def load_xml_hypotheses(xml_path):
    """
    Returns dict: {sample_id_without_ext: hypothesis_text}
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    hyps = {}
    for sample in root.iter("sample"):
        sample_id = sample.attrib["id"].replace(".wav", "")
        hyps[sample_id] = sample.text.strip()

    return hyps


def load_csv_references(csv_path):
    """
    Adds a sample_id column matching XML IDs:
    sw2005,B,2 -> sw2005_B_2
    """
    df = pd.read_csv(csv_path)

    if "file" in df:
        df["sample_id"] = (
            "sw"
            + df["file"].str.replace("sw", "", regex=False).str.zfill(5)
            + "_"
            + df["speaker"].astype(str)
            + "_"
            + df["turn"].astype(str)
        )
    else:
        df["sample_id"] = (
            "sw0"
            + df["ID"].str.replace("sw", "", regex=False).str.zfill(5)
        )


    return df



def extract_language_from_filename(filename):
    # en→x
    match = re.search(r'_en_([a-z]{2})_', filename)
    if match:
        return ("en", match.group(1))
    
    # x→en 
    match = re.search(r'_([a-z]{2})_en_', filename)
    if match:
        return (match.group(1), "en")
    
    return (None, None)


def extract_prompt_type_from_filename(filename):
    if "disfluency_aware" in filename:
        return "disfluency_aware"
    elif "standard" in filename:
        return "standard"
    return None