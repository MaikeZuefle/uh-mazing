import pandas as pd
import xml.etree.ElementTree as ET
import re
from jiwer import Compose, RemoveMultipleSpaces, Strip
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
    'lg': 'LG_disfluent',
    'en': 'EN_disfluent'  
}
MODEL_TYPES = ["canary_cascaded", "end2end", "gold_cascaded", "speechllm", "whisper_cascaded"]

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


def extract_decode_strategy_from_filename(filename):
    match = re.search(r'_(greedy|sampling_t[\d.]+|beam_search_n\d+|beam_sampling_n\d+|typical_p[\d.]+)_output\.xml$', filename)
    if match:
        return match.group(1)
    return "greedy"