import pandas as pd
import xml.etree.ElementTree as ET
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, RemoveMultipleSpaces, Strip, ExpandCommonEnglishContractions
normalizer = Compose([
    RemoveMultipleSpaces(),
    Strip()
])



def clean_disfluency_text(text: str) -> str:
    """
    Remove underscore-based disfluency markers.
    Example: _uh_ -> uh
    """
    return normalizer(text.replace("_", "").replace("ـ", ""))

# ----------------------------
# Load XML hypothesis
# ----------------------------
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

# ----------------------------
# Load CSV references
# ----------------------------
def load_csv_references(csv_path):
    """
    Adds a sample_id column matching XML IDs:
    sw2005,B,2 -> sw2005_B_2
    """
    df = pd.read_csv(csv_path)

    df["sample_id"] = (
        "sw"
        + df["file"].str.replace("sw", "", regex=False).str.zfill(5)
        + "_"
        + df["speaker"].astype(str)
        + "_"
        + df["turn"].astype(str)
    )

    return df