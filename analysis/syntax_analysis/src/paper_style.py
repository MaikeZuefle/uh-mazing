import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../benchmarking/src/eval"))

from create_latex_tables import LLM_JUDGE_SYSTEM_ORDER  # noqa: E402
from plot_llm_judge_scatter import (  # noqa: E402
    LANG_COLOR, LANG_DISPLAY, SYSTEM_DISPLAY, SYSTEM_MARKER, SYSTEM_MARKER_SIZE,
)

SYSTEM_ID = {
    "tower": "gold_cascaded/tower",
    "canary": "end2end/canary",
    "phi_multimodal": "speechllm/phi_multimodal",
    "gemini-2.5-flash": "end2end/gemini-2.5-flash",
}
LANGS = ["ar", "cs", "de", "es", "fr", "hi", "it", "zh"]
MIN_UTTERANCES = 20


def load_comparison(path):
    """Comparison rows with benchmark system ids, keeping only system-language pairs with enough usable outputs."""
    df = pd.read_csv(path)
    df["system_id"] = df["system"].map(SYSTEM_ID)
    df["kept_pct"] = 100 * df["n_spans_sys"] / df["n_spans_ref"]
    return df[df["n_utterances"] >= MIN_UTTERANCES]
