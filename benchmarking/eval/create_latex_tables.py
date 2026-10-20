import os
import pandas as pd

LANG_COLS = ["ar", "cs", "de", "es", "fr", "hi", "it", "zh"]

# audio-gemini / audio-chatgpt (the "with extra context" audio condition) are intentionally
# excluded from the tables below -- only the audio-noextra condition is reported.
EXCLUDED_MODELS = {"audio-gemini", "audio-chatgpt"}

# Single source of truth for each model's table category, display name, and subgroup
# (used for the partial \cmidrule within a category). One entry per model -- nothing
# below should need to be redefined per table.
MODELS = {
    "canary-llama": {"category": "Cascaded", "display": "\\textsc{Llama}\\textsubscript{Canary ASR}", "subgroup": "asr"},
    "canary-tower": {"category": "Cascaded", "display": "\\textsc{Tower}\\textsubscript{Canary ASR}", "subgroup": "asr"},
    "whisper-llama": {"category": "Cascaded", "display": "\\textsc{Llama}\\textsubscript{Whisper ASR}", "subgroup": "asr"},
    "whisper-tower": {"category": "Cascaded", "display": "\\textsc{Tower}\\textsubscript{Whisper ASR}", "subgroup": "asr"},
    "gold-transcript-llama": {"category": "Cascaded", "display": "\\textsc{Llama}\\textsubscript{Human transcript}", "subgroup": "gold"},
    "gold-transcript-tower": {"category": "Cascaded", "display": "\\textsc{Tower}\\textsubscript{Human transcript}", "subgroup": "gold"},
    "owsm": {"category": "End-to-end", "display": "\\textsc{OWSM}"},
    "canary": {"category": "End-to-end", "display": "\\textsc{Canary}"},
    "phi": {"category": "SpeechLLM", "display": "\\textsc{Phi-4-MM-instruct}"},
    "qwen": {"category": "SpeechLLM", "display": "\\textsc{Qwen2.5-Omni}"},
    "gold-text-gemini": {"category": "Commercial", "display": "\\textsc{Gemini}\\textsubscript{Human transcript}", "subgroup": "gold"},
    "gold-text-chatgpt": {"category": "Commercial", "display": "\\textsc{ChatGPT}\\textsubscript{Human transcript}", "subgroup": "gold"},
    "asr-text-gemini": {"category": "Commercial", "display": "\\textsc{Gemini}\\textsubscript{ASR transcript}", "subgroup": "asr"},
    "asr-text-chatgpt": {"category": "Commercial", "display": "\\textsc{ChatGPT}\\textsubscript{ASR transcript}", "subgroup": "asr"},
    "audio-noextra-gemini": {"category": "Commercial", "display": "\\textsc{Gemini}\\textsubscript{Audio}", "subgroup": "audio"},
    "audio-noextra-chatgpt": {"category": "Commercial", "display": "\\textsc{ChatGPT}\\textsubscript{Audio}", "subgroup": "audio"},
    "gold-transcript-chatgpt": {"category": "Commercial", "display": "\\textsc{ChatGPT}\\textsubscript{Human transcript}", "subgroup": "gold"},
    "gold-transcript-gemini": {"category": "Commercial", "display": "\\textsc{Gemini}\\textsubscript{Human transcript}", "subgroup": "gold"},
}

ASR_MODEL_DISPLAY = {
    "whisper": "\\textsc{Whisper}",
    "canary": "\\textsc{Canary}",
    "gpt": "\\textsc{ChatGPT}",
    "gemini": "\\textsc{Gemini}",
}

PROMPT_DISPLAY = {"standard": "Std.", "disfluency_aware": "Disfl.", "disfluent": "Disfl."}

PROMPT_NOTE = "Std.\\ = standard prompt; Disfl.\\ = disfluency-aware prompt."

TRANSLATION_CAPTIONS = {
    "comet": (
        "COMET-Kiwi scores across models, prompts, and target languages for "
        "English$\\to$X and X$\\to$English translation. "
        + PROMPT_NOTE,
        "tab:comet",
    ),
    "chrf": (
        "chrF scores across models, prompts, and target languages for "
        "English$\\to$X and X$\\to$English translation. "
        + PROMPT_NOTE,
        "tab:chrf",
    ),
}

MAIN_TABLE_CAPTIONS = {
    "comet": (
        "COMET-Kiwi scores (standard prompt, English$\\to$X) for the best "
        "automatic-transcript cascaded system, end-to-end, speechLLM, and audio-input "
        "commercial models. Full results, including disfluency-aware prompting, gold "
        "transcripts, and X$\\to$English, are in the appendix.",
        "tab:comet_main",
    ),
    "chrf": (
        "\\textbf{Cascaded systems lead in translation disfluent speech.} "
        "chrF scores (standard prompt, English$\\to$X) for the best cascaded system, "
        "end-to-end system, speechLLM, and commercial models. "
        "Full results \\cref{app:detailed_results}.",
        "tab:chrf_main",
    ),
}


def escape_latex(s):
    return str(s).replace("_", "\\_")


def prompt_display(val):
    if pd.isna(val):
        return "N/A"
    return PROMPT_DISPLAY.get(val, escape_latex(val))


FLOOR_QUANTILE = 0.3


def compute_bounds(values, higher_is_better=True, floor_quantile=FLOOR_QUANTILE):
    """Anchor the 'worst' end of the color scale at a quantile rather than the literal
    min/max, so a few very bad outlier rows don't compress the gradient for everyone else.
    The 'best' end stays the literal best value, so the top performer is still darkest."""
    s = pd.Series(values)
    if higher_is_better:
        return s.quantile(floor_quantile), s.max()
    return s.quantile(1 - floor_quantile), s.min()


TABLE_COLOR = "tabsalmon"   # requires \definecolor{tabsalmon}{HTML}{EC9374} in preamble
TABLE_COLOR_XEN = "tabblue"  # requires \definecolor{tabblue}{HTML}{3986BC} in preamble
TABLE_COLOR_INTENSITY = 50  # coral at 25% is nearly invisible; 50% gives a clear gradient


# Requires \definecolor{tabsalmon}{HTML}{EC9374} in the LaTeX preamble.
def colorize(value_str, worst, best, color=TABLE_COLOR, max_intensity=TABLE_COLOR_INTENSITY):
    if value_str == "--":
        return value_str
    norm = 0.0 if best == worst else (float(value_str) - worst) / (best - worst)
    norm = max(0.0, min(1.0, norm))
    intensity = round(norm * max_intensity)
    return f"\\cellcolor{{{color}!{intensity}}}{{{value_str}}}"


def fmt_cell(val):
    if pd.isna(val) or val == "":
        return "--"
    return f"{float(val):.2f}"


def split_sections(df):
    """Split a translations table CSV (with '=== ... ===' marker rows) into named sections."""
    sections = {}
    current, rows = None, []
    for _, row in df.iterrows():
        model = row["model"]
        if isinstance(model, str) and model.startswith("==="):
            if current is not None:
                sections[current] = rows
            current = model.strip("= ").strip()
            rows = []
            continue
        if pd.isna(model) or model == "" or model in EXCLUDED_MODELS:
            continue
        rows.append(row)
    if current is not None:
        sections[current] = rows
    return sections


CASCADED_ORDER = [
    "canary-llama", "whisper-llama", "gold-transcript-llama",
    "canary-tower", "whisper-tower", "gold-transcript-tower",
]
CASCADED_ORDER_INDEX = {name: idx for idx, name in enumerate(CASCADED_ORDER)}


def run_lengths(items):
    """For each index, the length of the contiguous run of equal values it belongs to."""
    lengths = [0] * len(items)
    i = 0
    while i < len(items):
        j = i
        while j < len(items) and items[j] == items[i]:
            j += 1
        for k in range(i, j):
            lengths[k] = j - i
        i = j
    return lengths


def build_row_lines(rows, color=TABLE_COLOR):
    """Build LaTeX row lines, grouping contiguous rows by model category into a leading
    \\multirow column, also \\multirow-ing the model column over its standard/disfluency-aware
    prompt pair, colorizing each language column by its min/max within these rows, inserting a
    \\midrule between category groups, and a partial \\cmidrule (skipping the category column)
    between known subgroups within a category."""
    ncols = 3 + len(LANG_COLS)
    col_bounds = {}
    col_best = {}
    for lang in LANG_COLS:
        values = [float(row[lang]) for row in rows if not pd.isna(row[lang]) and row[lang] != ""]
        fmt_values = [float(fmt_cell(row[lang])) for row in rows if not pd.isna(row[lang]) and row[lang] != ""]
        if values:
            col_bounds[lang] = compute_bounds(values, higher_is_better=True)
            col_best[lang] = max(fmt_values)

    lines = []
    i, first_group = 0, True
    while i < len(rows):
        category = MODELS[rows[i]["model"]]["category"]
        j = i
        while j < len(rows) and MODELS[rows[j]["model"]]["category"] == category:
            j += 1
        group = rows[i:j]
        if category == "Cascaded":
            group = sorted(group, key=lambda row: CASCADED_ORDER_INDEX[row["model"]])
        if not first_group:
            lines.append("\\midrule")
        first_group = False
        prev_subgroup = None
        model_run_len = run_lengths([row["model"] for row in group])
        for k, row in enumerate(group):
            subgroup = MODELS[row["model"]].get("subgroup")
            if k > 0 and subgroup != prev_subgroup:
                lines.append("\\addlinespace[3pt]")
            prev_subgroup = subgroup
            is_new_model = k == 0 or group[k]["model"] != group[k - 1]["model"]
            if is_new_model:
                model_disp = MODELS[row["model"]].get("display", escape_latex(row["model"]))
                n = model_run_len[k]
                model = f"\\multirow{{{n}}}{{*}}{{{model_disp}}}" if n > 1 else model_disp
            else:
                model = ""
            prompt = prompt_display(row["prompt_type"])
            cells = []
            for lang in LANG_COLS:
                raw = fmt_cell(row[lang])
                is_best = (
                    raw != "--"
                    and lang in col_best
                    and abs(float(raw) - col_best[lang]) < 1e-9
                )
                display = f"\\textbf{{{raw}}}" if is_best else raw
                if lang in col_bounds and raw != "--":
                    worst, best = col_bounds[lang]
                    norm = max(0.0, min(1.0, (float(raw) - worst) / (best - worst))) if best != worst else 0.0
                    intensity = round(norm * TABLE_COLOR_INTENSITY)
                    cell = f"\\cellcolor{{{color}!{intensity}}}{{{display}}}"
                else:
                    cell = display
                cells.append(cell)
            if k == 0:
                cat_cell = f"\\multirow{{{len(group)}}}{{*}}{{{category}}}" if len(group) > 1 else category
            else:
                cat_cell = ""
            lines.append(" & ".join([cat_cell, model, prompt] + cells) + " \\\\")
        i = j
    return lines


def build_translation_table(csv_path, metric):
    df = pd.read_csv(csv_path)
    sections = split_sections(df)
    caption, label = TRANSLATION_CAPTIONS[metric]

    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{lll" + "r" * len(LANG_COLS) + "}")
    lines.append("\\toprule")
    header = ["Category", "Model", "Prompt"] + [c.upper() for c in LANG_COLS]
    lines.append(" & ".join(f"\\textbf{{{h}}}" for h in header) + " \\\\")
    lines.append("\\midrule")

    section_order = ["EN → X TRANSLATIONS", "X → EN TRANSLATIONS"]
    section_titles = {
        "EN → X TRANSLATIONS": "EN $\\to$ X Translations",
        "X → EN TRANSLATIONS": "X $\\to$ EN Translations",
    }
    ncols = len(LANG_COLS) + 3
    first_section = True
    for section_name in section_order:
        if section_name not in sections:
            continue
        if not first_section:
            lines.append("\\midrule")
        lines.append(f"\\multicolumn{{{ncols}}}{{c}}{{\\textit{{{section_titles[section_name]}}}}} \\\\")
        lines.append("\\midrule")
        section_rows = sections[section_name]
        color = TABLE_COLOR_XEN if section_name == "X → EN TRANSLATIONS" else TABLE_COLOR
        if section_name == "X → EN TRANSLATIONS":
            section_rows = [row for row in section_rows if row["prompt_type"] != "disfluency_aware"]
        lines.extend(build_row_lines(section_rows, color=color))
        first_section = False

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


MAIN_TABLE_CASCADED_CANDIDATES = ["canary-llama", "canary-tower", "whisper-llama", "whisper-tower"]
MAIN_TABLE_OTHER_MODELS = ["owsm", "canary", "phi", "qwen"]
MAIN_TABLE_COMMERCIAL_MODELS = ["audio-noextra-gemini", "audio-noextra-chatgpt"]


def select_best_cascaded_asr_model(rows):
    """Pick the single best automatic-transcript (non-gold) cascaded model by mean COMET
    across languages, among canary-llama/canary-tower/whisper-llama/whisper-tower."""
    best_model, best_mean = None, -1
    for model in MAIN_TABLE_CASCADED_CANDIDATES:
        row = next((r for r in rows if r["model"] == model), None)
        if row is None:
            continue
        values = [float(row[lang]) for lang in LANG_COLS if not pd.isna(row[lang]) and row[lang] != ""]
        mean = sum(values) / len(values)
        if mean > best_mean:
            best_mean, best_model = mean, model
    return best_model


ALL_MODELS_ORDER = [
    "canary-llama", "whisper-llama", "gold-transcript-llama",
    "canary-tower", "whisper-tower", "gold-transcript-tower",
    "owsm", "canary",
    "phi", "qwen",
    "asr-text-gemini", "gold-text-gemini", "audio-noextra-gemini",
    "asr-text-chatgpt", "gold-text-chatgpt", "audio-noextra-chatgpt",
]

ALL_MODELS_CAPTIONS = {
    "chrf": (
        "\\textbf{Cascaded systems lead among open-source models, and commercial systems achieve the highest scores overall.} "
        "chrF scores (standard prompt) for all models and transcript conditions, "
        "English$\\to$X and X$\\to$English. "
        "Subscripts: Audio = speech input; ASR = automatic transcript; Human = human transcript.",
        "tab:chrf_main",
    ),
    "comet": (
        "COMET-Kiwi scores (standard prompt) for all models and transcript conditions, "
        "English$\\to$X and X$\\to$English. "
        "Subscripts: Audio = speech input; ASR = automatic transcript; Human = human transcript.",
        "tab:comet_main_all",
    ),
}


# Commercial models appear under different keys in X→EN vs EN→X data.
MODEL_ALIASES = {
    "gold-transcript-chatgpt": "gold-text-chatgpt",
    "gold-transcript-gemini": "gold-text-gemini",
}


def _canonical(model):
    return MODEL_ALIASES.get(model, model)


def _render_all_models_section(rows, lines, first_section, color=TABLE_COLOR):
    """Render one direction section (EN→X or X→EN) into lines for build_all_models_std_table."""
    order_index = {model: idx for idx, model in enumerate(ALL_MODELS_ORDER)}
    rows = sorted(
        (row for row in rows if _canonical(row["model"]) in order_index),
        key=lambda row: order_index[_canonical(row["model"])],
    )

    col_bounds = {}
    col_best = {}
    for lang in LANG_COLS:
        values = [float(row[lang]) for row in rows if not pd.isna(row[lang]) and row[lang] != ""]
        fmt_values = [float(fmt_cell(row[lang])) for row in rows if not pd.isna(row[lang]) and row[lang] != ""]
        if values:
            col_bounds[lang] = compute_bounds(values, higher_is_better=True)
            col_best[lang] = max(fmt_values)

    ncols = 2 + len(LANG_COLS)
    i, first_group = 0, True
    while i < len(rows):
        category = MODELS[rows[i]["model"]]["category"]
        j = i
        while j < len(rows) and MODELS[rows[j]["model"]]["category"] == category:
            j += 1
        group = rows[i:j]
        if category == "Cascaded":
            group = sorted(group, key=lambda row: CASCADED_ORDER_INDEX.get(row["model"], 99))
        if not first_group:
            lines.append("\\midrule")
        first_group = False
        prev_subgroup = None
        for k, row in enumerate(group):
            subgroup = MODELS[row["model"]].get("subgroup")
            if k > 0 and subgroup != prev_subgroup:
                lines.append("\\addlinespace[3pt]")
            prev_subgroup = subgroup
            model_disp = MODELS[row["model"]].get("display", escape_latex(row["model"]))
            cells = []
            for lang in LANG_COLS:
                raw = fmt_cell(row[lang])
                is_best = (
                    raw != "--"
                    and lang in col_best
                    and abs(float(raw) - col_best[lang]) < 1e-9
                )
                display = f"\\textbf{{{raw}}}" if is_best else raw
                if lang in col_bounds and raw != "--":
                    worst, best = col_bounds[lang]
                    norm = max(0.0, min(1.0, (float(raw) - worst) / (best - worst))) if best != worst else 0.0
                    intensity = round(norm * TABLE_COLOR_INTENSITY)
                    cell = f"\\cellcolor{{{color}!{intensity}}}{{{display}}}"
                else:
                    cell = display
                cells.append(cell)
            if k == 0:
                cat_cell = f"\\multirow{{{len(group)}}}{{*}}{{{category}}}" if len(group) > 1 else category
            else:
                cat_cell = ""
            lines.append(" & ".join([cat_cell, model_disp] + cells) + " \\\\")
        i = j


def build_all_models_std_table(csv_path, metric):
    """Full table: standard prompt only, all models and transcript conditions, EN→X and X→EN."""
    caption, label = ALL_MODELS_CAPTIONS[metric]
    df = pd.read_csv(csv_path)
    sections = split_sections(df)
    ncols = 2 + len(LANG_COLS)

    section_data = {
        "EN $\\to$ X: Speech-to-Text Translation": [r for r in sections.get("EN → X TRANSLATIONS", []) if r["prompt_type"] != "disfluency_aware"],
        "X $\\to$ EN: Text-to-Text Translation": [r for r in sections.get("X → EN TRANSLATIONS", []) if r["prompt_type"] != "disfluency_aware"],
    }

    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{ll" + "r" * len(LANG_COLS) + "}")
    lines.append("\\toprule")
    header = ["Category", "Model"] + [c.upper() for c in LANG_COLS]
    lines.append(" & ".join(f"\\textbf{{{h}}}" for h in header) + " \\\\")

    first_section = True
    for section_title, rows in section_data.items():
        if not rows:
            continue
        color = TABLE_COLOR_XEN if "X $\\to$ EN" in section_title else TABLE_COLOR
        lines.append("\\midrule")
        lines.append(f"\\multicolumn{{{ncols}}}{{c}}{{\\textit{{{section_title}}}}} \\\\")
        lines.append("\\midrule")
        _render_all_models_section(rows, lines, first_section, color=color)
        first_section = False

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


def build_main_translation_table(csv_path, metric):
    """Condensed main-paper table: standard prompt only, EN-to-X only, the single best
    automatic-transcript cascaded model (no gold transcript), and audio-input commercial
    models only (no gold-/ASR-text commercial conditions)."""
    caption, label = MAIN_TABLE_CAPTIONS[metric]
    df = pd.read_csv(csv_path)
    sections = split_sections(df)
    en_x_rows = [row for row in sections["EN → X TRANSLATIONS"] if row["prompt_type"] != "disfluency_aware"]

    best_cascaded = select_best_cascaded_asr_model(en_x_rows)
    keep_models = [best_cascaded] + MAIN_TABLE_OTHER_MODELS + MAIN_TABLE_COMMERCIAL_MODELS
    order_index = {model: idx for idx, model in enumerate(keep_models)}
    rows = sorted((row for row in en_x_rows if row["model"] in keep_models), key=lambda row: order_index[row["model"]])

    col_bounds = {}
    for lang in LANG_COLS:
        values = [float(row[lang]) for row in rows if not pd.isna(row[lang]) and row[lang] != ""]
        if values:
            col_bounds[lang] = compute_bounds(values, higher_is_better=True)

    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{ll" + "r" * len(LANG_COLS) + "}")
    lines.append("\\toprule")
    header = ["Category", "Model"] + [c.upper() for c in LANG_COLS]
    lines.append(" & ".join(f"\\textbf{{{h}}}" for h in header) + " \\\\")
    lines.append("\\midrule")

    i, first_group = 0, True
    while i < len(rows):
        category = MODELS[rows[i]["model"]]["category"]
        j = i
        while j < len(rows) and MODELS[rows[j]["model"]]["category"] == category:
            j += 1
        group = rows[i:j]
        if not first_group:
            lines.append("\\midrule")
        first_group = False
        for k, row in enumerate(group):
            model_disp = MODELS[row["model"]].get("display", escape_latex(row["model"]))
            cells = []
            for lang in LANG_COLS:
                cell = fmt_cell(row[lang])
                if lang in col_bounds:
                    worst, best = col_bounds[lang]
                    cell = colorize(cell, worst, best)
                cells.append(cell)
            if k == 0:
                cat_cell = f"\\multirow{{{len(group)}}}{{*}}{{{category}}}" if len(group) > 1 else category
            else:
                cat_cell = ""
            lines.append(" & ".join([cat_cell, model_disp] + cells) + " \\\\")
        i = j

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


def build_asr_table(csv_path):
    raw = pd.read_csv(csv_path, header=None)
    col_names = raw.iloc[1].tolist()
    data = raw.iloc[4:].reset_index(drop=True)
    data.columns = col_names
    metric_cols = col_names[2:]

    # WER columns are given as "20.37%" strings; strip the % and report it in the header instead.
    for col in metric_cols[:2]:
        data[col] = data[col].astype(str).str.rstrip("%")

    col_bounds = {}
    for col in metric_cols:
        values = [float(v) for v in data[col] if not pd.isna(v) and v != ""]
        if values:
            col_bounds[col] = compute_bounds(values, higher_is_better=False)

    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{ll" + "r" * 8 + "}")
    lines.append("\\toprule")
    lines.append(
        " & & \\multicolumn{2}{c}{\\textbf{WER (\\%)}} & \\multicolumn{3}{c}{\\textbf{E-Score}} "
        "& \\multicolumn{3}{c}{\\textbf{Z-Score}} \\\\"
    )
    lines.append(
        " & ".join(
            f"\\textbf{{{h}}}"
            for h in ["Model", "Prompt", "Disfl. ref", "Fluent ref", "P", "R", "F1", "EDITED", "INTJ", "PRN"]
        )
        + " \\\\"
    )
    lines.append("\\midrule")
    for _, row in data.iterrows():
        model = ASR_MODEL_DISPLAY.get(row["model"], escape_latex(row["model"]))
        prompt = prompt_display(row["prompt"])
        cells = []
        for col in metric_cols:
            cell = fmt_cell(row[col])
            if col in col_bounds:
                worst, best = col_bounds[col]
                cell = colorize(cell, worst, best)
            cells.append(cell)
        lines.append(" & ".join([model, prompt] + cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(
        "\\caption{Automatic speech recognition (ASR) results: word error rate (WER) against "
        "disfluent and fluent references, and disfluency retention (E-Score, Z-Score) for "
        "EDITED, INTJ, and PRN node types.}"
    )
    lines.append("\\label{tab:asr_results}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


LLM_JUDGE_LANG_COLS = ["ar", "cs", "de", "es", "fr", "hi", "it", "zh"]

LLM_JUDGE_SYSTEM_ORDER = [
    "gold_cascaded/tower", "end2end/canary", "speechllm/phi_multimodal", "end2end/gemini-2.5-flash",
]
# Maps each llm_judge system identifier onto its corresponding key in MODELS, so the display
# name (with the same Gold/ASR/Audio subscript convention) is reused rather than redefined.
LLM_JUDGE_SYSTEM_KEY = {
    "gold_cascaded/tower": "gold-transcript-tower",
    "end2end/canary": "canary",
    "speechllm/phi_multimodal": "phi",
    "end2end/gemini-2.5-flash": "audio-noextra-gemini",
}
LLM_JUDGE_PROMPT = "standard"

def is_unsupported(system, value):
    # Canary does not support ZH/HI/AR; those cells are degenerate 0.0 judge scores, not
    # real quality scores, so we treat them as missing rather than plotting/tabulating them.
    return system == "end2end/canary" and value == 0.0


def build_llm_judge_combined_table(style_csv, meaning_csv):
    dfs = {"Meaning": pd.read_csv(meaning_csv), "Style": pd.read_csv(style_csv)}
    ncols = 2 + len(LLM_JUDGE_LANG_COLS)

    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{ll" + "r" * len(LLM_JUDGE_LANG_COLS) + "}")
    lines.append("\\toprule")
    header = ["Model", "Prompt"] + [c.upper() for c in LLM_JUDGE_LANG_COLS]
    lines.append(" & ".join(f"\\textbf{{{h}}}" for h in header) + " \\\\")

    for section_name, df in dfs.items():
        col_bounds = {}
        for lang in LLM_JUDGE_LANG_COLS:
            values = [
                row[lang] for _, row in df.iterrows()
                if not pd.isna(row[lang]) and row[lang] != "" and not is_unsupported(row["system"], row[lang])
            ]
            if values:
                col_bounds[lang] = compute_bounds(values, higher_is_better=True)

        lines.append("\\midrule")
        lines.append(f"\\multicolumn{{{ncols}}}{{c}}{{\\textit{{{section_name}}}}} \\\\")
        lines.append("\\midrule")
        for system in LLM_JUDGE_SYSTEM_ORDER:
            row = df[df["system"] == system].iloc[0]
            model_disp = MODELS[LLM_JUDGE_SYSTEM_KEY[system]]["display"]
            prompt = prompt_display(LLM_JUDGE_PROMPT)
            cells = []
            for lang in LLM_JUDGE_LANG_COLS:
                val = row[lang]
                if pd.isna(val) or val == "" or is_unsupported(system, val):
                    cells.append("--")
                    continue
                cell = fmt_cell(val)
                if lang in col_bounds:
                    worst, best = col_bounds[lang]
                    cell = colorize(cell, worst, best)
                cells.append(cell)
            lines.append(" & ".join([model_disp, prompt] + cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append("\\caption{LLM-judge Meaning and Style Preservation scores by system and language.}")
    lines.append("\\label{tab:llm_judge}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


CORRELATION_METRIC_ORDER = ["llm_judge", "llm_judge_ref", "chrf", "comet"]
CORRELATION_METRIC_DISPLAY = {
    "llm_judge": "\\mbox{LLM-judge}",
    "llm_judge_ref": "\\mbox{LLM-judge}",
    "chrf": "chrF",
    "comet": "\\mbox{COMET-Kiwi}",
}
CORRELATION_AXIS_ORDER = ["style", "meaning", "overall"]
CORRELATION_AXIS_DISPLAY = {"style": "Style", "meaning": "Meaning", "overall": "Overall"}
# llm_judge and llm_judge_ref each predict a per-axis value (style/meaning/overall), so their
# row labels include the axis (e.g. "LLM-judge\textsubscript{Style}"); chrf/comet produce one
# fixed quality score per item and are compared against all three human axes without axis labels.
AXIS_MATCHED_METRICS = {"llm_judge", "llm_judge_ref"}
METRIC_HAS_REF = {"llm_judge": False, "llm_judge_ref": True, "chrf": True, "comet": False}


def build_correlation_table(csv_path):
    df = pd.read_csv(csv_path)
    avg = df.groupby(["human_axis", "metric", "granularity"])["value"].mean()

    metrics_present = [m for m in CORRELATION_METRIC_ORDER if m in df["metric"].unique()]

    # best value per (axis, granularity) for bolding
    best = {}
    for axis in CORRELATION_AXIS_ORDER:
        for gran in ("segment", "system"):
            vals = [v for m in metrics_present for v in [avg.get((axis, m, gran))] if v is not None]
            best[(axis, gran)] = max(vals) if vals else None

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{llrr}")
    lines.append("\\toprule")
    lines.append(
        " & ".join(f"\\textbf{{{h}}}" for h in [
            "Metric", "\\shortstack{Ref-\\\\based?}",
            "\\shortstack{Segm.\\\\Corr.}",
            "\\shortstack{Sys.\\\\Corr.}",
        ]) + " \\\\"
    )

    for axis in CORRELATION_AXIS_ORDER:
        lines.append("\\midrule")
        lines.append(f"\\multicolumn{{4}}{{c}}{{\\textit{{{CORRELATION_AXIS_DISPLAY[axis]}}}}} \\\\")
        lines.append("\\midrule")
        for metric in metrics_present:
            axis_matched = metric in AXIS_MATCHED_METRICS
            ref_symbol = "$\\checkmark$" if METRIC_HAS_REF.get(metric, False) else "$\\times$"
            if axis_matched:
                metric_disp = f"{CORRELATION_METRIC_DISPLAY[metric]}\\textsubscript{{{CORRELATION_AXIS_DISPLAY[axis]}}}"
            else:
                metric_disp = CORRELATION_METRIC_DISPLAY[metric]
            seg = avg.get((axis, metric, "segment"))
            sys_ = avg.get((axis, metric, "system"))
            seg_str = f"{seg * 100:.1f}" if seg is not None else "--"
            sys_str = f"{sys_ * 100:.1f}" if sys_ is not None else "--"
            if seg is not None and best.get((axis, "segment")) == seg:
                seg_str = f"\\textbf{{{seg_str}}}"
            if sys_ is not None and best.get((axis, "system")) == sys_:
                sys_str = f"\\textbf{{{sys_str}}}"
            lines.append(" & ".join([metric_disp, ref_symbol, seg_str, sys_str]) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(
        "\\caption{\\textbf{Reference-based metrics have best correlation with human judgments "
        "on disfluent translations.} Correlation (\\%) between human Style/Meaning/Overall "
        "judgments and automatic metrics, averaged over the eight languages and four systems. "
        "Segment-level: mean Kendall's $\\tau$ (within-item, across systems). "
        "System-level: soft pairwise accuracy (SPA).}"
    )
    lines.append("\\label{tab:correlation}")
    lines.append("\\end{table}")
    return "\n".join(lines)


CORRELATION_LANG_COLS = ["ar", "cs", "de", "es", "fr", "hi", "it", "zh"]

NODE_COMBO_LANG_COLS = ["zh", "es", "hi", "fr", "de", "it", "cs"]

NODE_COMBO_MODEL_ORDER = ["llama", "tower", "gemini-2.5-flash", "gpt52"]
NODE_COMBO_MODEL_KEY = {
    "llama": "gold-transcript-llama",
    "tower": "gold-transcript-tower",
    "gemini-2.5-flash": "gold-text-gemini",
    "gpt52": "gold-text-chatgpt",
}

# Row order reflects the combinatorial structure directly: fluent baseline, then all 3
# single-type combos, then all 3 pairs, then the triple, then the natural disfluent
# transcript. Within the singles and pairs groups, the EDITED-vs-not contrast is still
# visible (e.g. Edited Nodes Only scores well below Intj Only / Prn Only).
NODE_COMBO_TYPE_ORDER = [
    "Fluent",
    "Intj Only", "Prn Only", "Edited Nodes Only",
    "Intj PRN", "Edited Intj", "Edited PRN",
    "Disfluent",
]
# The raw CSV's labels mix casing inconsistently ("Intj" vs "PRN" vs "Edited Nodes Only" vs
# "Edited PRN"). Normalize every label to the paper's own notation (INTJ/PRN/EDITED, per the
# methods section) instead of passing the raw strings through.
# The synthetic triple-combo (Edited PRN Intj) is omitted: once the source-text cleaning bug
# was fixed, it scores within noise of the natural Disfluent transcript, so only Disfluent
# is kept to represent that point in the combination space.
NODE_COMBO_TYPE_DISPLAY = {
    "Fluent": "Fluent",
    "Disfluent": "Disfluent",
    "Intj Only": "INTJ Only",
    "Prn Only": "PRN Only",
    "Edited Nodes Only": "EDITED Only",
    "Intj PRN": "INTJ + PRN",
    "Edited Intj": "EDITED + INTJ",
    "Edited PRN": "EDITED + PRN",
}

NODE_COMBO_TYPE_GROUP = {
    "Fluent": "fluent",
    "Intj Only": "single", "Prn Only": "single", "Edited Nodes Only": "single",
    "Intj PRN": "pair", "Edited Intj": "pair", "Edited PRN": "pair",
    "Disfluent": "disfluent",
}


def build_node_combos_table(csv_path):
    df = pd.read_csv(csv_path)
    ncols = 2 + len(NODE_COMBO_LANG_COLS)

    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{ll" + "r" * len(NODE_COMBO_LANG_COLS) + "}")
    lines.append("\\toprule")
    header = ["Model", "Disfluency Type"] + [c.upper() for c in NODE_COMBO_LANG_COLS]
    lines.append(" & ".join(f"\\textbf{{{h}}}" for h in header) + " \\\\")
    lines.append("\\midrule")

    for i, model in enumerate(NODE_COMBO_MODEL_ORDER):
        if i > 0:
            lines.append("\\midrule")
        model_disp = MODELS[NODE_COMBO_MODEL_KEY[model]]["display"]
        model_rows = df[df["model"] == model]

        # Color bounds computed per model (not globally), so each model's own EDITED-vs-not
        # contrast is visible regardless of how its absolute scores compare to other models.
        col_bounds = {}
        for lang in NODE_COMBO_LANG_COLS:
            values = [v for v in model_rows[lang] if not pd.isna(v)]
            if values:
                col_bounds[lang] = compute_bounds(values, higher_is_better=True)

        types_present = [t for t in NODE_COMBO_TYPE_ORDER if t in model_rows["disfluency_type"].values]
        prev_group = None
        for k, dtype in enumerate(types_present):
            group = NODE_COMBO_TYPE_GROUP[dtype]
            if k > 0 and group != prev_group:
                lines.append(f"\\cmidrule{{2-{ncols}}}")
            prev_group = group
            row = model_rows[model_rows["disfluency_type"] == dtype].iloc[0]
            cells = []
            for lang in NODE_COMBO_LANG_COLS:
                val = row[lang]
                cell = fmt_cell(val)
                if lang in col_bounds and not (pd.isna(val) or val == ""):
                    worst, best = col_bounds[lang]
                    cell = colorize(cell, worst, best)
                cells.append(cell)
            model_cell = f"\\multirow{{{len(types_present)}}}{{*}}{{{model_disp}}}" if k == 0 else ""
            dtype_disp = NODE_COMBO_TYPE_DISPLAY.get(dtype, dtype)
            lines.append(" & ".join([model_cell, dtype_disp] + cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append(
        "\\caption{COMET-Kiwi scores when the English source contains only specific "
        "combinations of EDITED/INTJ/PRN disfluency nodes, text-only (gold transcript as "
        "input). Fluent and Disfluent are the unmodified reference transcripts.}"
    )
    lines.append("\\label{tab:node_combos}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


def build_correlation_per_language_table(csv_path):
    df = pd.read_csv(csv_path)
    pivot = df.set_index(["human_axis", "metric", "granularity", "lang"])["value"]

    metrics_present = [m for m in CORRELATION_METRIC_ORDER if m in df["metric"].unique()]
    ncols = 3 + len(CORRELATION_LANG_COLS)

    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{llc" + "r" * len(CORRELATION_LANG_COLS) + "}")
    lines.append("\\toprule")
    header = ["Metric", "Ref?", "Human Axis"] + [c.upper() for c in CORRELATION_LANG_COLS]
    lines.append(" & ".join(f"\\textbf{{{h}}}" for h in header) + " \\\\")

    # best value per (axis, granularity, lang) for bolding
    best_lang = {}
    for axis in CORRELATION_AXIS_ORDER:
        for gran in ("segment", "system"):
            for lang in CORRELATION_LANG_COLS:
                vals = [v for m in metrics_present for v in [pivot.get((axis, m, gran, lang))] if v is not None]
                best_lang[(axis, gran, lang)] = max(vals) if vals else None

    for granularity, granularity_disp in [("segment", "Segment-Corr"), ("system", "System-Corr")]:
        lines.append("\\midrule")
        lines.append(f"\\multicolumn{{{ncols}}}{{c}}{{\\textit{{{granularity_disp}}}}} \\\\")
        lines.append("\\midrule")
        for i, metric in enumerate(metrics_present):
            if i > 0:
                lines.append("\\cmidrule{3-" + str(ncols) + "}")
            axis_matched = metric in AXIS_MATCHED_METRICS
            metric_disp = (
                None if axis_matched
                else f"\\multirow{{{len(CORRELATION_AXIS_ORDER)}}}{{*}}{{{CORRELATION_METRIC_DISPLAY[metric]}}}"
            )
            ref_symbol = "$\\checkmark$" if METRIC_HAS_REF.get(metric, False) else "$\\times$"
            ref_disp = f"\\multirow{{{len(CORRELATION_AXIS_ORDER)}}}{{*}}{{{ref_symbol}}}"
            for k, axis in enumerate(CORRELATION_AXIS_ORDER):
                cells = []
                for lang in CORRELATION_LANG_COLS:
                    val = pivot.get((axis, metric, granularity, lang))
                    cell = f"{val * 100:.1f}" if val is not None else "--"
                    if val is not None and best_lang.get((axis, granularity, lang)) == val:
                        cell = f"\\textbf{{{cell}}}"
                    cells.append(cell)
                if axis_matched:
                    row_metric_cell = (
                        f"{CORRELATION_METRIC_DISPLAY[metric]}\\textsubscript{{{CORRELATION_AXIS_DISPLAY[axis]}}}"
                    )
                else:
                    row_metric_cell = metric_disp if k == 0 else ""
                ref_cell = ref_disp if k == 0 else ""
                lines.append(
                    " & ".join([row_metric_cell, ref_cell, CORRELATION_AXIS_DISPLAY[axis]] + cells) + " \\\\"
                )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append(
        "\\caption{Correlation (\\%) between human Style/Meaning/Overall judgments and automatic "
        "metrics, averaged over the eight languages and four systems in the human evaluation. "
        "Segment-level: mean Kendall's $\\tau$ (within-item, across systems). System-level: "
        "soft pairwise accuracy (SPA).}"
    )
    lines.append("\\label{tab:correlation_per_language}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


def create_latex_tables(comet_csv, chrf_csv, asr_csv, llm_judge_style_csv, llm_judge_meaning_csv, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "comet_table.tex"), "w", encoding="utf-8") as f:
        f.write(build_translation_table(comet_csv, "comet"))

    with open(os.path.join(output_dir, "comet_main_table.tex"), "w", encoding="utf-8") as f:
        f.write(build_main_translation_table(comet_csv, "comet"))

    with open(os.path.join(output_dir, "chrf_table.tex"), "w", encoding="utf-8") as f:
        f.write(build_translation_table(chrf_csv, "chrf"))

    with open(os.path.join(output_dir, "chrf_main_table.tex"), "w", encoding="utf-8") as f:
        f.write(build_main_translation_table(chrf_csv, "chrf"))

    with open(os.path.join(output_dir, "chrf_main_all_table.tex"), "w", encoding="utf-8") as f:
        f.write(build_all_models_std_table(chrf_csv, "chrf"))

    with open(os.path.join(output_dir, "comet_main_all_table.tex"), "w", encoding="utf-8") as f:
        f.write(build_all_models_std_table(comet_csv, "comet"))

    with open(os.path.join(output_dir, "asr_table.tex"), "w", encoding="utf-8") as f:
        f.write(build_asr_table(asr_csv))

    with open(os.path.join(output_dir, "llm_judge_table.tex"), "w", encoding="utf-8") as f:
        f.write(build_llm_judge_combined_table(llm_judge_style_csv, llm_judge_meaning_csv))

    correlation_csv = "results/correlation/correlation_results.csv"
    if os.path.exists(correlation_csv):
        with open(os.path.join(output_dir, "correlation_table.tex"), "w", encoding="utf-8") as f:
            f.write(build_correlation_table(correlation_csv))
        with open(os.path.join(output_dir, "correlation_per_language_table.tex"), "w", encoding="utf-8") as f:
            f.write(build_correlation_per_language_table(correlation_csv))

    node_combos_csv = "results/node_combos/analysis_node_combos_en-x_structured.csv"
    if os.path.exists(node_combos_csv):
        with open(os.path.join(output_dir, "node_combos_table.tex"), "w", encoding="utf-8") as f:
            f.write(build_node_combos_table(node_combos_csv))

    print(f"Saved comet, chrf, asr, llm-judge, correlation, and node-combos tables to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render COMET/chrF/ASR/LLM-judge result tables as LaTeX.")
    parser.add_argument("--comet", default="results/translations/evaluation_results_comet_table.csv")
    parser.add_argument("--chrf", default="results/translations/evaluation_results_chrf_table.csv")
    parser.add_argument("--asr", default="results/ASR_results/asr_scores.csv")
    parser.add_argument("--llm-judge-style", default="results/llm_judge/llm_judge_style_table.csv")
    parser.add_argument("--llm-judge-meaning", default="results/llm_judge/llm_judge_meaning_table.csv")
    parser.add_argument("--output-dir", default="results/latex_tables")
    args = parser.parse_args()
    create_latex_tables(
        args.comet, args.chrf, args.asr, args.llm_judge_style, args.llm_judge_meaning, args.output_dir
    )
