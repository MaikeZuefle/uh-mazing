import os
import pandas as pd

LANG_COLS = ["zh", "es", "hi", "fr", "de", "it", "sw", "cs", "ar"]

# audio-gemini / audio-chatgpt (the "with extra context" audio condition) are intentionally
# excluded from the tables below -- only the audio-noextra condition is reported.
EXCLUDED_MODELS = {"audio-gemini", "audio-chatgpt"}

# Single source of truth for each model's table category, display name, and subgroup
# (used for the partial \cmidrule within a category). One entry per model -- nothing
# below should need to be redefined per table.
MODELS = {
    "canary-llama": {"category": "Cascaded", "display": "\\textsc{Llama}\\textsubscript{Canary}", "subgroup": "asr"},
    "canary-tower": {"category": "Cascaded", "display": "\\textsc{Tower}\\textsubscript{Canary}", "subgroup": "asr"},
    "whisper-llama": {"category": "Cascaded", "display": "\\textsc{Llama}\\textsubscript{Whisper}", "subgroup": "asr"},
    "whisper-tower": {"category": "Cascaded", "display": "\\textsc{Tower}\\textsubscript{Whisper}", "subgroup": "asr"},
    "gold-transcript-llama": {"category": "Cascaded", "display": "\\textsc{Llama}\\textsubscript{Gold}", "subgroup": "gold"},
    "gold-transcript-tower": {"category": "Cascaded", "display": "\\textsc{Tower}\\textsubscript{Gold}", "subgroup": "gold"},
    "owsm": {"category": "End-to-end", "display": "\\textsc{OWSM}"},
    "canary": {"category": "End-to-end", "display": "\\textsc{Canary}"},
    "phi": {"category": "SpeechLLM", "display": "\\textsc{Phi-4}"},
    "qwen": {"category": "SpeechLLM", "display": "\\textsc{Qwen2.5}"},
    "gold-text-gemini": {"category": "Commercial", "display": "\\textsc{Gemini}\\textsubscript{Gold}", "subgroup": "gold"},
    "gold-text-chatgpt": {"category": "Commercial", "display": "\\textsc{ChatGPT}\\textsubscript{Gold}", "subgroup": "gold"},
    "asr-text-gemini": {"category": "Commercial", "display": "\\textsc{Gemini}\\textsubscript{ASR}", "subgroup": "asr"},
    "asr-text-chatgpt": {"category": "Commercial", "display": "\\textsc{ChatGPT}\\textsubscript{ASR}", "subgroup": "asr"},
    "audio-noextra-gemini": {"category": "Commercial", "display": "\\textsc{Gemini}\\textsubscript{Audio}", "subgroup": "audio"},
    "audio-noextra-chatgpt": {"category": "Commercial", "display": "\\textsc{ChatGPT}\\textsubscript{Audio}", "subgroup": "audio"},
    "gold-transcript-chatgpt": {"category": "Commercial", "display": "\\textsc{ChatGPT}\\textsubscript{Gold}", "subgroup": "gold"},
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


def colorize(value_str, worst, best, color="blue", max_intensity=25):
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


def build_row_lines(rows):
    """Build LaTeX row lines, grouping contiguous rows by model category into a leading
    \\multirow column, also \\multirow-ing the model column over its standard/disfluency-aware
    prompt pair, colorizing each language column by its min/max within these rows, inserting a
    \\midrule between category groups, and a partial \\cmidrule (skipping the category column)
    between known subgroups within a category."""
    ncols = 3 + len(LANG_COLS)
    col_bounds = {}
    for lang in LANG_COLS:
        values = [float(row[lang]) for row in rows if not pd.isna(row[lang]) and row[lang] != ""]
        if values:
            col_bounds[lang] = compute_bounds(values, higher_is_better=True)

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
                lines.append(f"\\cmidrule{{2-{ncols}}}")
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
                cell = fmt_cell(row[lang])
                if lang in col_bounds:
                    worst, best = col_bounds[lang]
                    cell = colorize(cell, worst, best)
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
        if section_name == "X → EN TRANSLATIONS":
            section_rows = [row for row in section_rows if row["prompt_type"] != "disfluency_aware"]
        lines.extend(build_row_lines(section_rows))
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


def build_main_comet_table(csv_path):
    """Condensed main-paper table: standard prompt only, EN-to-X only, the single best
    automatic-transcript cascaded model (no gold transcript), and audio-input commercial
    models only (no gold-/ASR-text commercial conditions)."""
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
    lines.append(
        "\\caption{COMET-Kiwi scores (standard prompt, English$\\to$X) for the best "
        "automatic-transcript cascaded system, end-to-end, speechLLM, and audio-input "
        "commercial models. Full results, including disfluency-aware prompting, gold "
        "transcripts, and X$\\to$English, are in the appendix.}"
    )
    lines.append("\\label{tab:comet_main}")
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


def create_latex_tables(comet_csv, chrf_csv, asr_csv, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "comet_table.tex"), "w", encoding="utf-8") as f:
        f.write(build_translation_table(comet_csv, "comet"))

    with open(os.path.join(output_dir, "comet_main_table.tex"), "w", encoding="utf-8") as f:
        f.write(build_main_comet_table(comet_csv))

    with open(os.path.join(output_dir, "chrf_table.tex"), "w", encoding="utf-8") as f:
        f.write(build_translation_table(chrf_csv, "chrf"))

    with open(os.path.join(output_dir, "asr_table.tex"), "w", encoding="utf-8") as f:
        f.write(build_asr_table(asr_csv))

    print(f"Saved comet_table.tex, comet_main_table.tex, chrf_table.tex, asr_table.tex to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render COMET/chrF/ASR result tables as LaTeX.")
    parser.add_argument("--comet", default="results/translations/evaluation_results_comet_table.csv")
    parser.add_argument("--chrf", default="results/translations/evaluation_results_chrf_table.csv")
    parser.add_argument("--asr", default="results/ASR_results/asr_scores.csv")
    parser.add_argument("--output-dir", default="results/latex_tables")
    args = parser.parse_args()
    create_latex_tables(args.comet, args.chrf, args.asr, args.output_dir)
