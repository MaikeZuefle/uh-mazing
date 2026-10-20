"""
Export the Uh-Mazing dataset for the LDC release.

Follows the LDC directory layout, all CSVs share the ID column:
    uh_mazing/
        data/speech/                           audio, one file per utterance
        data/transcripts/en_transcripts.csv    English fluent and disfluent source
        data/annotations/disfluency_types.csv  EDITED, INTJ, PRN counts and node-combo variants
        data/text/translations.csv             disfluent and fluent translation per language
        docs/                                  copied from ldc_docs/

The disfluent translations come from references.csv (which includes the human
post-edits). The fluent translations are derived by removing the _marked_
disfluent spans and detokenizing the result. All text is NFC-normalized and
written as UTF-8 with Unix line endings.

Audio: all snippets are written as 16 kHz, 16-bit, mono FLAC (LDC requires a
uniform audio format). Snippets at another sample rate (the 8 kHz ones cut
from the original Switchboard conversations) are upsampled. Filenames follow
the ID notation of references.csv (sw02005_A_19.wav -> sw2005_A_19.flac).
The snippets were originally cut at whole seconds, which cut off the end of
many utterances. The end times in translation-dataset-with-timestamps.csv (and
so the snippets in translation_dataset_audio_preprocessed/) were adjusted by
listening, from -0.4 to +1.38 s relative to the original cut.

Usage (from translation_data/):
    python utils_processing_data/export_ldc.py
"""

import argparse
import os
import re
import shutil
import unicodedata

import pandas as pd
import soundfile as sf
import torch
import torchaudio

TARGET_SR = 16000
LANGS = ["AR", "CS", "DE", "ES", "FR", "HI", "IT", "ZH"]
NODE_COMBO_COLUMNS = [
    "edited_count", "intj_count", "prn_count",
    "edited_nodes_only", "intj_only", "prn_only",
    "edited_intj", "intj_prn", "edited_prn", "edited_prn_intj",
]

COMMA = r"[,،，]"
SENTENCE_END = r"[.。।?？؟!！]"


def to_csv_id(filename):
    return re.sub(r"^sw0", "sw", os.path.splitext(filename)[0])


def clean_text(text):
    if not isinstance(text, str):
        return text
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", text).strip()


def remove_disfluencies(text):
    # _+ also catches spans with a doubled closing underscore (_word__)
    text = re.sub(r"_[^_]*_+", " ", text)
    text = re.sub(r"\s+", " ", text)
    # collapse commas left behind by removed spans: "that , , might" -> "that , might"
    text = re.sub(rf"({COMMA})(\s*{COMMA})+", r"\1", text)
    # drop a comma directly before a sentence end: "done , ." -> "done ."
    text = re.sub(rf"{COMMA}\s*({SENTENCE_END})", r"\1", text)
    # drop a comma directly after a sentence end: "beletrie . , teď" -> "beletrie . teď"
    text = re.sub(rf"({SENTENCE_END})(\s*{COMMA})+", r"\1", text)
    # drop punctuation left at the start of the sentence
    text = re.sub(rf"^(\s*({COMMA}|{SENTENCE_END}))+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def detokenize(text, lang):
    # the translations are tokenized: every token, including punctuation, is space-separated
    if lang == "ZH":
        # no spaces next to Chinese characters or full-width punctuation
        cjk = r"[　-〿一-鿿＀-￯《》“”·、]"
        text = re.sub(rf"(?<={cjk}) +| +(?={cjk})", "", text)

    # French keeps a space before ? ! : ;
    no_space_before = r",.،।…" if lang == "FR" else r",.،।…?!:;؟"
    text = re.sub(rf" +(?=[{re.escape(no_space_before)}])", "", text)
    text = re.sub(r" +(?=[)»“”])", "", text)
    text = re.sub(r"(?<=[(«„¿]) +", "", text)
    # C 'est -> C'est
    text = re.sub(r"(?<=\w) '(?=\w)", "'", text)

    # straight double quotes: no space after an opening or before a closing one.
    # A straight quote after an unclosed „ closes it (Czech „TEXAS HIGHWAY").
    out, low_open, straight_open = [], False, False
    for char in text:
        if char == "„":
            low_open = True
        elif char == "“":
            low_open = False
        elif char == '"':
            if low_open or straight_open:
                while out and out[-1] == " ":
                    out.pop()
                low_open = straight_open = False
            else:
                straight_open = True
        elif char == " " and out and out[-1] == '"' and straight_open:
            continue
        out.append(char)
    return "".join(out)


def export_text(references_csv, node_combos_csv, data_folder):
    refs = pd.read_csv(references_csv)
    combos = pd.read_csv(node_combos_csv)[["ID"] + NODE_COMBO_COLUMNS]
    df = refs[["ID", "EN_fluent", "EN_disfluent"]].merge(
        combos, on="ID", how="left", validate="one_to_one"
    )

    for lang in LANGS:
        disfluent = refs[f"{lang}_disfluent"].map(clean_text)
        df[f"{lang}_disfluent"] = disfluent
        df[f"{lang}_fluent"] = disfluent.map(lambda t: detokenize(remove_disfluencies(t), lang))

        for sample_id, text in zip(df["ID"], disfluent):
            if "_" not in text:
                print(f"WARNING: {lang} {sample_id} has no disfluency markup, fluent = disfluent")

    for col in df.columns:
        df[col] = df[col].map(clean_text)

    translation_columns = [f"{lang}_{kind}" for lang in LANGS for kind in ("disfluent", "fluent")]
    tables = {
        os.path.join(data_folder, "transcripts", "en_transcripts.csv"): ["EN_fluent", "EN_disfluent"],
        os.path.join(data_folder, "annotations", "disfluency_types.csv"): NODE_COMBO_COLUMNS,
        os.path.join(data_folder, "text", "translations.csv"): translation_columns,
    }
    for out_csv, columns in tables.items():
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        df[["ID"] + columns].to_csv(out_csv, index=False, encoding="utf-8", lineterminator="\n")
        print(f"Wrote {len(df)} rows to {out_csv}")


def export_audio(in_folder, out_folder):
    os.makedirs(out_folder, exist_ok=True)

    for filename in sorted(os.listdir(in_folder)):
        if not filename.endswith(".wav"):
            continue

        # soundfile instead of torchaudio.load, which needs FFmpeg via torchcodec
        audio, sample_rate = sf.read(os.path.join(in_folder, filename), dtype="float32")
        if sample_rate != TARGET_SR:
            audio = torchaudio.functional.resample(torch.from_numpy(audio), sample_rate, TARGET_SR)
            audio = torch.clamp(audio, -1.0, 1.0).numpy()
            print(f"Upsampled {filename} from {sample_rate} to {TARGET_SR} Hz")

        out_path = os.path.join(out_folder, f"{to_csv_id(filename)}.flac")
        sf.write(out_path, audio, TARGET_SR, subtype="PCM_16", format="FLAC")

    print(f"Exported audio to {out_folder}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--references_csv", default="references.csv")
    parser.add_argument("--node_combos_csv", default="uh-mazing-with-node-combos.csv")
    parser.add_argument("--audio_folder", default="translation_dataset_audio_preprocessed")
    parser.add_argument("--docs_folder", default="ldc_docs")
    parser.add_argument("--out_folder", default="uh_mazing")
    args = parser.parse_args()

    data_folder = os.path.join(args.out_folder, "data")
    export_text(args.references_csv, args.node_combos_csv, data_folder)
    export_audio(args.audio_folder, os.path.join(data_folder, "speech"))
    shutil.copytree(args.docs_folder, os.path.join(args.out_folder, "docs"), dirs_exist_ok=True)


if __name__ == "__main__":
    main()
