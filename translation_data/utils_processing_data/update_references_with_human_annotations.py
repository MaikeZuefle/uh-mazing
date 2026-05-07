"""
Updates references.csv with human-annotated disfluent translations.

For each (utt_id, lang) in references_human_annotated.csv, picks the majority-vote
text among all annotators (ties broken by first occurrence). Overwrites the
corresponding {LANG}_disfluent column in references.csv.

ID mapping: references.csv uses "sw2005_A_19", annotated file uses "sw02005_A_19".
The annotated utt_id is converted to the references.csv format by stripping the
leading zero: sw0XXXXX -> swXXXXX.
"""

import csv
import re
from collections import Counter
from pathlib import Path

REFERENCES_CSV = Path(__file__).parent.parent / "references.csv"
ANNOTATED_CSV = Path(__file__).parent.parent / "references_human_annotated.csv"

LANG_TO_COLUMN = {
    "AR": "AR_disfluent",
    "CS": "CS_disfluent",
    "DE": "DE_disfluent",
    "ES": "ES_disfluent",
    "FR": "FR_disfluent",
    "HI": "HI_disfluent",
    "IT": "IT_disfluent",
    "ZH": "ZH_disfluent",
}


def normalize_utt_id(utt_id):
    """Convert annotated ID format to references.csv format: sw02005_A_19 -> sw2005_A_19."""
    return re.sub(r"^sw0", "sw", utt_id)


def pick_majority(texts):
    """Return the most common text; ties broken by first occurrence."""
    counts = Counter(texts)
    max_count = max(counts.values())
    for text in texts:
        if counts[text] == max_count:
            return text


def load_annotations(annotated_csv):
    """
    Returns dict: (ref_id, lang) -> text_annotated  (majority-voted).
    """
    from collections import defaultdict
    raw = defaultdict(list)  # (ref_id, lang) -> [text, ...]

    with open(annotated_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lang = row["lang"].strip()
            if lang not in LANG_TO_COLUMN:
                continue
            ref_id = normalize_utt_id(row["utt_id"].strip())
            text = row["text_annotated"].strip()
            raw[(ref_id, lang)].append(text)

    return {key: pick_majority(texts) for key, texts in raw.items()}


def update_references(references_csv, annotated_csv):
    annotations = load_annotations(annotated_csv)

    with open(references_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    updated = 0
    for row in rows:
        ref_id = row["ID"].strip()
        for lang, col in LANG_TO_COLUMN.items():
            key = (ref_id, lang)
            if key in annotations:
                row[col] = annotations[key]
                updated += 1

    with open(references_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Updated {updated} cells across {len(rows)} rows in {references_csv}")
    missing = [k for k in annotations if k not in {(r["ID"], lang) for r in rows for lang in LANG_TO_COLUMN}]
    if missing:
        print(f"Warning: {len(missing)} annotations had no matching row in references.csv:")
        for k in missing[:10]:
            print(f"  {k}")


if __name__ == "__main__":
    update_references(REFERENCES_CSV, ANNOTATED_CSV)
