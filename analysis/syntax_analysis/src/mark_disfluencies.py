"""Mark disfluencies in system outputs, or in references with their markup removed, using an LLM."""
import argparse
import csv
import json
import os
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from markup import tokenize_and_flag

LANG_NAMES = {"AR": "Arabic", "CS": "Czech", "DE": "German", "ES": "Spanish",
              "FR": "French", "HI": "Hindi", "IT": "Italian", "ZH": "Chinese"}
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_FIELDS = ["system", "lang", "ID", "n_tokens", "disfluent_token_indices", "marked_text"]
DEGENERATE_REPEATS = 8
TARGET_SCRIPT = {"AR": "ARABIC", "HI": "DEVANAGARI", "ZH": "CJK"}
MIN_TARGET_SCRIPT_SHARE = 0.5
MAX_SOURCE_OVERLAP = 0.4
CJK = r"[　-〿一-鿿＀-￯]"


def is_punct(char, keep_hyphen=False):
    return unicodedata.category(char).startswith("P") and not (keep_hyphen and char == "-")


def tokenize(text, lang):
    """Split text into tokens like the references: punctuation separate, Chinese per character."""
    if lang == "ZH":
        return re.findall(r"[A-Za-z0-9]+(?:['’.\-][A-Za-z0-9]+)*|\S", text)
    text = re.sub(f"({CJK})", r" \1 ", text)
    tokens = []
    for chunk in text.split():
        start, end = 0, len(chunk)
        while start < end and is_punct(chunk[start]):
            start += 1
        while end > start and is_punct(chunk[end - 1], keep_hyphen=True):
            end -= 1
        if start:
            tokens.append(chunk[:start])
        if start < end:
            tokens.append(chunk[start:end])
        if end < len(chunk):
            tokens.append(chunk[end:])
    return tokens


def max_repeats(tokens, max_n=6):
    """Most consecutive repetitions of any word n-gram, used to detect degenerate repetition loops."""
    words = [t for t in tokens if any(c.isalnum() for c in t)]
    best = 1
    for n in range(1, max_n + 1):
        i = 0
        while i + n <= len(words):
            k = 1
            while words[i + k * n:i + (k + 1) * n] == words[i:i + n]:
                k += 1
            best = max(best, k)
            i += (k - 1) * n if k > 1 else 1
    return best


def words(text):
    return re.findall(r"[^\W\d_]+", text.replace("_", " ").lower())


def wrong_language(text, lang, source):
    """Whether an output is not in the target language: wrong script for AR/HI/ZH, or mostly words of the English source."""
    if lang in TARGET_SCRIPT:
        letters = [c for c in text if c.isalpha()]
        in_script = sum(TARGET_SCRIPT[lang] in unicodedata.name(c, "") for c in letters)
        if letters and in_script / len(letters) < MIN_TARGET_SCRIPT_SHARE:
            return True
    source_words = set(words(source))
    output_words = words(text)
    return bool(output_words) and sum(w in source_words for w in output_words) / len(output_words) > MAX_SOURCE_OVERLAP


def exclusion_reason(text, tokens, lang, source):
    if max_repeats(tokens) >= DEGENERATE_REPEATS:
        return "repetition_loop"
    if wrong_language(text, lang, source):
        return "wrong_language"
    return None


def load_items(mode, systems_csv, references_csv, langs):
    """Items to mark, plus the system outputs excluded as repetition loops or not in the target language."""
    refs = pd.read_csv(references_csv).set_index("ID")
    items, excluded = [], []
    if mode == "references":
        for lang in langs:
            for ref_id, text in refs[f"{lang}_disfluent"].items():
                tokens = [word for word, _ in tokenize_and_flag(str(text))]
                items.append({"system": "reference", "lang": lang, "ID": ref_id, "tokens": tokens})
    else:
        outputs = pd.read_csv(systems_csv)
        for _, row in outputs[outputs["lang"].isin(langs)].iterrows():
            text = "" if pd.isna(row["translation"]) else str(row["translation"]).replace("_", " ").strip()
            if not text:
                continue
            ref_id = re.sub(r"^sw0", "sw", row["item_id"].removesuffix(".wav"))
            item = {"system": row["system"].split("/")[-1], "lang": row["lang"], "ID": ref_id,
                    "tokens": tokenize(text, row["lang"])}
            reason = exclusion_reason(text, item["tokens"], row["lang"], refs.loc[ref_id, "EN_disfluent"])
            if reason:
                excluded.append({"system": item["system"], "lang": item["lang"], "ID": ref_id, "reason": reason})
            else:
                items.append(item)
    for item in items:
        item["source"] = refs.loc[item["ID"], "EN_disfluent"]
    return items, pd.DataFrame(excluded, columns=["system", "lang", "ID", "reason"])


def render_prompt(template, item):
    numbered = "\n".join(f"{i}: {token}" for i, token in enumerate(item["tokens"]))
    return (template.replace("{source}", item["source"])
            .replace("{language}", LANG_NAMES[item["lang"]])
            .replace("{tokens}", numbered))


def call_llm(client, model, prompt):
    for attempt in range(4):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"},
            )
            return json.loads(response.choices[0].message.content)
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** (attempt + 1))


def parse_indices(answer, n_tokens):
    """Valid token indices from the LLM answer, ignoring anything malformed."""
    raw = answer.get("disfluent_token_indices") if isinstance(answer, dict) else None
    indices = set()
    for value in raw if isinstance(raw, list) else []:
        if isinstance(value, str) and value.strip().isdigit():
            value = int(value)
        if isinstance(value, int) and not isinstance(value, bool) and 0 <= value < n_tokens:
            indices.add(value)
    return sorted(indices)


def marked_text(tokens, indices):
    return " ".join(f"_{token}_" if i in indices else token for i, token in enumerate(tokens))


def load_done(out_csv):
    if not os.path.exists(out_csv):
        return set()
    done = pd.read_csv(out_csv)
    return set(zip(done["system"], done["lang"], done["ID"]))


def write_reference_format(out_csv, excluded, references_csv, out_dir):
    """Write one CSV per system in the references.csv format, leaving excluded outputs empty and languages with unmarked utterances out."""
    marked = pd.read_csv(out_csv, keep_default_na=False)
    skip = set(zip(excluded["system"], excluded["lang"], excluded["ID"]))
    keep = pd.Series([key not in skip for key in zip(marked["system"], marked["lang"], marked["ID"])],
                     index=marked.index, dtype=bool)
    marked = marked.loc[keep]
    refs = pd.read_csv(references_csv)
    for system, rows in marked.groupby("system"):
        table = refs[["ID", "EN_disfluent"]].copy()
        for lang, lang_rows in rows.groupby("lang"):
            texts = lang_rows.set_index("ID")["marked_text"]
            n_excluded = sum(1 for s, l, _ in skip if s == system and l == lang)
            if len(texts) + n_excluded != len(table):
                print(f"{system} {lang}: {len(texts)}/{len(table) - n_excluded} utterances marked, column left out")
                continue
            table[f"{lang}_disfluent"] = table["ID"].map(texts)
        path = os.path.join(out_dir, f"{system}.csv")
        table.to_csv(path, index=False)
        print(f"Wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["systems", "references"], default="systems")
    parser.add_argument("--systems_csv",
                        default=os.path.join(HERE, "../data/system_outputs.csv"))
    parser.add_argument("--references_csv", default=os.path.join(HERE, "../../../translation_data/references.csv"))
    parser.add_argument("--prompt", default=os.path.join(HERE, "prompts/mark_disfluencies.txt"))
    parser.add_argument("--output", default=os.path.join(HERE, "../data/marked"))
    parser.add_argument("--model", default="gpt-5.2-2025-12-11")
    parser.add_argument("--langs", nargs="+", default=list(LANG_NAMES))
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of items per system and language")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel API requests")
    parser.add_argument("--dry_run", action="store_true", help="Print the first prompt and exit")
    args = parser.parse_args()

    template = open(args.prompt, encoding="utf-8").read()
    items, excluded = load_items(args.mode, args.systems_csv, args.references_csv, args.langs)
    os.makedirs(args.output, exist_ok=True)
    if args.mode == "systems":
        excluded.to_csv(os.path.join(args.output, "excluded_outputs.csv"), index=False)
        print(f"{len(excluded)} outputs excluded: {excluded['reason'].value_counts().to_dict()}")
    if args.limit:
        items = pd.DataFrame(items).groupby(["system", "lang"]).head(args.limit).to_dict("records")
    if args.dry_run:
        print(render_prompt(template, items[0]))
        return

    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), organization=os.getenv("OPENAI_ORG_KEY"),
                    project=os.getenv("OPENAI_PROJECT_KEY"))

    out_csv = os.path.join(args.output, f"marked_{args.mode}.csv")
    done = load_done(out_csv)
    todo = [item for item in items if (item["system"], item["lang"], item["ID"]) not in done]
    print(f"{len(items)} items, {len(todo)} still to mark")

    new_file = not os.path.exists(out_csv)
    failed = 0
    with open(out_csv, "a", newline="", encoding="utf-8") as f, ThreadPoolExecutor(args.workers) as pool:
        writer = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        if new_file:
            writer.writeheader()
        futures = {pool.submit(call_llm, client, args.model, render_prompt(template, item)): item for item in todo}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Marking {args.mode}"):
            item = futures[future]
            try:
                indices = parse_indices(future.result(), len(item["tokens"]))
            except Exception as e:
                failed += 1
                tqdm.write(f"{item['system']} {item['lang']} {item['ID']}: failed ({e})")
                continue
            writer.writerow({"system": item["system"], "lang": item["lang"], "ID": item["ID"],
                             "n_tokens": len(item["tokens"]), "disfluent_token_indices": json.dumps(indices),
                             "marked_text": marked_text(item["tokens"], set(indices))})
            f.flush()
    if failed:
        print(f"{failed} items failed, rerun the same command to retry them")

    if args.mode == "systems":
        write_reference_format(out_csv, excluded, args.references_csv, args.output)


if __name__ == "__main__":
    main()
