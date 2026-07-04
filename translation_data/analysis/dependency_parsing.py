#!/usr/bin/env python3
"""
Dependency parsing of disfluent reference translations.

Two complementary analyses are run for each language:

1. Disfluent parse  (dependency_results.csv)
   The full disfluent text (underscores stripped) is parsed with Stanza.
   Stanza is called with tokenize_pretokenized=True so its token indices align
   1:1 with the input list built from the _word_ annotation.  Each token row
   records its deprel/upos and whether it was marked as disfluent.
   Caveat: the parser sees disfluent input, so parse quality may be lower.

2. Fluent-neighbor analysis  (fluent_adjacency_results.csv)
   Only the fluent tokens are extracted and parsed.  For each disfluent token,
   the deprel/upos of its immediate left and right fluent neighbors are recorded.
   This approach is unaffected by parser quality on disfluent text.

Output files (written to --output dir):
  dependency_results.csv       - one row per token: utt_id, lang, token, upos,
                                 deprel, head, is_disfluent
  deprel_summary.csv           - deprel+upos counts for disfluent tokens per lang
  fluent_adjacency_results.csv - one row per disfluent token: utt_id, lang,
                                 disfl_token, left/right neighbor token+upos+deprel
"""
import argparse
import os
import re

import pandas as pd
import stanza

LANG_COL_MAP = {
    "en": "EN_disfluent",
    "ar": "AR_disfluent",
    "cs": "CS_disfluent",
    "de": "DE_disfluent",
    "es": "ES_disfluent",
    "fr": "FR_disfluent",
    "hi": "HI_disfluent",
    "it": "IT_disfluent",
    "zh": "ZH_disfluent",
    "sw": "SW_disfluent",
}


def tokenize_and_flag(text):
    """
    Split annotated disfluent text into (clean_word, is_disfluent) pairs.

    Handles:
      _word_      -> (word, True)
      _word_,     -> (word, True)  then  (',', False)  [trailing punct detached]
      _,_         -> (',', True)
      _, _        -> (',', True)   [split annotation artifact, underscore stripped]
      plain word  -> (word, False)
      lone _      -> skipped
    """
    # Detach trailing punctuation from closing underscore: _word_, -> _word_ ,
    text = re.sub(r'_([^_\s]+)_([,.])', r'_\1_ \2', text)
    result = []
    for token in text.split():
        if token.startswith('_') and token.endswith('_') and len(token) > 1:
            word = token[1:-1]
            if word:
                result.append((word, True))
        elif token == '_':
            pass  # artifact from _, _ split
        elif '_' in token:
            # Partial annotation: strip underscores and treat as disfluent
            word = token.replace('_', '')
            if word:
                result.append((word, True))
        else:
            result.append((token, False))
    return result


def parse_utterances(nlp, df, lang, col):
    rows = []
    for _, row in df.iterrows():
        token_flags = tokenize_and_flag(str(row[col]))
        if not token_flags:
            continue
        clean_tokens = [w for w, _ in token_flags]
        flags = [f for _, f in token_flags]
        try:
            doc = nlp([clean_tokens])  # list-of-sentences, each a list of tokens
        except Exception as e:
            print(f"  Parse error {row['ID']} ({lang}): {e}")
            continue
        if not doc.sentences:
            continue
        sent = doc.sentences[0]
        for word in sent.words:
            # word.parent.id is 1-indexed surface token position; handles MWT expansion
            surf_idx = word.parent.id - 1
            is_dis = flags[surf_idx] if surf_idx < len(flags) else False
            head_idx = word.head - 1
            head_text = sent.words[head_idx].text if 0 <= head_idx < len(sent.words) else "ROOT"
            rows.append({
                "utt_id": row["ID"],
                "lang": lang,
                "token": word.text,
                "upos": word.upos,
                "deprel": word.deprel,
                "head": head_text,
                "is_disfluent": is_dis,
            })
    return rows


def fluent_adjacency(nlp, df, lang, col):
    """
    For each disfluent token, record the deprel/upos of its nearest fluent
    neighbors (left and right) in a parse of the fluent-only token sequence.
    """
    rows = []
    for _, row in df.iterrows():
        token_flags = tokenize_and_flag(str(row[col]))
        if not token_flags:
            continue

        # Fluent tokens with their original positions in token_flags
        fluent_positions = [(i, w) for i, (w, f) in enumerate(token_flags) if not f]
        fluent_tokens = [w for _, w in fluent_positions]
        orig_pos = [i for i, _ in fluent_positions]

        if not fluent_tokens:
            continue

        try:
            doc = nlp([fluent_tokens])
        except Exception as e:
            print(f"  Parse error {row['ID']} ({lang}): {e}")
            continue
        if not doc.sentences:
            continue
        sent = doc.sentences[0]

        # Map fluent token index -> (deprel, upos, text); use parent.id for MWT safety
        fluent_parse = {}
        for word in sent.words:
            fi = word.parent.id - 1
            if fi not in fluent_parse:
                fluent_parse[fi] = (word.deprel, word.upos, word.text)

        # For each disfluent token, find nearest fluent neighbors by original position
        for i, (w, is_dis) in enumerate(token_flags):
            if not is_dis:
                continue

            left_fi = next((fi for fi in range(len(orig_pos) - 1, -1, -1) if orig_pos[fi] < i), None)
            right_fi = next((fi for fi in range(len(orig_pos)) if orig_pos[fi] > i), None)

            def info(fi):
                if fi is None or fi not in fluent_parse:
                    return None, None, None
                deprel, upos, text = fluent_parse[fi]
                return text, upos, deprel

            lt, lu, ld = info(left_fi)
            rt, ru, rd = info(right_fi)
            rows.append({
                "utt_id": row["ID"],
                "lang": lang,
                "disfl_token": w,
                "left_token": lt, "left_upos": lu, "left_deprel": ld,
                "right_token": rt, "right_upos": ru, "right_deprel": rd,
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to references.csv")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--stanza_dir", default=None,
                        help="Stanza resources directory (default: ~/stanza_resources)")
    parser.add_argument("--langs", nargs="+", default=None,
                        help="Languages to process (default: all). E.g. --langs en de zh")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    stanza_kwargs = {"dir": args.stanza_dir} if args.stanza_dir else {}

    lang_map = {k: v for k, v in LANG_COL_MAP.items()
                if args.langs is None or k in args.langs}

    df = pd.read_csv(args.input)
    all_dep_rows = []
    all_adj_rows = []

    for lang, col in lang_map.items():
        if col not in df.columns:
            print(f"Skipping {lang}: column {col} not in CSV")
            continue
        print(f"\n[{lang}] Downloading model...")
        stanza.download(lang, verbose=False, **stanza_kwargs)
        print(f"[{lang}] Parsing {len(df)} utterances...")
        nlp = stanza.Pipeline(lang, processors="tokenize,pos,depparse",
                               tokenize_pretokenized=True, verbose=False, **stanza_kwargs)

        dep_rows = parse_utterances(nlp, df, lang, col)
        all_dep_rows.extend(dep_rows)
        n_dis = sum(1 for r in dep_rows if r["is_disfluent"])
        print(f"[{lang}] disfluent parse: {len(dep_rows)} tokens, {n_dis} disfluent")

        adj_rows = fluent_adjacency(nlp, df, lang, col)
        all_adj_rows.extend(adj_rows)
        print(f"[{lang}] fluent adjacency: {len(adj_rows)} disfluent tokens recorded")

    results = pd.DataFrame(all_dep_rows)
    out_csv = os.path.join(args.output, "dependency_results.csv")
    results.to_csv(out_csv, index=False)
    print(f"\nSaved {len(results)} rows -> {out_csv}")

    dis = results[results["is_disfluent"]]
    summary = (
        dis.groupby(["lang", "deprel", "upos"])
        .size()
        .reset_index(name="count")
        .sort_values(["lang", "count"], ascending=[True, False])
    )
    summary_csv = os.path.join(args.output, "deprel_summary.csv")
    summary.to_csv(summary_csv, index=False)
    print(f"Saved deprel summary -> {summary_csv}")

    adj = pd.DataFrame(all_adj_rows)
    adj_csv = os.path.join(args.output, "fluent_adjacency_results.csv")
    adj.to_csv(adj_csv, index=False)
    print(f"Saved fluent adjacency -> {adj_csv}")


if __name__ == "__main__":
    main()
