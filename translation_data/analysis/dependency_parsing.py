#!/usr/bin/env python3
"""
Dependency parsing of disfluent reference translations.

This script only does RAW parsing -- it makes no interpretive choices (what
counts as a "span", how to find a "neighbor", whether to skip punctuation,
etc). Those are all analysis decisions and belong in analyse_dependencies.py,
which reads the CSVs below. The split matters because Stanza parsing is the
slow, re-download-a-model, re-run-on-a-cluster part of the pipeline: every
raw fact either parse could plausibly need should be dumped here ONCE, so
that changing an analysis choice later never requires re-parsing.

Two parses are run for each language:

1. Disfluent parse  (dependency_results.csv)
   The full disfluent text (underscores stripped) is parsed with Stanza.
   Stanza is called with tokenize_pretokenized=True so its token indices align
   1:1 with the input list built from the _word_ annotation.  Each token row
   records its deprel/upos/head and whether it was marked as disfluent.
   Caveat: the parser sees disfluent input, so parse quality may be lower.

2. Fluent-only parse  (fluent_parse_results.csv)
   Only the fluent tokens are extracted and parsed as their own sentence, so
   this parse is unaffected by parser quality on disfluent text. Every fluent
   word's own deprel/upos/head is recorded, keyed by orig_token_idx (its
   position in the SAME token-index space as dependency_results.csv), so
   downstream code can align the two parses without needing Stanza again --
   e.g. to find "the nearest fluent word before position 5" or "skip
   punctuation neighbors", both of which are analysis-time decisions.

Output files (written to --output dir):
  dependency_results.csv    - one row per token: utt_id, lang, token_idx, token,
                              upos, deprel, head, head_token_idx, is_disfluent.
                              token_idx/head_token_idx index into the surface
                              token stream (head_token_idx=-1 means ROOT), so
                              downstream analysis can tell whether a word's head
                              falls inside or outside a run of disfluent tokens.
  deprel_summary.csv        - deprel+upos counts for disfluent tokens per lang
  fluent_parse_results.csv  - one row per FLUENT word (disfluent tokens are
                              simply absent, not represented at all): utt_id,
                              lang, orig_token_idx, token, upos, deprel, head,
                              head_orig_token_idx. orig_token_idx uses the same
                              index space as dependency_results.csv's token_idx.

Per-language results are checkpointed to --output/per_lang/*_{lang}.csv as each
language finishes, so a crash (or an unsupported language, e.g. Stanza has no
Swahili model) only costs that language, not the whole run. Rerunning skips
languages that already have a checkpoint; delete the per_lang file to redo one.
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


def surface_index(word):
    """0-indexed surface token position; word.parent.id is a tuple for MWT expansions."""
    parent_id = word.parent.id
    if isinstance(parent_id, tuple):
        parent_id = parent_id[0]
    return parent_id - 1


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
            surf_idx = surface_index(word)
            is_dis = flags[surf_idx] if surf_idx < len(flags) else False
            head_idx = word.head - 1
            if 0 <= head_idx < len(sent.words):
                head_word = sent.words[head_idx]
                head_text = head_word.text
                head_token_idx = surface_index(head_word)
            else:
                head_text = "ROOT"
                head_token_idx = -1
            rows.append({
                "utt_id": row["ID"],
                "lang": lang,
                "token_idx": surf_idx,
                "token": word.text,
                "upos": word.upos,
                "deprel": word.deprel,
                "head": head_text,
                "head_token_idx": head_token_idx,
                "is_disfluent": is_dis,
            })
    return rows


def parse_fluent_tokens(nlp, df, lang, col):
    """
    Parse the fluent-only token sequence (disfluent tokens stripped out
    entirely, not just masked) so every fluent word's deprel/upos/head is
    unaffected by parse-quality issues on disfluent text. One row per fluent
    word. orig_token_idx/head_orig_token_idx are expressed in the same
    surface-token index space as dependency_results.csv's token_idx, so
    downstream code can relate the two parses purely by index arithmetic.
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

        def to_orig_idx(fi):
            return orig_pos[fi] if 0 <= fi < len(orig_pos) else None

        for word in sent.words:
            fi = surface_index(word)
            head_idx = word.head - 1
            if 0 <= head_idx < len(sent.words):
                head_word = sent.words[head_idx]
                head_text = head_word.text
                head_orig_idx = to_orig_idx(surface_index(head_word))
            else:
                head_text = "ROOT"
                head_orig_idx = -1
            rows.append({
                "utt_id": row["ID"],
                "lang": lang,
                "orig_token_idx": to_orig_idx(fi),
                "token": word.text,
                "upos": word.upos,
                "deprel": word.deprel,
                "head": head_text,
                "head_orig_token_idx": head_orig_idx,
            })
    return rows


def checkpoint_paths(output_dir, lang):
    cache_dir = os.path.join(output_dir, "per_lang")
    return (
        os.path.join(cache_dir, f"dependency_results_{lang}.csv"),
        os.path.join(cache_dir, f"fluent_parse_results_{lang}.csv"),
    )


def process_language(lang, col, df, output_dir, download_kwargs, pipeline_kwargs):
    """Parse one language and checkpoint its results to per_lang/*.csv. Returns False on failure."""
    dep_path, fluent_path = checkpoint_paths(output_dir, lang)
    if os.path.exists(dep_path) and os.path.exists(fluent_path):
        print(f"[{lang}] Already processed, skipping (delete {dep_path} to reprocess).")
        return True

    print(f"\n[{lang}] Downloading model...")
    try:
        stanza.download(lang, verbose=False, **download_kwargs)
        nlp = stanza.Pipeline(lang, processors="tokenize,pos,lemma,depparse",
                               tokenize_pretokenized=True, verbose=False, **pipeline_kwargs)
    except Exception as e:
        print(f"[{lang}] Skipping: could not load Stanza model ({e})")
        return False

    print(f"[{lang}] Parsing {len(df)} utterances...")
    dep_rows = parse_utterances(nlp, df, lang, col)
    n_dis = sum(1 for r in dep_rows if r["is_disfluent"])
    print(f"[{lang}] disfluent parse: {len(dep_rows)} tokens, {n_dis} disfluent")

    fluent_rows = parse_fluent_tokens(nlp, df, lang, col)
    print(f"[{lang}] fluent-only parse: {len(fluent_rows)} tokens")

    os.makedirs(os.path.dirname(dep_path), exist_ok=True)
    pd.DataFrame(dep_rows).to_csv(dep_path, index=False)
    pd.DataFrame(fluent_rows).to_csv(fluent_path, index=False)
    print(f"[{lang}] Checkpointed -> {dep_path}, {fluent_path}")
    return True


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
    download_kwargs = {"model_dir": args.stanza_dir} if args.stanza_dir else {}
    pipeline_kwargs = {"dir": args.stanza_dir} if args.stanza_dir else {}

    lang_map = {k: v for k, v in LANG_COL_MAP.items()
                if args.langs is None or k in args.langs}

    df = pd.read_csv(args.input)

    for lang, col in lang_map.items():
        if col not in df.columns:
            print(f"Skipping {lang}: column {col} not in CSV")
            continue
        process_language(lang, col, df, args.output, download_kwargs, pipeline_kwargs)

    # Aggregate whatever per-language checkpoints exist (this run and any prior ones).
    all_dep_rows = []
    all_fluent_rows = []
    for lang in lang_map:
        dep_path, fluent_path = checkpoint_paths(args.output, lang)
        if os.path.exists(dep_path):
            all_dep_rows.append(pd.read_csv(dep_path))
        if os.path.exists(fluent_path):
            all_fluent_rows.append(pd.read_csv(fluent_path))

    results = pd.concat(all_dep_rows, ignore_index=True) if all_dep_rows else pd.DataFrame()
    out_csv = os.path.join(args.output, "dependency_results.csv")
    results.to_csv(out_csv, index=False)
    print(f"\nSaved {len(results)} rows -> {out_csv}")

    dis = results[results["is_disfluent"]] if not results.empty else results
    summary = (
        dis.groupby(["lang", "deprel", "upos"])
        .size()
        .reset_index(name="count")
        .sort_values(["lang", "count"], ascending=[True, False])
    )
    summary_csv = os.path.join(args.output, "deprel_summary.csv")
    summary.to_csv(summary_csv, index=False)
    print(f"Saved deprel summary -> {summary_csv}")

    fluent = pd.concat(all_fluent_rows, ignore_index=True) if all_fluent_rows else pd.DataFrame()
    fluent_csv = os.path.join(args.output, "fluent_parse_results.csv")
    fluent.to_csv(fluent_csv, index=False)
    print(f"Saved fluent-only parse -> {fluent_csv}")


if __name__ == "__main__":
    main()
