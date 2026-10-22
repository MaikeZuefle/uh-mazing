# The Role of Disfluencies in Speech Translation

[![arXiv](https://img.shields.io/badge/arXiv-2608.02138-b31b1b.svg)](https://arxiv.org/abs/2608.02138)

Current speech translation systems, including SpeechLLMs, are trained on cleaned text and tend to strip disfluencies like filled pauses and false starts rather than translate them. We show this comes at a cost: disfluencies carry meaning that gets lost when speech is cleaned up. To study this systematically, we introduce **UH-MAZING**, a benchmark of human-translated, disfluency-annotated Switchboard speech covering English into eight target languages. Across these languages and several architectures, we find that false starts and self-repairs, not filled pauses or discourse markers, drive most of the translation-quality loss, and that models which fail to preserve a disfluency tend to omit it rather than mistranslate it. We show inference-time decoding can mitigate this without retraining.

> **Note:** The UH-MAZING dataset has been submitted to the Linguistic Data Consortium (LDC). We will link it here as soon as it is publicly available. Until then, please contact us if you need the data.

This repository contains the code for building the benchmark, running and evaluating the translation systems, and the analyses in the paper.


## Repository structure

```
uh-mazing/
├── translation_data/               benchmark data and data preparation
├── benchmarking/                   translation experiments and evaluation
├── analysis/                       emotion and syntax analyses
├── humeval_disfluency_meaning/     human study on disfluencies and emotion
└── humeval_translation_quality/    human evaluation of translations
```


## Running the translation experiments

Example scripts that run the translation systems of the paper are in `benchmarking/scripts/scripts_inference/`, and the evaluation scripts are in `benchmarking/scripts/scripts_eval/`.


## Analyses

**Emotion analysis** (`analysis/emotion_analysis/`): analyses the human emotion annotations in the Audio, Disfluent, and Fluent conditions, and compares them with an LLM annotator over seven seeds.

**Syntax analysis** (`analysis/syntax_analysis/`): parses the reference translations, marks disfluencies in system outputs with GPT-5.2, and compares the grammatical context of disfluencies in system outputs and human translations.

## Human evaluation

`humeval_disfluency_meaning/` and `humeval_translation_quality/` contain the campaign setup and the collected annotations of the two human studies, run on Pearmut.

## Citation

```bibtex
@misc{züfle2026roledisfluenciesspeechtranslation,
      title={The Role of Disfluencies in Speech Translation},
      author={Maike Züfle and Maria Teleki and Fabian Retkowski and Vilém Zouhar and Oliver Grabner and Alexander Waibel and James Caverlee and Jan Niehues},
      year={2026},
      eprint={2608.02138},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2608.02138},
}
```
