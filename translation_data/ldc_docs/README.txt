Uh-Mazing: Disfluency-Aware Speech Translation

1. Data Selection: Switchboard

We use the Switchboard corpus (Godfrey et al., 1993), which consists of
spontaneous English conversational telephone speech with detailed disfluency
annotations, making it well-suited for studying genuine disfluent speech
(Kundu et al., 2022). We align the transcripts with the corresponding audio
recordings and retain the original disfluency annotations. Following Shriberg
(1994), we focus on three disfluency classes: interjections (INTJ),
parentheticals (PRN), and edited speech (EDITED).

We select 80 utterances from the Switchboard corpus, manually chosen to contain
multiple disfluencies and to span a range of lengths (17-212 words, median 51,
totalling 25 minutes of audio).


2. Human Translation

Uh-Mazing consists of 640 human translations of 80 carefully curated
Switchboard utterances (Godfrey et al., 1993), which were selected for high
disfluency density and a wide range of lengths. Each utterance is translated
from English into Mandarin Chinese (ZH), Spanish (ES), Hindi (HI), French (FR),
German (DE), Italian (IT), Czech (CS), and Modern Standard Arabic (AR). Target
languages were chosen to span diverse language families and regions, enabling
analysis of how disfluency and translation behaviour vary across languages.

For each pair, the source side consists of the speech recording, its
time-aligned disfluent transcript, and a fluent transcript obtained by
stripping the marked disfluencies (edited, interjections, parentheticals); the
target side consists of a disfluent and a fluent translation, produced by the
annotation process described below.


3. Annotation Protocol

Translations were collected via Prolific in two rounds with independently
recruited annotators, following Taguchi et al. (2025). In Round 1, annotators
translated each disfluent English source into the target language in batches
of 20 utterances, marking disfluent tokens with underscores (e.g. _uh_). In
Round 2, a second annotator per language reviewed each Round 1 translation.
This standardises the disfluency markup and provides a per-translation
confidence signal, since agreement between the two rounds is stronger evidence
of correctness than either round alone. We use Round 2 outputs as the final
dataset.


4. File Structure

uh_mazing/
  data/
    speech/                              80 audio files, one per utterance
    transcripts/en_transcripts.csv       English disfluent and fluent transcripts
    annotations/disfluency_types.csv     disfluency type counts and type variants
    text/translations.csv                disfluent and fluent translations into the
                                         8 target languages
  docs/
    README.txt                           this file

Every CSV file contains one row per utterance, in the same order in all files.
The ID column identifies the utterance, so rows, audio files and the different
CSV files can be matched via the ID.


Utterance IDs

IDs have the form sw<conversation>_<channel>_<turn>, e.g. sw2005_A_19:
  sw2005   Switchboard conversation 2005
  A        speaker channel (A or B) of the conversation
  19       turn number within the conversation
The audio file for an utterance is data/speech/<ID>.flac.


Disfluency markup

In the disfluent transcripts and translations, every disfluent token is
enclosed in underscores, including punctuation that belongs to the disfluency,
e.g. "_you_ _know_ _,_". Tokens without underscores are fluent.


data/speech/

FLAC-compressed audio, 16 kHz, 16-bit, mono.


data/transcripts/en_transcripts.csv

  ID             utterance ID
  EN_fluent      English transcript with all disfluencies removed
  EN_disfluent   English transcript with disfluencies, marked with underscores


data/annotations/disfluency_types.csv

Disfluency types come from the parse trees of the Switchboard Treebank, which
labels disfluencies with three node types:
  EDITED   speech the speaker restarts or corrects ("for, for quite some time")
  INTJ     interjections and filled pauses ("uh", "um", "well")
  PRN      parentheticals ("you know", "I mean")

  ID            utterance ID
  <type>_count  number of disfluencies of that type in the utterance
                (edited_count, intj_count, prn_count)
  type variants utterance keeping only the disfluencies of the named types
                and removing all others (edited_nodes_only, intj_only,
                prn_only, edited_intj, intj_prn, edited_prn, edited_prn_intj)


data/text/translations.csv

  ID                 utterance ID
  <LANG>_disfluent   disfluent translation, marked with underscores
  <LANG>_fluent      fluent translation

<LANG> is one of AR (Modern Standard Arabic), CS (Czech), DE (German),
ES (Spanish), FR (French), HI (Hindi), IT (Italian) and ZH (Mandarin Chinese).

The disfluent translations are tokenized: all tokens, including punctuation,
are separated by spaces, and Chinese is separated by character. The fluent
translations are derived from the disfluent ones by removing all marked tokens
and are detokenized.


5. Citation

If you use this dataset, please cite:


@misc{züfle2026roledisfluenciesspeechtranslation,
      title={The Role of Disfluencies in Speech Translation},
      author={Maike Züfle and Maria Teleki and Fabian Retkowski and Vilém Zouhar and Oliver Grabner and Alexander Waibel and James Caverlee and Jan Niehues},
      year={2026},
      eprint={2608.02138},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2608.02138},
}
