How to set up the human campaign:

0. Extract `generated_output_non_commerical` and `preprocessed_audio_switchboard` to `humeval/data/` together with `audio_en_en_standard_gold_output.xml`.
1. Install pearmut: `pip install pearmut`
2. Create the campaign files using `scripts/01-prepare_campaign.py`, potentially adding the pass/fail tokens from Prolific
3. Add campaign files `pearmut add humeval/data/campaigns/*.json`
4. Start campaign locally `pearmut run`

If the campaign needs to be hosted on the internet with ngrok, use:

5. `pearmut run --port 8001 --server https://pearmut.ngrok.io` and `ngrok http --url=pearmut.ngrok.io 8001` in separate terminals