# How to generate swb_segments_train_pairs.csv

1. Follow the instructions from the dres repo (https://github.com/mariateleki/dres):
    ```
    # Obtain Switchboard (Treebank-3, LDC99T42) and place it like so:
    data/treebank_3/

    # Clone repo
    git clone https://github.com/mariateleki/dres.git
    cd disfluency_removal

    # Install the disfluency_removal package locally
    pip install -e .

    # Run preprocessing
    python -m disfluency_removal.data_1_generate_flat
    python -m disfluency_removal.data_2_generate_full_segments
    ```
2. Run an additional processing step:
    ```
    python -m disfluency_removal.data_write_csvs.py
    ```