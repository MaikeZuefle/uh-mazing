import os
import pandas as pd
import torch
import torchaudio

# Paths
csv_file = "translation-dataset-with-timestamps.csv"
input_audio_folder = "translation_data/translation_dataset_audio"
output_audio_folder = "translation_data/translation_dataset_audio_preprocessed"

# Make output folder if it doesn't exist
os.makedirs(output_audio_folder, exist_ok=True)

# Load CSV
df = pd.read_csv(csv_file)

# Iterate over rows
for idx, row in df.iterrows():
    file_id = row['file'].split("sw")[-1]  # e.g., sw2005
    file_id = f"sw0{file_id}"
    speaker = row['speaker']  # A or B
    turn = row['turn']
    start_time = float(row['start_time'])  # seconds
    end_time = float(row['end_time'])      # seconds

    filename = f"{file_id}_{speaker}_{turn}.wav"
    audio_file_path = os.path.join(input_audio_folder, filename)

    if not os.path.exists(audio_file_path):
        print(f"{audio_file_path} does not exist!")
        continue


    waveform, sample_rate = torchaudio.load(audio_file_path)  # waveform: [channels, samples]

    # Select channel (0 = left / A, 1 = right / B)
    channel_idx = 0 if speaker == 'A' else 1
    channel_audio = waveform[channel_idx]  # [samples]

    # Save snippet
    output_filename = f"{file_id}_{speaker}_{turn}.wav"
    torchaudio.save(os.path.join(output_audio_folder, output_filename), channel_audio.unsqueeze(0), sample_rate)
    print(f"Saved {output_filename}")

print("All audio snippets processed.")
