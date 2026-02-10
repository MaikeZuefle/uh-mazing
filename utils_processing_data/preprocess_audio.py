import os
import pandas as pd
import torch
import torchaudio
import math


def time_str_to_seconds(time_str):
    """Convert 'MM:SS' or 'HH:MM:SS' string to total seconds (float)."""
    parts = [float(p) for p in time_str.split(":")]
    if len(parts) == 2:  # MM:SS
        minutes, seconds = parts
        return minutes * 60 + seconds
    elif len(parts) == 3:  # HH:MM:SS
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds
    else:
        raise ValueError(f"Invalid time format: {time_str}")

def is_missing(value):
    """
    Returns True if the value is None, NaN, or an empty string.
    Works for floats, ints, strings, and pandas NaNs.
    """
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str):
        value = value.strip().lower()
        if value in ["", "nan"]:
            return True
    return False

def preprocess_audio(csv_file, input_audio_folder, output_audio_folder):
    # Make output folder if it doesn't exist
    os.makedirs(output_audio_folder, exist_ok=True)

    # Load CSV
    df = pd.read_csv(csv_file)


    # Iterate over rows
    for idx, row in df.iterrows():
        start_time = time_str_to_seconds(row['start_time'])
        end_time = time_str_to_seconds(row['end_time'])

        alternative_file = row["alternative_file"]
        alternative_speaker = row["alternative_speaker"]

        file_id = row['file'].split("sw")[-1]  # e.g., sw2005
        file_id = f"sw0{file_id}"
        turn = row['turn']
        original_speaker = row['speaker'] 
        if is_missing(alternative_file):
            speaker = row['speaker']  # A or B
            filename = f"{file_id}_{speaker}_{turn}.wav"

        else:
            filename = f"{alternative_file}.wav"
            speaker = alternative_speaker # overwrite speaker

        audio_file_path = os.path.join(input_audio_folder, filename)

        if not os.path.exists(audio_file_path):
            print(f"{audio_file_path} does not exist!")
            continue


        waveform, sample_rate = torchaudio.load(audio_file_path)  # waveform: [channels, samples]

        # Select channel (0 = left / A, 1 = right / B)
        channel_idx = 0 if speaker == 'A' else 1
        channel_audio = waveform[channel_idx]  # [samples]


        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)

        # Slice the audio
        snippet_audio = channel_audio[start_sample:end_sample]  # [samples]



        # Save snippet
        output_filename = f"{file_id}_{original_speaker}_{turn}.wav"
        torchaudio.save(os.path.join(output_audio_folder, output_filename), snippet_audio.unsqueeze(0), sample_rate)
        print(f"Saved {output_filename}")

    print("All audio snippets processed.")

if __name__ == "__main__":
    csv_file = "translation_data/translation-dataset-with-timestamps.csv"
    input_audio_folder = "translation_data/translation_dataset_audio"
    output_audio_folder = "translation_data/translation_dataset_audio_preprocessed"
    preprocess_audio(csv_file, input_audio_folder, output_audio_folder)