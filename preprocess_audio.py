import os
import pandas as pd
from pydub import AudioSegment

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
    start_time = float(row['start_time']) * 1000  # pydub works in milliseconds
    end_time = float(row['end_time']) * 1000

    filename = f"{file_id}_{speaker}_{turn}.wav"
    audio_file_path = os.path.join(input_audio_folder, filename)

    if not os.path.exists(audio_file_path):
        print(f"{audio_file_path} does not exist!")
        continue


    # Load audio
    audio = AudioSegment.from_wav(audio_file_path)

    # Select channel
    if speaker.upper() == 'A':
        channel_audio = audio.split_to_mono()[0]  # Left channel
    else:
        channel_audio = audio.split_to_mono()[1]  # Right channel

    # Cut the audio using timestamps
    try:
        snippet = channel_audio[start_time:end_time]
    except:
        breakpoint()

    # Save snippet
    output_filename = f"{file_id}_{speaker}_{row['turn']}.wav"
    snippet.export(os.path.join(output_audio_folder, output_filename), format="wav")

    print(f"Saved {output_filename}")

print("All audio snippets processed.")
