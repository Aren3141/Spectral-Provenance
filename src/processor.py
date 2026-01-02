import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import warnings

# Suppress warnings to keep terminal clean
warnings.filterwarnings("ignore")

# --- CONFIG ---
# Define the path to the file that tells us which audio is fake vs real
PROTOCOL_PATH = "data/raw/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"

# Define Source Audio
AUDIO_DIR = "data/raw/LA/ASVspoof2019_LA_train/flac"

# Define where we save the images
OUTPUT_DIR = "data/processed"

# Physics Settings (MUST match main.py)
SAMPLE_RATE = 44100
N_FFT = 2048
HOP_LENGTH = 512

def setup_directories():
    # Creating folder structure for the output images
    
    for category in ["bonafide", "spoof"]:
        path = os.path.join(OUTPUT_DIR, category)
        os.makedirs(path, exist_ok=True)
        print(f"[INFO] Directory ready: {path}")

def generate_spectrogram(file_path, save_path):
    
    # The Process
    # 1. Loads Audio
    # 2. Performs STFT (Fourier Transform)
    # 3. Converts to Mel-Scale (Logarithmic)
    # 4. Saves as an Image (No axes, just pure data)
    
    try: #Error Handling
        # 1. Load Audio (High Res)
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        
        # 2. STFT to Mel Spectrogram Conversion
        # We assume a fixed length to make sure all images are the same size for the AI
        # (For now, we let them vary, but typically CNNs need fixed sizes. We will handle resizing later on!)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        S_db = librosa.amplitude_to_db(S, ref=np.max)

        # 3. Plotting (stripping axes to save only the data)
        plt.figure(figsize=(4, 4))
        librosa.display.specshow(S_db, sr=sr, hop_length=HOP_LENGTH)
        plt.axis('off') # Remove axes
        plt.tight_layout(pad=0) # Remove padding
        
        # 4. Save
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close() # Close memory to prevent any crashes
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}") # Just in case

def main():
    print("SPECTRAL-PROVENANCE: DATA INGESTION PROTOCOL")
    setup_directories()

    # Load the Protocol 
    # format: speaker_id, filename, system_id, null, key (bonafide/spoof)
    print(f"[INFO] Loading Protocol from: {PROTOCOL_PATH}")
    if not os.path.exists(PROTOCOL_PATH):
        print(f"[ERROR] Protocol file not found :( Check your paths.")
        return

    df = pd.read_csv(PROTOCOL_PATH, sep=" ", header=None, names=["speaker", "filename", "system", "null", "label"])
    
    # This is a little subset you can try processing first, comment out to process it all later. 
    # df = df.head(50) 
    
    print(f"[INFO] Found {len(df)} files to process.")

    # The Loop
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Generating Spectrograms"):
        filename = row['filename']
        label = row['label'] # 'bonafide' or 'spoof'
        
        source_file = os.path.join(AUDIO_DIR, f"{filename}.flac")
        target_file = os.path.join(OUTPUT_DIR, label, f"{filename}.png")
        
        # Skip if it already exists 
        if os.path.exists(target_file):
            continue
            
        if os.path.exists(source_file):
            generate_spectrogram(source_file, target_file)
        else:
            # In the case that its in a different split
            pass
        

    print("[SUCCESS] Data Transmutation Complete.")

if __name__ == "__main__":
    main()