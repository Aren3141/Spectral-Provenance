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

# CONFIG 
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

# Segmentation Settings NEW
SEGMENT_DURATION = 4 # seconds
SAMPLES_PER_SEGMENT = SEGMENT_DURATION * SAMPLE_RATE

def setup_directories():
    # Creating folder structure for the output images
    for category in ["bonafide", "spoof"]:
        path = os.path.join(OUTPUT_DIR, category)
        os.makedirs(path, exist_ok=True)
        print(f"[INFO] Directory ready: {path}")

def generate_spectrogram_segments(file_path, save_dir, filename_base):
    """
    Loads audio, slices it into fixed 4-second chunks, and saves spectrograms.
    """
    try:
        # Load Audio
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        
        # Calculate Segments
        total_samples = len(y)
        
        # Handle empty files
        if total_samples == 0:
            return

        num_segments = int(np.ceil(total_samples / SAMPLES_PER_SEGMENT))
        
        for i in range(num_segments):
            start = i * SAMPLES_PER_SEGMENT
            end = start + SAMPLES_PER_SEGMENT
            
            # Extract the chunk
            chunk = y[start:end]
            
            # Padding FIX
            # If chunk is shorter than 4s, add silence (zeros) to the end
            if len(chunk) < SAMPLES_PER_SEGMENT:
                padding = SAMPLES_PER_SEGMENT - len(chunk)
                chunk = np.pad(chunk, (0, padding), mode='constant')
            
            # Generate Spectrogram
            S = librosa.stft(chunk, n_fft=N_FFT, hop_length=HOP_LENGTH)
            S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
            
            # Save Image
            save_path = os.path.join(save_dir, f"{filename_base}_seg{i}.png")
            
            # Create the plot (4x4 inches to match main.py logic)
            plt.figure(figsize=(4, 4))
            librosa.display.specshow(S_db, sr=sr, hop_length=HOP_LENGTH, cmap='gray')
            plt.axis('off')
            
            # Ensure we don't save white borders
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close() # Close memory for safety

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def main():
    print("SPECTRAL-PROVENANCE: DATA INGESTION PROTOCOL (SEGMENTATION ENABLED)")
    setup_directories()

    # Load the Protocol 
    print(f"[INFO] Loading Protocol from: {PROTOCOL_PATH}")
    if not os.path.exists(PROTOCOL_PATH):
        print(f"[ERROR] Protocol file not found :( Check your paths.")
        return

    df = pd.read_csv(PROTOCOL_PATH, sep=" ", header=None, names=["speaker", "filename", "system", "null", "label"])
    
    # Test on just 50 files first, COMMENT out later
    # df = df.head(50) 
    
    print(f"[INFO] Found {len(df)} files to process.")

    # The loop
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Generating Segments"):
        filename = row['filename']
        label = row['label'] # 'bonafide' or 'spoof'
        
        target_dir = os.path.join(OUTPUT_DIR, label)
        
        # RESUME LOGIC
        # We check if the FIRST segment already exists.
        # If "LA_T_xxxxxx_seg0.png" is there, we assume we finished this file.
        expected_first_seg = os.path.join(target_dir, f"{filename}_seg0.png")
        
        if os.path.exists(expected_first_seg):
            # If it exists, we skip processing this file.
            # tqdm will update, bar will fly past files already processed.
            continue
        

        source_file = os.path.join(AUDIO_DIR, f"{filename}.flac")
        
        # Check if source exists
        if os.path.exists(source_file):
            generate_spectrogram_segments(source_file, target_dir, filename)

if __name__ == "__main__":
    main()