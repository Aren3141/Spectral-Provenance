import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os

# Import the brain
from src.model import SpectrogramCNN

# --- CONFIG ---
# We now point to V2 Segmentation Model
MODEL_PATH = "models/spectral_cnn_v2.pth"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Physics Constants (Must match processor.py)
SAMPLE_RATE = 44100
N_FFT = 2048
HOP_LENGTH = 512
SEGMENT_DURATION = 4 # seconds
SAMPLES_PER_SEGMENT = SEGMENT_DURATION * SAMPLE_RATE

# Page Setup
st.set_page_config(page_title="Spectral-Provenance", layout="wide")

st.title("Spectral-Provenance: Deepfake Audio Detector")
st.markdown("""
**System Status:** Online | **Engine:** SpectrogramCNN (v2) | **Logic:** Sliding Window Segmentation
""")

# Load the Model
@st.cache_resource
def load_model():
    model = SpectrogramCNN()
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval() 
        return model
    else:
        st.error(f"Model not found at {MODEL_PATH}. Did you run train.py?")
        return None

model = load_model()

# PREPROCESSING PIPELINE (UPDATED FOR SEGMENTATION)
# FIX: Cut-offs no longer considered when below 50% of the audio file.

def process_audio_segments(audio_file):
    
    # Takes a raw audio file, chops it into 4s segments.
    # FILTERS: Discards segments with <50% valid audio (unless it's the only segment).
    
    # Load audio
    y, sr = librosa.load(audio_file, sr=SAMPLE_RATE)
    
    total_samples = len(y)
    num_segments = int(np.ceil(total_samples / SAMPLES_PER_SEGMENT))
    
    segment_images = []
    
    for i in range(num_segments):
        start = i * SAMPLES_PER_SEGMENT
        end = start + SAMPLES_PER_SEGMENT
        chunk = y[start:end]
        
        # THE FIX: VALIDITY CHECK 
        # Calculate how much real audio is in this chunk
        valid_ratio = len(chunk) / SAMPLES_PER_SEGMENT
        
        # If the chunk is less than 50% real audio, AND it's not the only chunk...
        # We assume it's just a trailing tail and discard it to avoid "Silence Artifacts"
        if valid_ratio < 0.5 and num_segments > 1:
            continue

        # Padding if too short (Standard consistency)
        if len(chunk) < SAMPLES_PER_SEGMENT:
            padding = SAMPLES_PER_SEGMENT - len(chunk)
            chunk = np.pad(chunk, (0, padding), mode='constant')
            
        # Spectrogram Generation
        S = librosa.stft(chunk, n_fft=N_FFT, hop_length=HOP_LENGTH)
        S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
        
        # Plotting
        plt.figure(figsize=(4, 4))
        librosa.display.specshow(S_db, sr=sr, hop_length=HOP_LENGTH, cmap='gray')
        plt.axis('off')
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        buf.seek(0)
        
        img = Image.open(buf).convert('L')
        timestamp = f"{i*4}s - {(i+1)*4}s"
        segment_images.append((img, timestamp))
        
    return segment_images

import io # missed this import earlier

# UI LAYOUT
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Channel")
    uploaded_file = st.file_uploader("Upload Audio", type=["flac", "wav", "mp3"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        
        if st.button("Run Forensic Analysis"):
            if model is None:
                st.error("Model is offline.")
            else:
                with st.spinner("Segmenting & Analyzing..."):
                    # Segment the file
                    segments = process_audio_segments(uploaded_file)
                    
                    # Run Inference on each chunk
                    results = []
                    transform = transforms.Compose([
                        transforms.Resize((128, 128)),
                        transforms.Grayscale(num_output_channels=1),
                        transforms.ToTensor()
                    ])
                    
                    st.write(f"Analyzed {len(segments)} segments.")
                    
                    bonafide_score = 0
                    spoof_score = 0
                    
                    # Display results per segment
                    for img, time_label in segments:
                        # Prepare input
                        img_tensor = transform(img).unsqueeze(0).to(DEVICE)
                        
                        # Predict
                        with torch.no_grad():
                            output = model(img_tensor)
                            probs = F.softmax(output, dim=1)
                            # class 0 = bonafide, class 1 = spoof
                            
                        p_real = probs[0][0].item()
                        p_fake = probs[0][1].item()
                        
                        # Accumulate average
                        bonafide_score += p_real
                        spoof_score += p_fake
                        
                        # Visual for this segment
                        with st.expander(f"Segment {time_label}: {'Spoof' if p_fake > 0.5 else 'Bonafide'}"):
                            st.image(img, width=150)
                            st.write(f"Spoof Confidence: {p_fake*100:.2f}%")

                    # 3. Final Verdict (Average Pooling)
                    avg_spoof = spoof_score / len(segments)
                    
                    st.markdown("---")
                    if avg_spoof > 0.5:
                        st.error(f"### FINAL VERDICT: SPOOF ({avg_spoof*100:.2f}%)")
                    else:
                        st.success(f"### FINAL VERDICT: BONAFIDE ({100 - avg_spoof*100:.2f}%)")

with col2:
    st.info("System Architecture: Sliding Window")
    st.markdown("""
    **Why multiple segments?**
    The model analyzes audio in 4-second blocks to maintain physical consistency. 
    If a 10-second file is uploaded, it is split into 3 blocks (4s, 4s, 2s+pad).
    
    **Decision Logic:**
    The system averages the probability scores across all segments to reach a final verdict.
    """)