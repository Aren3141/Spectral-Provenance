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
import io
from scipy.signal import butter, lfilter

# Import the brain
from src.model import SpectrogramCNN

# --- CONFIG ---
# We point to V3 (Noise Augmented)
MODEL_PATH = "models/spectral_cnn_v3.pth"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Physics Constants
SAMPLE_RATE = 44100
N_FFT = 2048
HOP_LENGTH = 512
SEGMENT_DURATION = 4 # seconds
SAMPLES_PER_SEGMENT = SEGMENT_DURATION * SAMPLE_RATE

# --- NEW: PHYSICS ENGINE (Low Pass Filter) ---
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# --- UTILITY CLASS: NOISE INJECTION ---
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

# Page Setup
st.set_page_config(page_title="Spectral-Provenance", layout="wide")

st.title("Spectral-Provenance: Deepfake Audio Detector")
st.markdown("""
**System Status:** Online | **Engine:** SpectrogramCNN (v3) | **Logic:** Low-Pass Filter (8kHz) + Dithering
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

def process_audio_segments(audio_file):
    # Load audio
    y, sr = librosa.load(audio_file, sr=SAMPLE_RATE)
    
    # --- CRITICAL FIX: THE HD CURSE ---
    # The training data (ASVspoof) is 16kHz (Nyquist 8kHz).
    # Your mic is 44.1kHz. The model thinks anything >8kHz is a deepfake artifact.
    # We apply a hard Low-Pass Filter at 8kHz to match the training distribution.
    y = lowpass_filter(y, cutoff=8000, fs=SAMPLE_RATE, order=6)
    # ----------------------------------
    
    total_samples = len(y)
    num_segments = int(np.ceil(total_samples / SAMPLES_PER_SEGMENT))
    
    segment_images = []
    
    for i in range(num_segments):
        start = i * SAMPLES_PER_SEGMENT
        end = start + SAMPLES_PER_SEGMENT
        chunk = y[start:end]
        
        valid_ratio = len(chunk) / SAMPLES_PER_SEGMENT
        
        if valid_ratio < 0.5 and num_segments > 1:
            continue

        if len(chunk) < SAMPLES_PER_SEGMENT:
            padding = SAMPLES_PER_SEGMENT - len(chunk)
            chunk = np.pad(chunk, (0, padding), mode='constant')
            
        S = librosa.stft(chunk, n_fft=N_FFT, hop_length=HOP_LENGTH)
        S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
        
        plt.figure(figsize=(4, 4))
        librosa.display.specshow(S_db, sr=sr, hop_length=HOP_LENGTH, cmap='gray')
        plt.axis('off')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        buf.seek(0)
        
        img = Image.open(buf).convert('L')
        timestamp = f"{i*4}s - {(i+1)*4}s"
        segment_images.append((img, timestamp))
        
    return segment_images

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
                with st.spinner("Applying Low-Pass Filter & Analyzing..."):
                    segments = process_audio_segments(uploaded_file)
                    results = []
                    
                    # We keep the noise injection to fix the black void issue
                    transform = transforms.Compose([
                        transforms.Resize((128, 128)),
                        transforms.Grayscale(num_output_channels=1),
                        transforms.ToTensor(),
                        AddGaussianNoise(0., 0.05) 
                    ])
                    
                    st.write(f"Analyzed {len(segments)} segments.")
                    
                    bonafide_score = 0
                    spoof_score = 0
                    
                    for img, time_label in segments:
                        img_tensor = transform(img).unsqueeze(0).to(DEVICE)
                        
                        with torch.no_grad():
                            output = model(img_tensor)
                            probs = F.softmax(output, dim=1)
                            
                        p_real = probs[0][0].item()
                        p_fake = probs[0][1].item()
                        
                        bonafide_score += p_real
                        spoof_score += p_fake
                        
                        status_color = ":red" if p_fake > 0.5 else ":green"
                        with st.expander(f"Segment {time_label} {status_color}[{'Spoof' if p_fake > 0.5 else 'Bonafide'}]"):
                            st.image(img, width=150)
                            st.write(f"Spoof Confidence: {p_fake*100:.2f}%")

                    avg_spoof = spoof_score / len(segments)
                    
                    st.markdown("---")
                    if avg_spoof > 0.5:
                        st.error(f"### FINAL VERDICT: SPOOF ({avg_spoof*100:.2f}%)")
                    else:
                        st.success(f"### FINAL VERDICT: BONAFIDE ({100 - avg_spoof*100:.2f}%)")

with col2:
    st.info("System Architecture: Domain Adaptation")
    st.markdown("""
    **Physics Engine Update:**
    Your audio is being processed with an **8kHz Low-Pass Filter**.
    
    **Why?**
    The AI was trained on telephony-grade audio (ASVspoof). 
    High-fidelity microphones capture frequencies (>8kHz) that the AI 
    misinterprets as generative artifacts. By filtering the input, 
    we match the 'spectral shape' of the training data.
    """)