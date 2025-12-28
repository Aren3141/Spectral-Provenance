import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import streamlit as st

# 1. Page Setup (The UI)
st.set_page_config(page_title="Spectral Provenance", layout="wide")
st.title("Spectral-Provenance: Acoustic Analysis")
st.write("### Phase 1: Signal Visualization")

# 2. File Uploader (The Input)
# We use Streamlit's built-in uploader since its cleaner than hardcoding, makes life easier.
uploaded_file = st.file_uploader("Upload an Audio File (.mp3, .wav)", type=["mp3", "wav"])

if uploaded_file is not None: # Some sweet error handling :))
    # 3. Loading the Audio (The Physics)
    # Librosa usually needs a file path, but streamlit gives a byte stream
    # We load it with a specific sample rate given as sr
    st.audio(uploaded_file, format='audio/audio')
    
    st.write("Processing signal...")
    
    # Load audio into a numpy array (y) at 22050 Hz (standard sampling rate)
    # y = the amplitude of the wave at each point in time
    # sr = sampling rate
    y, sr = librosa.load(uploaded_file, sr=22050)
    
    # 4. Visualizing the Waveform (The Graph)
    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr, ax=ax, color="blue")
    ax.set_title("Time-Domain Waveform (Amplitude vs Time)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    
    # Display in Streamlit
    st.pyplot(fig)
    
    # 5. Show the shape of the numpy array, showing that its just a vector
    st.info(f"Maths Stats: This audio is a vector of {y.shape[0]} floating point numbers.")