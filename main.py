import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(page_title="Spectral Provenance", layout="wide")

# The Header
st.title("Spectral-Provenance: Acoustic Analysis Lab")
st.markdown("""
**Status:** Phase 1.5 (Frequency Domain Analysis)
This tool analyzes audio physics. We visualize both the **Time Domain** (Amplitude) and **Frequency Domain** (Spectrogram).
""")

# The Sidebar
st.sidebar.header("Data Input")
uploaded_file = st.sidebar.file_uploader("Upload an Audio Sample (.mp3, .wav)", type=["mp3", "wav"])

# The Main Logic
if uploaded_file is not None:
    # 1. Display Audio Player
    st.sidebar.audio(uploaded_file, format='audio/audio')
    
    # 2. Loading the Audio 
    with st.spinner('Calculating Waveform Physics...'):
        # CHANGED: Increased sr to 44100 to see frequencies up to 22kHz
        # We need this high resolution to spot the "16kHz cut-off" artifacts that usually occur with Deepfakes 
        y, sr = librosa.load(uploaded_file, sr=44100)

    # 3. Stats Panel
    st.subheader("Signal Statistics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Sample Rate (Hz)", f"{sr}")
    col2.metric("Duration (s)", f"{round(librosa.get_duration(y=y, sr=sr), 2)}")
    col3.metric("Vector Dimensions", f"{y.shape[0]}")

    # 4. Visualization 1: Time Domain
    st.divider()
    st.subheader("1. Time-Domain Analysis (Waveform)")
    st.markdown("Shows **Amplitude** (Loudness) over **Time**.")
    
    fig1, ax1 = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax1, color="#1f77b4")
    ax1.set_title("Raw Audio Waveform")
    st.pyplot(fig1)

    # 5. Visualization 2: Frequency Domain (The New Code)
    st.divider()
    st.subheader("2. Frequency-Domain Analysis (Mel-Spectrogram)")
    st.markdown("""
    To detect 'robotic' artifacts, we use a **Short-Time Fourier Transform (STFT)**.
    This heatmap shows **Frequency intensity** over time.
    """)

    with st.spinner("Computing STFT and Mel-Scale conversion..."):
        # The Mathsy Bit:
        # 1. STFT: Converts signal to complex numbers like magnitude and phase
        # 2. Abs: We only need the Magnitude so the loudness of the pitch
        # 3. DB: Convert to Decibels (logarithmic scale for human hearing ).
        
        # Calculate Short-Time Fourier Transform
        D = librosa.stft(y)  
        
        # Convert to Mel-Scale (approximates human ear sensitivity)
        # We map the linear frequencies to Mel bands
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

        fig2, ax2 = plt.subplots(figsize=(10, 4))
        # y_axis='log' creates a logarithmic scale for frequency
        img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', ax=ax2)
        fig2.colorbar(img, ax=ax2, format="%+2.0f dB")
        ax2.set_title("Mel-Spectrogram (Frequency Heatmap)")
        
        st.pyplot(fig2)
        
        st.info("""
        **Forensic Note:** Deepfake models often struggle with high frequencies. 
        Look for unnatural 'black voids' or hard cut-offs above 8kHz-10kHz.
        """)
        
else:
    st.info("Waiting for input... Upload a file in the sidebar to begin analysis.")