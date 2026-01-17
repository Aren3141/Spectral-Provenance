import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import io
import os

# Import the brain
from src.model import SpectrogramCNN

# CONFIG
MODEL_PATH = "models/spectral_cnn_v1.pth"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# PAGE SETUP
st.set_page_config(page_title="Spectral-Provenance", page_icon="🕵️", layout="wide")

st.title("Spectral-Provenance: Deepfake Audio Detector")
st.markdown("""
**System Status:** Online | **Engine:** SpectrogramCNN (v1) | **Device:** Apple Silicon (MPS)
""")

# LOAD THE BRAIN
@st.cache_resource
def load_model():
    model = SpectrogramCNN()
    # Load the weights trained
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval() # Set to eval mode (locks Dropout)
        return model
    else:
        st.error(f"Model not found at {MODEL_PATH}. Did you run train.py?")
        return None

model = load_model()

# PREPROCESSING PIPELINE
def generate_spectrogram_image(audio_buffer):
    
    # Converts audio bytes to Mel Spectrogram to PIL Image
    # This mimics exactly how we generated the Training Data
    
    # Load Audio
    y, sr = librosa.load(audio_buffer, sr=44100) # Force 44.1kHz
    
    # STFT & Mel Scale
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    # Render to Memory (No Axis, No Labels)
    fig = plt.figure(figsize=(10, 10))
    plt.axis('off')
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel')
    plt.tight_layout(pad=0)
    
    # Save to Buffer (RAM) instead of Disk
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    
    return Image.open(buf)

def predict(image, model):
    """
    Passes the image through the Neural Network.
    """
    # Same transforms as train.py
    transform_pipeline = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])
    
    # Prepare Image
    img_tensor = transform_pipeline(image).unsqueeze(0) # Add Batch Dimension (1, 1, 128, 128)
    img_tensor = img_tensor.to(DEVICE)
    
    # Inference
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1) # Convert logits to %
        
    # Get Results
    confidence, predicted_class = torch.max(probabilities, 1)
    return predicted_class.item(), confidence.item(), probabilities[0].tolist()

# USER INTERFACE 
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Input Audio")
    uploaded_file = st.file_uploader("Upload FLAC/WAV", type=["flac", "wav", "mp3"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        
        if st.button("Analyze Spectrogram"):
            with st.spinner("Generating Spectral Image..."):
                # Convert Audio to Image
                spec_image = generate_spectrogram_image(uploaded_file)
                
                # Run Inference
                class_idx, conf, probs = predict(spec_image, model)
                
                # Display Results
                classes = ['Bonafide (Real)', 'Spoof (Fake)']
                result_color = "green" if class_idx == 0 else "red"
                
                st.markdown(f"### Prediction: :{result_color}[{classes[class_idx]}]")
                st.metric("Confidence", f"{conf*100:.2f}%")
                
                # Debug Info
                with st.expander("See Tensor Details"):
                    st.write(f"Real Probability: {probs[0]:.4f}")
                    st.write(f"Fake Probability: {probs[1]:.4f}")

with col2:
    if uploaded_file is not None and 'spec_image' in locals():
        st.subheader("Spectral Analysis")
        st.image(spec_image, caption="Generated Mel-Spectrogram (Input to CNN)", use_container_width=True)
        st.markdown("""
        **What is the AI seeing?**
        * The CNN scans this image for "hard cut-offs" in the high frequencies.
        * Real human voices fill the spectrum organically.
        * Synthetic voices often leave 'black voids' or checkerboard artifacts.
        """)