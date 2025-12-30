# Current Architecture: Time-Domain Ingestion

This module handles the physical ingestion of audio files. Its primary goal is to convert analog sound (or compressed MP3s) into a Vector that we can manipulate.

## 1. The Libraries

* **`librosa`**: The core signal processing engine. We use this instead of standard Python audio libraries because it is optimized for Music Information Retrieval (MIR). It automatically handles decoding and resampling.
* **`numpy`**: The mathematics background. Audio in this project is treated as a 1D Tensor (Vector) where $y[t]$ is the amplitude at time $t$.
* **`streamlit`**: The Interface layer. It allows us to separate the backend logic from the frontend visualization without writing complex JavaScript which would take up too much unnecessary time

## 2. Key Decisions in `main.py`

### `sr=22050` (Sampling Rate)
* **Code:** `y, sr = librosa.load(..., sr=22050)`
* **Why:** Humans hear up to ~20kHz. According to the **Nyquist-Shannon Sampling Theorem**, to capture a frequency of $f$, we need a sampling rate of at least $2f$.
    * $2 \times 20,000 = 40,000$. Standard CD quality is 44.1kHz.
    * We use 22.05kHz (half of CD quality) because most speech creation models (Deepfakes) are trained at this lower rate to save computational power. It preserves the quality and understanding of the speech while halving the size of our data arrays which is simply computationally efficient.

### The Waveform (Time-Domain)
* **Code:** `librosa.display.waveshow(...)`
* **Why:** This visualizes **Amplitude vs. Time**.
* **Limitation:** This view is good for seeing "loudness," but useless for detecting deepfakes. A deepfake sounds just as loud as a real voice. The "fake" artifacts are hidden in the *timbre* (frequencies), which is why we need to move to Sprint 2 (Frequency Domain).
