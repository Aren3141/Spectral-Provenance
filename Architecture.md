# Current Architecture: Audio Ingestion & Spectral Analysis

This module handles the physical ingestion of audio files and mapping the frequencies within them. Its primary goal is to convert raw signal data into visual inputs (Spectrograms) suitable for forensic analysis.

## 1. The Libraries

* **`librosa`**: The core signal processing engine. We use this instead of standard Python audio libraries because it is optimized for Music Information Retrieval (MIR). It automatically handles decoding and resampling.
* **`numpy`**: The maths backend. Audio in this project is treated as a 1D Tensor (Vector) where $y[t]$ is the amplitude at time $t$.
* **`streamlit`**: The Interface layer. It allows us to separate backend from frontend without having to write complex JavaScript.

## 2. Key Decisions in `main.py`

### `sr=44100` (Sampling Rate)
* **Code:** `y, sr = librosa.load(..., sr=44100)`
* **Why:** We use the standard CD-quality sampling rate to stick to the **Nyquist-Shannon Sampling Theorem**.
    * To verify deepfake artifacts at **16kHz**, we need a sampling rate of at least $32kHz$.
    * Our previous sprint used 22.05kHz, which capped our visibility at ~11kHz. By doubling this to 44.1kHz, we extend our range of visibility up to **22.05kHz**, covering the entire spectrum of human hearing and critical high-frequency generation errors.

### The Waveform (Time-Domain)
* **Code:** `librosa.display.waveshow(...)`
* **Why:** This visualizes **Amplitude vs. Time**.
* **Utility:** Useful for checking file integrity and silence, but not enough to check for spoofing as it lacks frequency measure.

## 3. Frequency-Domain Analysis (Phase 1.5)

To identify AI-generated artifacts, we move past the Time Domain into the Frequency Domain.

### Short-Time Fourier Transform (STFT)
* **Code:** `D = librosa.stft(y)`
* **The Problem:** The standard Fourier Transform loses time information (it tells you *what* frequencies were present, sort of like a list, but not *when*).
* **The Solution:** We use STFT, which breaks the audio into short "windows" and applies a Fourier Transform to each. This creates a complex matrix representing frequency intensity over time.

### The Mel-Spectrogram
* **Code:** `librosa.amplitude_to_db(np.abs(D))`
* **Why:** Raw frequencies are linear, but human hearing (and voice generation) is logarithmic.
* **The Mel Scale:** We map the STFT output to the Mel Scale to emphasize the frequencies important for human speech.
* **Deepfake Detection Utility:** This visualization allows us to spot:
    * **High-Frequency Cutoffs:** Generative models often fail to produce noise above certain thresholds (e.g., 16kHz), leaving these black gaps in the spectrum.
    * **Spectral Texture:** Smoothness or checkerboard patterns that differ from biological noise.
