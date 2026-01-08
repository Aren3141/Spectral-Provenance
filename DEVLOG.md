# Development Log: Spectral-Provenance

## Entry 1: Project Initialization and The Time Domain (Dec 27, 2025)
**Status:** Phase 1 (Ingestion)
- **Objective:** Establish an audio ingestion pipeline to visualize raw waveform data.
- **Implementation:** Integrated `librosa` for signal processing and `streamlit` for the frontend, saving time from any complex unnecessary Javascript write-up.
- **Theoretical Constraint:** Initially utilized a sampling rate of 22.05kHz. While sufficient for standard speech synthesis (TTS), I hypothesized this would not be enough for forensic analysis, as it caps the Nyquist frequency at ~11kHz, potentially hiding high-frequency artifacts common in GANs.
- **Outcome:** Successfully rendered Time-Domain (Amplitude vs Time) plots.
- **Action Item:** Investigate Frequency-Domain analysis to identify "robotic" signatures invisible in the waveform.

## Entry 2: The Physics and Frequency Domain (Dec 30, 2025)
**Status:** Phase 1.5 (Spectral Analysis)
- **Pivot:** Moved from Time Domain to Frequency Domain using the Short-Time Fourier Transform (STFT).
- **Key Engineering Decision (The Nyquist Correction):**
    - *Problem:* Deepfake models often exhibit a hard cutoff at 16kHz.
    - *Solution:* changed sampling rate (`sr`) from 22.05kHz to **44.1kHz**. This expanded our range or spectrum to 22.05kHz, allowing detection of anything anomalous at high frequencies.
- **The Mathematics:**
    - Implemented STFT to decompose signals into complex magnitude and phase.
    - Applied the **Mel Scale** conversion to map linear frequencies to our human hearing, which is logarithmic.
- **Visual Output:** Generated Mel-Spectrograms (Heatmaps) capable of showing "black void" artifacts in synthetic audio.

## Entry 3: Data Transmutation (Jan 01, 2026)
**Status:** Phase 2 (Dataset Acquisition)
- **Objective:** Convert the ASVspoof 2019 dataset (Audio) into a Computer Vision-ready format.
- **Infrastructure:**
    - Coded a strict `.gitignore` policy to separate Source Code (<1MB) from Data (>25GB).
    - Established a `data/raw` vs `data/processed` directory structure for ease of access.
- **The "Processor" Protocol:**
    - Wrote `src/processor.py` to ingest 25,000+ `.flac` files.
    - **The "Naked" Spectrogram:** Configured `matplotlib` to strip all axes, labels, and borders (`plt.axis('off')`).
    - *Justification:* The Neural Network requires pure signal data. Human-readable labels are noise that could confuse the Model.
- **Outcome:** Successfully generated a dataset of labeled Spectrogram images, sorted into `bonafide` (Real) and `spoof` (Fake).
- **Next Steps:** Architecting the Convolutional Neural Network (CNN) in `src/model.py`.

## Entry 4: The Brain Initialization (Jan 08, 2026)
**Status:** Phase 3 (Model Implementation)
- **Objective:** Build the Convolutional Neural Network architecture.
- **Constraint:** Model must run efficiently on M3 MacBook Air using MPS without consuming excessive RAM.
- **Architecture Plan:**
    - Input: 128x128x1 (Grayscale Mel-Spectrograms).
    - Layers: 2 or 3 Convolutional blocks (Conv2d + ReLU + MaxPool).
    - Classifier: Fully Connected (Linear) layers for binary classification (Bonafide vs Spoof).
- **Action Item:** Implement `SpectrogramCNN` class in `src/model.py`.
