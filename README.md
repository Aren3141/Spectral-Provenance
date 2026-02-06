# Spectral-Provenance
An integrity protocol for the era where seemingly everything is fake. Uses deep learning to expose invisible frequency signatures left by GANs and diffusion models in audio files.

> **Status:** Active Development | **Focus:** Cybersecurity & Signal Processing

## 1. Abstract
The exponential rate of development of Generative Adversarial Networks (GANs) and Diffusion models has made AI generated speech practically indistinguishable from human speech to the naked ear. However, these models often leave minute, unnatural artifacts in the frequency domain. **Spectral-Provenance** is an algorithmic approach to detecting these small details. By converting raw waveforms into Mel-Spectrograms and analyzing them using a custom Convolutional Neural Network (CNN), we can classify audio with high precision, and this project, by doing this, aims to offer a defensive tool against deepfake misinformation.

## 2. The Problem
Voice cloning is no longer a theoretical threat; it is an active vector for social engineering and fraud. Our current way of detecting AI generated speech relies heavily on metadata, which can easily be spoofed. We need a biological/physics-based approach that analyzes the signal itself.

## 3. The Architecture (Planned)
* **Input:** `.wav` / `.mp3` audio files.
* **Preprocessing:** Fast Fourier Transform (FFT) to generate Mel-Spectrograms.
* **Model:** A PyTorch-based CNN trained (potentially) on the ASVspoof dataset.
* **Frontend:** Streamlit interface for real-time analysis visualization. Can be changed later.

## 4. Tech Stack
* **Language:** Python Latest Version (3.14 to date, Pi!)
* **Compute:** PyTorch (MPS acceleration for Mac)
* **Analysis:** Librosa, NumPy, Matplotlib
* **Interface:** Streamlit

## 5. Roadmap
- [x] **Phase 1:** Audio ingestion pipeline & Spectrogram visualization (The Maths).
- [x] **Phase 2:** Model architecture & training loop setup (The Logic).
- [x] **Phase 3:** Training on ASVspoof (The Optimization).
- [ ] **Phase 4:** **Generalization & Physics Alignment.**
    - *Current State:* Addressing "Domain Shift" between Studio Data and Real-World Microphones.
    - *Action:* Implementing a "Native 16kHz" pipeline to eliminate high-frequency mismatches.
- [ ] **Phase 5:** Inference engine & Streamlit Web UI (The Product).
---
*Author: [Aren Koprulu]*


