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

## Entry 5: The Architecture Definition (Jan 11, 2026)
**Status:** Phase 3 (Model Locked)
- **Objective:** Define the layers of the Neural Network in `src/model.py`.
- **Architecture Choice:** 3-Block CNN.
    - **Why:** Deepfakes in the logical access dataset often have specific texture artifacts. A CNN is the standard for recognising these textures.
    - **Dimensions:** Input (128x128) -> Conv1/Pool (64x64) -> Conv2/Pool (32x32) -> Conv3/Pool (16x16).
    - **Flattening:** The resulting feature map (128 channels * 16 * 16) results in 32,768 features entering the dense layer.
- **Hardware Check:** Validated input/output shapes using `torch.randn`. The model suits the 16GB on a Macbook M3 Air.
- **Next Steps:** Create `src/train.py` to construct the Training Loop (Loss Calculation & Backpropagation).

## Entry 6: The Training Run (Jan 17, 2026)
**Status:** Phase 4 (Training Complete)
- **Objective:** Train the SpectrogramCNN on the processed dataset.
- **Outcome:** The model improved accuracy rapidly.
    - **Epoch 1:** ~89% Accuracy (The model identified the "Black Void" artifacts immediately).
    - **Epoch 10:** 99.49% Accuracy (Refined texture analysis).
- **Performance:** M3 Air (MPS) handled the load efficiently (~60s per epoch dropping from ~140s).
- **Artifact:** Model saved as `models/spectral_cnn_v1.pth`.

## Entry 7: The Interface Deployment (Jan 17, 2026)
**Status:** Phase 5 (Operational)
- **Objective:** Deploy the trained model to a user-facing dashboard.
- **Implementation:** Built a Streamlit interface in `main.py`.
- **Pipeline:**
    1.  User uploads Audio.
    2.  System preprocesses to Mel-Spectrogram (matching Training Config).
    3.  Model Inference (MPS Acceleration).
    4.  Output: Classification + Confidence Score.
- **Outcome:** **SUCCESS.** The system runs in real-time on the M3 Air.
- **Next Steps:** Final Evaluation and Project Write-up.

## Entry 8: The Pipeline Synchronization (Jan 17, 2026)
**Status:** Phase 5 (Interface Debugging)
- **Issue:** Initial deployment showed high accuracy on test set but flagged valid training files as "Spoof".
- **Root Cause Analysis:** Discovered a mismatch between `processor.py` (Training) and `main.py` (Inference).
    - **Math Mismatch:** Processor used `amplitude_to_db` (incorrect physics but consistent), Main used `power_to_db`.
    - **Resolution Mismatch:** Processor generated small plots (4x4 inches), Main generated large plots (10x10 inches). Resizing these to 128px created different line thicknesses.
- **The Fix:** Rewrote `main.py` image generation to strictly mimic `processor.py` artifacts (figsize=4x4, amplitude_to_db).
- **Outcome:**
    - Validated with Training File A: **99.9% Bonafide Confidence.**
    - Validated with Training File B: **64% Spoof Confidence** (Outlier due to time-compression distortion).
    - Validated with Live Mic: **Spoof** (Confirmed Domain Shift hypothesis).
- **Conclusion:** The system is operational. The "AI Detection" is highly sensitive to acoustic environments such as room reverb.
