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

## 4. The Data Pipeline (Phase 2: Ingestion)

To train a Convolutional Neural Network (CNN), we must convert our problem from the **Audio Domain** (1D Signals) to the **Visual Domain** (2D Images).

### The Processor (`src/processor.py`)
This script handles the mass-transmutation of the ASVspoof 2019 dataset.

1.  **Protocol Mapping:** It reads the `ASVspoof2019.LA.cm.train.trn.txt` file to determine the Label for every file.
2.  **Sorting:** It automatically sorts generated images into `data/processed/bonafide` (Real) and `data/processed/spoof` (Fake).
3.  **Image Generation:**
    * **Input:** `.flac` audio file.
    * **Process:** STFT $\rightarrow$ Mel-Scale $\rightarrow$ Decibel Conversion.
    * **Output:** `.png` image (128x128 approx).

### Critical Decision: The "Naked" Spectrogram
* **Code:** `plt.axis('off')`, `plt.tight_layout(pad=0)`
* **Why:** Standard `matplotlib` charts include white borders, axes, and labels. These are "noise" to a Neural Network.
* **Result:** The generated images contain *only* the data we need. This maximizes the "Signal-to-Noise Ratio" for the AI, ensuring it learns from the audio artifacts, not the font size of the axis labels. We want to minimise any inaccuracies and potential confusion.


## The Architecture (Phase 3: The Brain)
* **Input:** 128x128 Pixel Mel-Spectrograms (Grayscale, 1 Channel).
* **Class:** `SpectrogramCNN` (Defined in `src/model.py`)
* **Layer Breakdown:**
    1.  **Conv Block 1:** * Input: 1 Channel (Grayscale)
        * Filters: 32 (Kernel 3x3)
        * Pooling: MaxPool 2x2 
        * *Output Dimension:* 32 Channels x 64 x 64
    2.  **Conv Block 2:**
        * Filters: 64 (Kernel 3x3)
        * Pooling: MaxPool 2x2
        * *Output Dimension:* 64 Channels x 32 x 32
    3.  **Conv Block 3:**
        * Filters: 128 (Kernel 3x3)
        * Pooling: MaxPool 2x2
        * *Output Dimension:* 128 Channels x 16 x 16
    4.  **Classifier:**
        * **Flatten:** Unrolls 3D volume into 1D vector ($128 \times 16 \times 16 = 32,768$ features).
        * **FC1 (Dense):** 128 Neurons + ReLU.
        * **Dropout:** 0.5 (Randomly zeroes 50% of neurons to prevent overfitting).
        * **FC2 (Output):** 2 Neurons (Logits for Bonafide vs Spoof).


## 4. Training Configuration
* **Optimizer:** Adam (LR=0.001).
* **Loss Function:** CrossEntropyLoss.
* **Input Constraint:** All images resized to **128x128** Grayscale.
* **Dynamics:** Rapid convergence suggests high separability between real and spoofed audio in the frequency domain.