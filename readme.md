## 🏁 Objective
The primary goal of this project is to classify 50 different types of environmental sounds from the dataset provided.
We approach this by transforming a **1D audio time series classification** problem into a **2D image classification** task using **Mel Spectrograms** and a **pre-trained CNN** model.

---

## 🧠 Overall Strategy
Our strategy integrates **signal processing (time series concepts)** with **deep learning (image classification)**:

1. Treat audio files as **1D time series data**.
2. Apply **time-domain augmentations** to enrich the dataset.
3. Convert each waveform into a **2D Mel Spectrogram** — a visual representation of frequencies over time.
4. Feed these 2D spectrograms into a **pre-trained EfficientNet-B0** model for classification.

---

## ⚙️ Key Components

### 1. Time-Domain Augmentation (Raw Waveform)
Before feature extraction, we manipulate the raw **1D waveform** to create diverse samples using `torch_audiomentations`.

#### Techniques Used
| Augmentation | Description | Purpose |
|---------------|-------------|----------|
| **Gain** | Randomly increases/decreases amplitude (volume). | Makes model volume-invariant. |
| **AddColoredNoise** | Adds synthetic background noise (pink/white noise). | Simulates real-world noisy conditions. |
| **PitchShift** | Shifts pitch up/down without altering speed. | Increases robustness to pitch variations. |
| **Shift** | Shifts waveform in time domain. | Handles delays and alignment variations. |

🧩 **Why it helps:**  
These augmentations ensure the model focuses on the *essence* of the sound rather than recording conditions — improving generalization and preventing overfitting.

---

### 2. Time-Frequency Analysis (Feature Extraction)
Once augmented, the 1D waveform is converted into a **2D Mel Spectrogram** using `torchaudio.transforms.MelSpectrogram`.

#### What Happens Internally
- The waveform is divided into **small, overlapping windows** (`WIN_LENGTH`, `HOP_LENGTH`).
- For each window, a **Short-Time Fourier Transform (STFT)** computes frequency information.
- Frequencies are then **mapped to the Mel scale**, mimicking human hearing sensitivity.
- The result is a 2D matrix:  
  - **X-axis:** Time  
  - **Y-axis:** Frequency bins  
  - **Values:** Amplitude (in dB)

📊 This conversion captures how frequency energy changes over time — enabling CNNs to recognize patterns similar to image textures.

---

### 3. Deep Learning Model (EfficientNet-B0)
- We use a **pre-trained EfficientNet-B0** model from the `timm` library.
- The first layer is modified to accept **1-channel (grayscale)** spectrograms instead of 3-channel RGB images.
- Input spectrograms are resized to **224×224** before being passed into the network.

#### Training Details
- **Loss Function:** CrossEntropy with label smoothing  
- **Optimizer:** AdamW  
- **Scheduler:** CosineAnnealingLR  
- **Batch Size:** 32  
- **Epochs:** 50  

---

### 4. Evaluation Strategy (5-Fold Cross-Validation)
We use the dataset’s predefined `fold` column for robust evaluation.

1. For each fold (1–5):
   - Train on 4 folds.
   - Validate on the remaining fold.
2. Save the best model per fold.
3. Compute the **average accuracy** across all folds.

This ensures the model’s performance is consistent and not dependent on a single split.

---

## 📈 Summary of Time Series Concepts
| Step | Concept | Implementation | Role |
|------|----------|----------------|------|
| **1** | Time-Domain Augmentation | Gain, Noise, PitchShift, Shift | Improves robustness to recording variability |
| **2** | Time-Frequency Analysis | Mel Spectrogram (STFT + Mel scaling) | Converts 1D temporal data → 2D frequency-time data |

---

## 🧩 Tech Stack
- **Python**
- **PyTorch**
- **Torchaudio**
- **Torch Audiomentations**
- **Timm (EfficientNet models)**
- **Librosa**
- **Matplotlib / Seaborn**

---

## 📁 Project Structure
