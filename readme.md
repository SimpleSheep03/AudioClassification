# 🎧 Audio Classification with EfficientNet-B2

This repository contains an end-to-end audio classification pipeline built using **PyTorch**, **torchaudio**, and **timm**.  
The project focuses on classifying environmental sounds from the dataset (or similar structured datasets) using **mel spectrograms**, **delta features**, and **mixup augmentation**.  
It includes a full cross-validation training pipeline and a test-time inference script for unseen audio clips.

---

## 🚀 Features
- EfficientNet-B2 backbone for high-performance classification  
- Mel-spectrogram + delta + delta-delta as 3-channel image inputs  
- SpecAugment (time & frequency masking)  
- Waveform augmentations using torch-audiomentations  
- Mixup training for improved generalization  
- 5-fold cross-validation  
- Inference-ready pipeline for test datasets  
- Reproducibility ensured with fixed seeds  

---

## 🧠 Model Overview
The model converts each audio waveform into a 3-channel image:
1. Mel-spectrogram  
2. Delta (velocity)  
3. Delta-Delta (acceleration)

These are then resized and fed into **EfficientNet-B2** (from [timm](https://github.com/huggingface/pytorch-image-models)).

---

---

## ⚙️ Installation

### 1️⃣ Clone this repository
```
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2️⃣ Create a virtual environment
```
python -m venv venv
source venv/bin/activate    # (Linux/Mac)
venv\Scripts\activate       # (Windows)
```

### 3️⃣ Install dependencies
```
pip install -r requirements.txt
```

**Requirements:**
- torch  
- torchaudio  
- timm  
- torch-audiomentations  
- librosa  
- pandas  
- numpy  
- matplotlib  
- seaborn  
- tqdm  

---

## 🧩 Dataset Format
The metadata CSV (e.g., `sound_50.csv`) should have the following columns:

| filename           | fold | target | category  |
|--------------------|-------|--------|------------|
| 1-100032-A-0.wav   | 1     | 0      | dog_bark   |
| 1-100038-A-14.wav  | 1     | 14     | rain       |

Audio files are expected under:
```
DATA_PATH/audio/
```

Each sample should be ≤ 5 seconds, sampled at **44.1 kHz**.

---

## 🏋️‍♂️ Training
To train using **5-fold cross-validation**, run the notebook cells in order.

The training loop will:
- Train on 4 folds and validate on the remaining one  
- Save the best model for each fold (`best_model_fold_X.pth`)  
- Print fold accuracies and overall average accuracy  

Example output:
```
===== FOLD 1 =====
Epoch 36/50 | Train Loss: 1.4147, Train Acc (mixup-aware): 0.8271 | Val Loss: 1.1687, Val Acc: 0.9025
  -> New best validation accuracy: 0.9025. Model saved.

Best validation accuracy for fold 1: 0.9025
Average CV Accuracy: 0.9025 ± 0.018
```

---

## 🧪 Inference
To run inference on unseen audio data:

Update paths in the inference section:
```
TEST_AUDIO_PATH = "path/to/test_set"
MODEL_PATH = "path/to/best_model_fold_2.pth"
OUTPUT_CSV = "test_predictions.csv"
```

Run the inference cells in the notebook.

Predictions will be saved as:
```
test_predictions.csv
```

**Sample Output:**
| id              | prediction |
|-----------------|-------------|
| 7-280602-A-001  | 1           |
| 7-280602-A-002  | 0           |

---

## 🧩 Key Components

| Component | Description |
|------------|-------------|
| Dataset | Custom PyTorch dataset for training/validation |
| TestDataset | Dataset for inference |
| spec_transform | Mel-spectrogram + AmplitudeToDB |
| delta_transform | Delta computation for temporal derivatives |
| waveform_augmenter | Gain, Noise, PitchShift, and Shift |
| mixup_data() | Mixup implementation |
| train_one_epoch() / validate_one_epoch() | Training loop |
| get_model() | Loads EfficientNet-B2 from timm |

---

## 📊 Results

| Fold | Best Val Accuracy |
|------|--------------------|
| 1    | 0.90 |
| 2    | 0.89 |
| 3    | 0.90 |
| 4    | 0.90 |
| 5    | 0.89 |
| **Average** | **0.90 ± 0.011** |


---

## 💾 Output
The final trained models are saved as:
```
best_model_fold_1.pth
best_model_fold_2.pth
...
```

Predictions for test data are saved in:
```
test_predictions.csv
```

---

## 💡 Tips
- If you face CUDA OOM errors, reduce `CFG.BATCH_SIZE` or use a smaller model like `efficientnet_b0`.  
- You can enable clean (non-mixup) training accuracy by setting:
  ```
  CFG.CLEAN_TRAIN_ACC = True
  ```
- Try different values of `CFG.MIXUP_ALPHA` for stronger regularization.

---
