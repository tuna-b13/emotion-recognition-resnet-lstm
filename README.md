# Emotion Recognition  ResNet18 + LSTM

*Individual contribution to a CE903 group project  University of Essex*

Video-based facial emotion recognition using a ResNet18 + LSTM architecture, trained on the RAVDESS dataset.

![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)

---

## About this repo

This was a **CE903 group project** comparing 6 deep learning architectures (CNN, SE-CNN+BiLSTM, ResNet18+LSTM, ViT, DETR, AVNet) for facial emotion recognition. The group's best-performing model overall was **DETR, at 90% test accuracy**.

This repo contains **my individual contribution**: the ResNet18 + LSTM model, built and evaluated independently.

---

## Overview

The model recognises 5 emotions from video clips by extracting facial frames and modelling temporal patterns across the sequence.

- **Architecture:** Pretrained ResNet18 (feature extraction) + 2 layer LSTM (temporal modelling)
- **Dataset:** RAVDESS — 24 actors, speech and song modality
- **Classes:** neutral, happy, sad, angry, fearful
- **Validation Accuracy:** 77.6% (576 videos)

---

## Project Structure

```
emotion-recognition-resnet-lstm/
├── scripts/
│   ├── prepare_ravdess.py       # Organise raw RAVDESS videos by emotion label
│   ├── extract_faces.py         # Extract face frames using Haar Cascade
│   └── make_split.py            # Actor-level train/val/test split
│
├── data/
│   ├── train.json               # Training split
│   ├── val.json                 # Validation split
│   └── test.json                # Test split
│   (raw videos / extracted frames / manual test videos are git-ignored — too large for the repo)
│
├── models/
│   ├── emotion_resnet_lstm_best.pth   # Best checkpoint
│   └── train_history.json             # Loss/accuracy per epoch
│   (per-epoch checkpoints are git-ignored)
│
├── plots/                       # Generated visualisation outputs
│
├── ResNet_lstm_train.py         # Training script
├── ResNet_lstm_predictvd.py     # Validation evaluation script
├── ResNet_lstm_visualize.py     # Plot training curves and confusion matrix
└── realtime_demo.py             # Live webcam inference with HUD overlay
```

---

## Requirements

```
torch
torchvision
opencv-python
Pillow
numpy
scikit-learn
pandas
tqdm
matplotlib
seaborn
```

Install all:

```bash
pip3 install torch torchvision opencv-python Pillow numpy scikit-learn pandas tqdm matplotlib seaborn
```

---

## Usage

### 1. Prepare the dataset

Place raw RAVDESS videos in `data/ravdess_raw/`, then run the pipeline in order:

```bash
python3 scripts/prepare_ravdess.py
python3 scripts/extract_faces.py
python3 scripts/make_split.py
```

### 2. Train the model

```bash
python3 ResNet_lstm_train.py
```

- Best model is saved to `models/emotion_resnet_lstm_best.pth`
- Training history is saved to `models/train_history.json`

### 3. Evaluate on validation set

```bash
python3 ResNet_lstm_predictvd.py
```

Outputs per-class accuracy, speech vs. song breakdown, classification report, confusion matrix, and saves `resnet_lstm_val_results.csv` / `resnet_lstm_classification_report.txt`.

### 4. Generate training plots

```bash
python3 ResNet_lstm_visualize.py
```

### 5. Real-time webcam demo

```bash
python3 realtime_demo.py
```

Press `q` to quit.

---

## Results

### Training Summary

| Setting          | Value                             |
|-------------------|------------------------------------|
| Epochs            | 30                                  |
| Optimizer         | AdamW (lr=5e-5, wd=1e-4)            |
| Loss              | CrossEntropyLoss + class weights    |
| Batch size        | 8                                   |
| Sequence length   | 16 frames                           |
| **Val accuracy**  | **77.6% (576 videos)**              |
| **Macro F1**      | **0.773**                           |

### Validation Classification Report

| Emotion       | Precision | Recall | F1    | Support |
|---------------|-----------|--------|-------|---------|
| neutral       | 0.520     | 1.000  | 0.684 | 64      |
| happy         | 1.000     | 0.797  | 0.887 | 128     |
| sad           | 0.722     | 0.609  | 0.661 | 128     |
| angry         | 0.955     | 0.828  | 0.887 | 128     |
| fearful       | 0.735     | 0.758  | 0.746 | 128     |
| **macro avg** | **0.786** | **0.798** | **0.773** | **576** |

**Overall accuracy: 77.6%** — Speech: 77.8% · Song: 77.4% (no modality bias)

### Per-Class Accuracy

| Emotion  | Accuracy |
|----------|----------|
| neutral  | 100.0%   |
| happy    | 79.7%    |
| sad      | 60.9%    |
| angry    | 82.8%    |
| fearful  | 75.8%    |

### Confusion Matrix (row = true, col = predicted)

|          | neutral | happy | sad | angry | fearful |
|----------|---------|-------|-----|-------|---------|
| neutral  | **64**  | 0     | 0   | 0     | 0       |
| happy    | 22      | **102**| 2  | 0     | 2       |
| sad      | 26      | 0     | **78**| 0   | 24      |
| angry    | 3       | 0     | 10  | **106**| 9      |
| fearful  | 8       | 0     | 18  | 5     | **97**  |

### Cross-Dataset Generalisation (CREMA-D, unseen)

A supplementary evaluation was run on 50 clips from CREMA-D — a dataset never seen during training or validation, differing from RAVDESS in actors, recording conditions, and portrayal style.

**Overall accuracy: 46.0% (23/50)**

| Emotion  | Accuracy |
|----------|----------|
| angry    | 90% (9/10)  |
| fearful  | 60% (6/10)  |
| happy    | 60% (6/10)  |
| neutral  | 10% (1/10)  |
| sad      | 10% (1/10)  |

High-arousal emotions (angry, happy) generalise reasonably well; low-arousal emotions (neutral, sad) collapse out-of-distribution — the same failure pattern seen in the RAVDESS confusion matrix, suggesting it reflects a structural model limitation rather than dataset-specific overfitting.

---

## Model Architecture

```
Input: (B, 16, 3, 224, 224)
  └─ ResNet18 (pretrained, ImageNet)
       └─ FC replaced: 512 → 256
  └─ 2-layer LSTM (hidden=256, dropout=0.3)
  └─ Dropout(0.4) → Linear(256 → 5)
Output: 5-class emotion logits
```

---

## Dataset

**RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song)
- 24 actors (12 male, 12 female)
- Modalities: speech and song
- Emotions used: neutral, happy, sad, angry, fearful
- Split: 16 actors train / 4 actors val / 4 actors test (actor-level, no identity leakage)

**CREMA-D** used only for out-of-distribution evaluation (not trained on).

---

## License

MIT — see [LICENSE](LICENSE).
