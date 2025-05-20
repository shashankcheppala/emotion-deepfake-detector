# Emotion-Aware Hybrid Model for Deepfake-Video Detection

A two-stream deepfake detection system that fuses spatial frame features with temporal emotion patterns. The model combines EfficientNetB0-based RGB frame embeddings with FER-extracted per-frame emotion probabilities, achieving strong performance on the CelebDF-V2 dataset.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shashankcheppala/emotion-deepfake-detector/blob/main/notebooks/deepfake_project.ipynb)

---

## Architecture

- Visual Stream:  
  TimeDistributed EfficientNetB0 → GlobalAveragePooling → Conv1D temporal head

- Emotion Stream:  
  8×7 emotion matrix → GlobalAveragePooling1D

- Fusion:  
  Concatenate → Dense(256→64) → Dropout → Sigmoid output

- Training:  
  Binary Focal Loss (γ=2, α=0.75), Warm-up + Cosine LR Decay, Mixed Precision

---

## Validation Results

| Model                  | Val AUC |
|------------------------|---------|
| Emotion-Only Logistic  | ~0.645  |
| FrameCNN (scratch)     | ~0.563  |
| Emotion RNN            | ~0.657  |
| VideoMAE + Logistic    | ~0.680  |
| Hybrid CNN + Emotion   | ~0.850  |

---

## Repository Structure

```
emotion-deepfake-detector/
├── notebooks/              
│   └── deepfake_project.ipynb
├── src/                 
│   ├── preprocessing.py
│   ├── training.py
│   └── evaluate.py
├── models/          
├── data/               
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```


---

## Setup Instructions

git clone https://github.com/shashankcheppala/emotion-deepfake-detector.git
cd emotion-deepfake-detector
pip install -r requirements.txt

## How to Run

### Training the Hybrid Model


from src.training import train_hybrid_model

train_hybrid_model(
    frames_dir="path/to/frames",
    emotion_pkl="path/to/emotions.pkl",
    save_path="models/hybrid_model.keras"
)

## Running Evaluation


from src.evaluate import evaluate_model

evaluate_model(
    model_path="models/hybrid_model.keras",
    frames_dir="path/to/frames",
    emotion_pkl="path/to/emotions.pkl"
)

## Dataset

- **CelebDF-V2**  
  Dataset link: https://github.com/yuezunli/Celeb-DF

- **Preprocessed Format**
  - 8×224×224 RGB frame arrays per clip (.npy)
  - 8×7 emotion probability vectors per clip (.pkl)

Preprocessing scripts are available in `src/preprocessing.py`.

## Features

- Hybrid detection using both visual and emotion-based signals
- TimeDistributed EfficientNetB0 with fine-tuned temporal Conv1D
- Emotion extraction via FER + MTCNN
- Focal Loss for robust handling of imbalanced data
- Cosine learning rate scheduler with warm-up
- Full evaluation pipeline: ROC, AUC, classification reports
- Multiple baselines: logistic, RNN, CNN, and transformer-based

## License

This repository is released under the MIT License. See `LICENSE` file for details.


