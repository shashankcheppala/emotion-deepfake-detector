Here's your **complete professional `README.md`** – copy-paste this entire block into your `emotion-deepfake-detector/README.md` file:

---

```markdown
# Emotion-Aware Hybrid Model for Deepfake-Video Detection

A two-stream deepfake detection system that fuses spatial frame features with temporal emotion patterns. The model combines EfficientNetB0-based RGB frame embeddings with FER-extracted per-frame emotion probabilities, achieving strong performance on the CelebDF-V2 dataset.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shashankcheppala/emotion-deepfake-detector/blob/main/notebooks/deepfake_project.ipynb)

---

## 🧠 Architecture

- **Visual Stream:**  
  TimeDistributed EfficientNetB0 → GlobalAveragePooling → Conv1D temporal head  
- **Emotion Stream:**  
  8×7 emotion matrix → GlobalAveragePooling1D  
- **Fusion:**  
  Concatenate → Dense(256→64) → Dropout → Sigmoid output  
- **Training:**  
  Binary Focal Loss (γ=2, α=0.75), Warm-up + Cosine LR Decay, Mixed Precision

---

## 📊 Validation Results

| Model                  | Val AUC |
|------------------------|---------|
| Emotion-Only Logistic  | ~0.645  |
| FrameCNN (scratch)     | ~0.563  |
| Emotion RNN            | ~0.657  |
| VideoMAE + Logistic    | ~0.680  |
| **Hybrid CNN + Emotion** | **~0.850** |

---

## 📁 Repository Structure

```

emotion-deepfake-detector/
├── notebooks/           # Jupyter notebooks for experiments
│   └── deepfake\_project.ipynb
├── src/                 # Core Python modules (preprocessing, training, evaluation)
│   ├── preprocessing.py
│   ├── training.py
│   └── evaluate.py
├── models/              # Trained .keras model weights
├── data/                # Sample or link-only (no raw datasets)
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE

````

---

## 🔧 Setup Instructions

```bash
git clone https://github.com/shashankcheppala/emotion-deepfake-detector.git
cd emotion-deepfake-detector
pip install -r requirements.txt
````

---

## 🚀 How to Run

### Training the Hybrid Model

```python
from src.training import train_hybrid_model

train_hybrid_model(
    frames_dir="path/to/frames",
    emotion_pkl="path/to/emotions.pkl",
    save_path="models/hybrid_model.keras"
)
```

---

## 📦 Dataset

* **CelebDF-V2:**
  [https://github.com/yuezunli/Celeb-DF](https://github.com/yuezunli/Celeb-DF)

* Preprocessed:

  * 8×224×224 RGB frames per clip (.npy)
  * 8×7 per-frame emotion vectors (.pkl)

---

## 📈 Features

* 🧠 Emotion-Aware Detection
* 🎞️ EfficientNetB0 + Conv1D for temporal consistency
* 📊 ROC, AUC, and full evaluation pipeline
* ⚡ Mixed Precision & Cosine Decay LR Scheduling
* 🧪 Baselines: Emotion-only, CNN-only, RNN, VideoMAE+Logistic

---

## 👤 Author

**Shashank Cheppala**
M.S. in Data Analytics, University of Illinois Springfield
[GitHub](https://github.com/shashankcheppala) • [LinkedIn](https://www.linkedin.com/in/shashank-cheppala-6455ab1a4/)

---

## 🪪 License

This repository is released under the [MIT License](LICENSE).

```

---

Let me know if you want me to generate `requirements.txt` based on the code you used in the project.
```
