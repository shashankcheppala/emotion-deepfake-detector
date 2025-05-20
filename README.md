# Emotion-Aware Hybrid Model for Deepfake-Video Detection

This project implements a two-stream hybrid deepfake detector that fuses visual features from RGB frames with emotion vectors extracted from facial expressions. It leverages EfficientNetB0 for spatial encoding and a lightweight emotion RNN branch to capture temporal affective cues. All experiments were conducted on the CelebDF-V2 dataset.

## Highlights

- Precomputed 8-frame RGB stacks per video
- Per-frame emotion extraction via FER + MTCNN
- Baselines:
  - Emotion-only Logistic Regression
  - Scratch-trained FrameCNN
  - Bi-LSTM on emotion vectors
  - Frozen VideoMAE + Logistic
- Final Hybrid Model:
  - TimeDistributed EfficientNetB0
  - Conv1D temporal head + emotion fusion
  - Binary Focal Loss + cosine learning rate decay

## Performance Summary (Validation AUC)

| Model                  | AUC    |
|------------------------|--------|
| Emotion Logistic       | ~0.645 |
| FrameCNN               | ~0.563 |
| Emotion RNN            | ~0.657 |
| VideoMAE + Logistic    | ~0.680 |
| **Hybrid CNN + Emotion** | **~0.85**  |

## Data

Dataset: [CelebDF-V2](https://github.com/yuezunli/Celeb-DF)  
Precomputed `.npy` and `.pkl` files are stored in Google Drive.  
Use `data/README.md` for instructions on structure and format.

## Tech Stack

- Python, TensorFlow/Keras, PyTorch
- Google Colab, OpenCV, matplotlib
- FER, HuggingFace Transformers, EfficientNet

## Repo Structure

emotion-aware-deepfake-detector/
├── notebooks/
├── utils/
├── models/
├── data/
├── README.md
├── requirements.txt
└── .gitignore


## Author

Developed by **Shashank Cheppala**  
> Master's in Data Analytics | University of Illinois Springfield  
> [GitHub](https://github.com/shashankcheppala)



