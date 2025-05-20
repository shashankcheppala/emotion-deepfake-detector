
import numpy as np
import tensorflow as tf
from src.training import build_hybrid_model

model = tf.keras.models.load_model(
    'models/deepfake_hybrid_model.keras',
    custom_objects={'EfficientNetPreprocessor': tf.keras.utils.get_custom_objects()['EfficientNetPreprocessor']}
)

frames = np.load('sample_clip.npy')       # Shape: (8, 224, 224, 3)
emotions = np.load('sample_emotion.npy')  # Shape: (8, 7)

frames = np.expand_dims(frames, axis=0)
emotions = np.expand_dims(emotions, axis=0)

prob = model.predict([frames, emotions])[0][0]
label = "Fake" if prob >= 0.5 else "Real"

print(f"Predicted class: {label} (Probability: {prob:.3f})")
