import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, applications

@tf.keras.utils.register_keras_serializable()
class EfficientNetPreprocessor(layers.Layer):
    def call(self, x):
        return applications.efficientnet.preprocess_input(x * 255.0)

def build_hybrid_model(num_frames=8, img_h=224, img_w=224, dropout_rate=0.4, l2_reg=1e-4):
    reg = regularizers.l2(l2_reg)
    frames_in = layers.Input((num_frames, img_h, img_w, 3), name="frames")
    emos_in   = layers.Input((num_frames, 7), name="emotions")

    x = layers.GaussianNoise(0.01)(frames_in)
    x = layers.TimeDistributed(EfficientNetPreprocessor())(x)
    base = applications.EfficientNetB0(include_top=False, weights="imagenet")

    for i, layer in enumerate(base.layers):
        layer.trainable = (i >= int(0.6 * len(base.layers)))

    x = layers.TimeDistributed(base)(x)
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
    x = layers.Conv1D(256, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x)

    e = layers.GlobalAveragePooling1D()(emos_in)

    combined = layers.Concatenate()([x, e])
    for units in [256, 64]:
        combined = layers.Dense(units, activation="relu", kernel_regularizer=reg)(combined)
        combined = layers.BatchNormalization()(combined)
        combined = layers.Dropout(dropout_rate)(combined)

    output = layers.Dense(1, activation="sigmoid", dtype="float32")(combined)
    return models.Model([frames_in, emos_in], output)
