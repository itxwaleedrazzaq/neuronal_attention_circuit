import numpy as np
import tensorflow as tf
from liquid_attention import LAN
from keras._tf_keras.keras.layers import Input, Dense, Conv1D
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.optimizers import AdamW
from keras._tf_keras.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

# Configuration
MODEL_NAME = "mnist_LAN"
FEATURE_DIR = "tf_features"
WEIGHTS_DIR = "model_weights"
STATS_DIR = "statistics"

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


# -----------------------------
# Preprocessing functions
# -----------------------------
def binarize(img: np.ndarray, threshold: int = 128) -> np.ndarray:
    """Threshold image to binary values (0 or 1)."""
    return (img >= threshold).astype(np.int32)


def sequentialize(img: np.ndarray) -> np.ndarray:
    """Flatten 28x28 image into sequence of length 784."""
    return img.flatten()


def event_based_compression(seq: np.ndarray):
    """Convert binary sequence into (value, time-lag) event format."""
    events = []
    prev_val = seq[0]
    t = 1
    for i in range(1, len(seq)):
        if seq[i] == prev_val:
            t += 1
        else:
            events.append((prev_val, t))
            prev_val = seq[i]
            t = 1
    events.append((prev_val, t))
    return events


def normalize_time(events, max_len: int = 256):
    """Rescale durations so that total length fits within max_len."""
    total_time = sum(t for _, t in events)
    scale = max_len / total_time
    return [(val, t * scale) for val, t in events]


def pad_events(events, max_len: int = 256):
    """Pad or truncate events to fixed length."""
    events = events[:max_len]
    if len(events) < max_len:
        events += [(0, 0)] * (max_len - len(events))
    return events


def preprocess_dataset(images, labels, max_len: int = 256):
    """Full preprocessing pipeline for a dataset."""
    event_data = []
    for img in images:
        binary_img = binarize(img)
        seq = sequentialize(binary_img)
        events = event_based_compression(seq)
        events = normalize_time(events, max_len)
        padded = pad_events(events, max_len)
        event_data.append(padded)
    return np.array(event_data, dtype=np.float32), labels


# -----------------------------
# Apply preprocessing
# -----------------------------
x_train_events, y_train = preprocess_dataset(x_train, y_train)
x_test_events, y_test = preprocess_dataset(x_test, y_test)

print("Example processed sequence shape:", x_train_events[0].shape)  # (256, 2)
print("Training set shape:", x_train_events.shape)  # (60000, 256, 2)

# Convert to TensorFlow dataset
train_ds = (
    tf.data.Dataset.from_tensor_slices((x_train_events, y_train))
    .shuffle(10000)
    .batch(32)
)
test_ds = tf.data.Dataset.from_tensor_slices((x_test_events, y_test)).batch(32)


# -----------------------------
# Model
# -----------------------------
def build_model(input_shape=(256, 2)):
    """Build LAN-based classification model for MNIST."""
    inputs = Input(shape=input_shape)
    x = Conv1D(64, kernel_size=5, strides=5, padding="same", activation="relu")(inputs)
    x = Conv1D(64, kernel_size=5, strides=5, padding="same", activation="relu")(x)
    x = LAN(d_model=64, num_heads=8, delta_t=0.1, mode="exact", return_sequences=False)(x)
    x = Dense(32, activation="relu")(x)
    outputs = Dense(10, activation="softmax")(x)
    return Model(inputs, outputs)


model = build_model()
model.summary()

model.compile(
    optimizer=AdamW(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# -----------------------------
# Training
# -----------------------------
reduce_lr = ReduceLROnPlateau(
    monitor="val_accuracy", factor=0.3, patience=3, min_lr=1e-10, mode="max"
)
checkpoint = ModelCheckpoint(
    filepath=f"{WEIGHTS_DIR}/{MODEL_NAME}.weights.h5",
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    save_weights_only=True,
)

model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=150,
    callbacks=[reduce_lr, checkpoint],
)

# -----------------------------
# Evaluation
# -----------------------------
# model.save(f"{WEIGHTS_DIR}/{MODEL_NAME}.keras")
# model.load_weights(f"{WEIGHTS_DIR}/{MODEL_NAME}.weights.h5")

loss, acc = model.evaluate(test_ds)
print(f"Test Accuracy: {acc * 100:.2f}%")
