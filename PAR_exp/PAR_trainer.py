import pandas as pd
import tensorflow as tf
from keras._tf_keras.keras.layers import Input, Dense, Reshape, Conv1D
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.optimizers import AdamW
from keras._tf_keras.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from liquid_attention import LAN

# Configuration
MODEL_NAME = "PAR_LAN_Exact"
WEIGHTS_DIR = "model_weights"
STATS_DIR = "statistics"
DATA_PATH = "PAR/balanced_person_B.csv"

# Load dataset
df = pd.read_csv(DATA_PATH)

# Features and labels
X = df[["x", "y", "z", "time_seconds"]].values
y = df["activity_id"].values
num_classes = df["activity_id"].nunique()

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.1, random_state=42
)

# TensorFlow datasets
train_ds = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .shuffle(10000)
    .batch(20)
)
val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(20)


def build_model(input_shape=(4,), num_classes=num_classes):
    """Builds the LAN-based classification model."""
    inputs = Input(shape=input_shape)
    x = Reshape((1, input_shape[0]))(inputs)
    x = Conv1D(64, kernel_size=5, padding="same", activation="relu")(x)
    x = Conv1D(64, kernel_size=3, padding="same", activation="relu")(x)
    x = LAN(d_model=32, num_heads=4, mode="exact", return_sequences=False)(x)
    x = Dense(32, activation="relu")(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    return Model(inputs, outputs)


# Build and compile model
model = build_model()
model.compile(
    optimizer=AdamW(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# Callbacks
reduce_lr = ReduceLROnPlateau(
    monitor="val_accuracy", factor=0.5, patience=5, min_lr=1e-10, mode="max"
)
checkpoint = ModelCheckpoint(
    filepath=f"{WEIGHTS_DIR}/{MODEL_NAME}.weights.h5",
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    save_weights_only=True,
)

# Training 
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=500,
    callbacks=[checkpoint, reduce_lr],
)

# Save full model (optional)
# model.save(f"{WEIGHTS_DIR}/{MODEL_NAME}.keras")

# Load best weights and evaluate
# model.load_weights(f"{WEIGHTS_DIR}/{MODEL_NAME}.weights.h5")
# loss, accuracy = model.evaluate(val_ds)
# print(f"Test Accuracy: {accuracy * 100:.2f}%")
