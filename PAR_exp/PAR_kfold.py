import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Reshape, Conv1D, LSTMCell, GRUCell,
    RNN, SimpleRNNCell, MultiHeadAttention, Flatten, Attention
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from ncps.tf import LTCCell, CfCCell
from ncps.wirings import FullyConnected
from baseline_cells import CTRNNCell, ODELSTM, PhasedLSTM, GRUODE, ODEformer, CTA
from tensorflow.keras.optimizers import AdamW
from neuronal_attention_circuit import NAC
from NAC_with_Pairwise import NAC_PW
from NAC_with_FC import NAC_FC
from NAC_with_PWFC import NAC_PWFC


# Config
base_model_name = 'Person_Activity_Recognition'
feature_dir = 'PAR'
weights_dir = 'model_weights'
os.makedirs(weights_dir, exist_ok=True)

# Load dataset
df = pd.read_csv(f"{feature_dir}/balanced_person_B.csv")

# Features and labels
X = df[["x", "y", "z", "time_seconds"]].values
y = df["activity_id"].values
num_classes = len(np.unique(y))

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Wiring for custom cells
wiring = FullyConnected(32)

# Model builder
def build_model(cell_type, input_shape=(4,), num_classes=5):
    inp = Input(shape=input_shape)
    x = Reshape((1, input_shape[0]))(inp)
    x = Conv1D(64, kernel_size=5, activation='relu', padding='same')(x)
    x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)

    if cell_type == "RNNCell":
        x = RNN(SimpleRNNCell(32), return_sequences=False)(x)
    elif cell_type == "LSTMCell":
        x = RNN(LSTMCell(32), return_sequences=False)(x)
    elif cell_type == "GRUCell":
        x = RNN(GRUCell(32), return_sequences=False)(x)
    elif cell_type == "LTCCell":
        x = RNN(LTCCell(wiring), return_sequences=False)(x)
    elif cell_type == "CfCCell":
        x = RNN(CfCCell(32), return_sequences=False)(x)
    elif cell_type == "ODELSTM":
        x = RNN(ODELSTM(32), return_sequences=False)(x)
    elif cell_type == "PhasedLSTM":
        x = RNN(PhasedLSTM(32), return_sequences=False)(x)
    elif cell_type == "GRUODE":
        x = RNN(GRUODE(32), return_sequences=False)(x)
    elif cell_type == "CTRNNCell":
        x = RNN(CTRNNCell(32, num_unfolds=5, method='euler'), return_sequences=False)(x)
    elif cell_type == "Attention":
        x = Attention()([x, x])
        x = Flatten()(x)
    elif cell_type == "MultiHeadAttention":
        x = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
        x = Flatten()(x)
    elif cell_type == "odeformer":
        x = ODEformer(hidden_dim=64, num_heads=16, ff_dim=64)(x)
        x = Flatten()(x)
    elif cell_type == "CTA":
        x = CTA(hidden_size=64)(x)
    elif cell_type == "NAC_Exact":
        x = NAC(d_model=64, num_heads=16, mode='exact',dt=1.0, return_sequences=False)(x)
    elif cell_type == "NAC_Euler":
        x = NAC(d_model=64, num_heads=16, mode='euler',dt=1.0, euler_steps=6, return_sequences=False)(x)
    elif cell_type == "NAC_Steady":
        x = NAC(d_model=64, num_heads=16, mode="steady",dt=1.0, return_sequences=False)(x)
    elif cell_type == "NAC_FC":
        x = NAC_FC(d_model=64, num_heads=16, mode="exact",dt=1.0, return_sequences=False)(x)
    elif cell_type == "NAC_PW":
        x = NAC_PW(d_model=64, num_heads=16, mode="exact",delta_t=1.0, return_sequences=False)(x)
    elif cell_type == "NAC_PWFC":
        x = NAC_PWFC(d_model=64, num_heads=16, mode="exact",delta_t=1.0, return_sequences=False)(x)
    else:
        raise ValueError(f"Unknown cell type: {cell_type}")

    x = Dense(32, activation="relu")(x)
    out = Dense(num_classes, activation="softmax")(x)
    return Model(inp, out)

# List of model types
model_types = [
    # "RNNCell", "LSTMCell", "GRUCell",
    # "GRUODE", "CTRNNCell", "PhasedLSTM",
    # "ODELSTM", "CfCCell", "LTCCell",
    # "MultiHeadAttention", "Attention", "odeformer",
    # "NAC_Exact", "NAC_Euler", "NAC_Steady",
    # "NAC_FC","NAC_PW","NAC_PWFC"
    "CTA"
]

# Callbacks
def get_callbacks(model_name):
    return [
        ModelCheckpoint(
            f"{weights_dir}/{model_name}.weights.h5",
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=0
        )
    ]

# Run k-fold CV
k_folds = 5
results = {}

for cell_type in model_types:
    model_name = f"{base_model_name}_{cell_type}"
    print(f"\nTraining {model_name} with {k_folds}-fold CV...")

    fold_accuracies = []

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y), 1):
        print(f"  Fold {fold}/{k_folds}")

        # Split data
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_ds = (
            tf.data.Dataset.from_tensor_slices((X_train, y_train))
            .shuffle(buffer_size=10000)
            .batch(20)
        )
        val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(20)

        # Build model
        model = build_model(cell_type, input_shape=(4,), num_classes=num_classes)
        model.compile(
            optimizer=AdamW(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        callbacks = get_callbacks(f"{model_name}_fold{fold}")

        # Train
        model.fit(train_ds, validation_data=val_ds, epochs=100,
                  callbacks=callbacks, verbose=0)

        # Load best weights and evaluate
        model.load_weights(f"{weights_dir}/{model_name}_fold{fold}.weights.h5")
        _, val_acc = model.evaluate(val_ds, verbose=0)

        fold_accuracies.append(val_acc * 100)  # percentage

    # Store CV stats
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)

    results[cell_type] = {"fold_accuracies": fold_accuracies,
                          "mean": mean_acc, "std": std_acc}

    print(f"{model_name} Fold Accuracies: {fold_accuracies}")
    print(f"{model_name} Mean Accuracy: {mean_acc:.2f}%, Std: {std_acc:.2f}%")

# Final summary
print("\n=== Final Model Results ===")
for cell_type, data in results.items():
    print(f"{base_model_name}_{cell_type}: "
          f"Mean={data['mean']:.2f}%, Std={data['std']:.2f}%")
