# main.py
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import numpy as np
import pandas as pd
from utils.preprocess import process_features
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

from tensorflow.keras.layers import (
    Input, Dense, Reshape, Conv1D, LSTMCell, GRUCell,
    RNN, SimpleRNNCell, MultiHeadAttention, Flatten, Attention
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from ncps.tf import LTCCell, CfCCell
from ncps.wirings import FullyConnected
from baseline_cells import CTRNNCell, ODELSTM, PhasedLSTM, GRUODE, ODEformer, CTA
from tensorflow.keras.optimizers import RMSprop
from neuronal_attention_circuit import NAC
from NAC_with_Pairwise import NAC_PW
from NAC_with_FC import NAC_FC
from NAC_with_PWFC import NAC_PWFC

base_model_name = 'Degradation_Estimation'
feature_dir = 'tf_features_pronostia'
weights_dir = 'model_weights'
stat_dir = 'statistics'


def score(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    error = y_pred - y_true

    mask_early = error < 0  # early prediction
    mask_late = error >= 0  # late prediction

    score_early = tf.reduce_sum(tf.exp(-error[mask_early] / 13) - 1)
    score_late = tf.reduce_sum(tf.exp(error[mask_late] / 10) - 1)

    return score_early + score_late

bearings = ['Bearing1_4', 'Bearing1_5', 'Bearing1_6', 'Bearing1_7']

# Load and preprocess data
dfs = [pd.read_csv(f'{feature_dir}/{bearing}_features.csv') for bearing in bearings]

# Process features
horizontal_data = [np.array(df['Horizontal'].apply(eval).tolist()) for df in dfs]
X_h = np.vstack([process_features(data) for data in horizontal_data])
vertical_data = [np.array(df['Vertical'].apply(eval).tolist()) for df in dfs]
X_v = np.vstack([process_features(data) for data in vertical_data])
vibration_features = np.concatenate((X_h, X_v), axis=-1)

# Get other features
t_data = np.concatenate([df['Time'].values.reshape(-1, 1) for df in dfs], axis=0)
T_data = np.concatenate([(df['Temperature'].values + 273.15).reshape(-1, 1) for df in dfs], axis=0)
y = np.concatenate([df['Degradation'].values.reshape(-1, 1) for df in dfs], axis=0)
RPM = np.concatenate([df['RPM'].values.reshape(-1, 1) for df in dfs], axis=0)
Load = np.concatenate([df['Load'].values.reshape(-1, 1) for df in dfs], axis=0)

# Combine features and normalize
X = np.concatenate([vibration_features, t_data, T_data], axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Wiring for custom cells
wiring = FullyConnected(32)

# Model builder
def build_model(cell_type, input_shape=(16,), num_classes=1):
    inp = Input(shape=input_shape)
    x = Reshape((1,input_shape[0]))(inp)
    x = Conv1D(32,kernel_size=3, activation='relu', padding='same')(x)
    x = Conv1D(16,kernel_size=2, activation='relu', padding='same')(x)

    if cell_type == "RNNCell":
        x = RNN(SimpleRNNCell(16), return_sequences=False)(x)
    elif cell_type == "LSTMCell":
        x = RNN(LSTMCell(16), return_sequences=False)(x)
    elif cell_type == "GRUCell":
        x = RNN(GRUCell(16), return_sequences=False)(x)
    elif cell_type == "LTCCell":
        x = RNN(LTCCell(wiring), return_sequences=False)(x)
    elif cell_type == "CfCCell":
        x = RNN(CfCCell(16), return_sequences=False)(x)
    elif cell_type == "ODELSTM":
        x = RNN(ODELSTM(16), return_sequences=False)(x)
    elif cell_type == "PhasedLSTM":
        x = RNN(PhasedLSTM(16), return_sequences=False)(x)
    elif cell_type == "GRUODE":
        x = RNN(GRUODE(16), return_sequences=False)(x)
    elif cell_type == "CTRNNCell":
        x = RNN(CTRNNCell(16, num_unfolds=5, method='euler'), return_sequences=False)(x)
    elif cell_type == "Attention":
        x = Attention()([x, x])
        x = Flatten()(x)
    elif cell_type == "MultiHeadAttention":
        x = MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
        x = Flatten()(x)
    elif cell_type == "odeformer":
        x = ODEformer(hidden_dim=16, num_heads=8, ff_dim=64)(x)
        x = Flatten()(x)
    elif cell_type == "CTA":
        x = CTA(hidden_size=16)(x)
    elif cell_type == "NAC_Exact":
        x = NAC(d_model=16, num_heads=8, mode='exact',delta_t=1.5,sparsity=0.7, topk=10, activation='relu', return_sequences=False)(x)
    elif cell_type == "NAC_Euler":
        x = NAC(d_model=16, num_heads=8, mode='euler',delta_t=1.5,sparsity=0.7, topk=10, activation='relu',euler_steps=6, return_sequences=False)(x)
    elif cell_type == "NAC_Steady":
        x = NAC(d_model=16, num_heads=8, mode="steady",delta_t=1.5,sparsity=0.7, topk=10, activation='relu',return_sequences=False)(x)
    elif cell_type == "NAC_FC":
        x = NAC_FC(d_model=16, num_heads=8, mode="exact",delta_t=1.5,sparsity=0.7, topk=10, activation='relu',return_sequences=False)(x)
    elif cell_type == "NAC_PW":
        x = NAC_PW(d_model=16, num_heads=8, mode="exact",delta_t=1.5, activation='relu',return_sequences=False)(x)
    elif cell_type == "NAC_PWFC":
        x = NAC_PWFC(d_model=16, num_heads=8, mode="exact",delta_t=1.0, activation='relu',return_sequences=False)(x)
    else:
        raise ValueError(f"Unknown cell type: {cell_type}")
    
    out = Dense(num_classes, activation='linear')(x)
    return Model(inp, out)

# List of model types
model_types = [
    # "RNNCell", "LSTMCell", "GRUCell",
    # "GRUODE", "CTRNNCell", "PhasedLSTM",
    # "ODELSTM", "CfCCell", "LTCCell",
    # "MultiHeadAttention", "Attention", "odeformer",
    # "NAC_Exact", "NAC_Euler", "NAC_Steady",
    # "NAC_FC","NAC_PWFC",
    # "NAC_PW",
    # 'NAC_PWFC',
    'CTA',
]


# Callbacks
def get_callbacks(model_name):
    return [
        ModelCheckpoint(
            f"{weights_dir}/{model_name}.weights.h5",
            monitor="val_loss",
            mode="min",
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

    fold_score = []

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled, y), 1):
        print(f"  Fold {fold}/{k_folds}")

        # Split data
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_ds = (
            tf.data.Dataset.from_tensor_slices((X_train, y_train))
            .shuffle(buffer_size=10000)
            .batch(64)
        )
        val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(64)

        # Build model
        model = build_model(cell_type, input_shape=(16,))
        model.compile(
            optimizer=RMSprop(learning_rate=0.001),
            loss="mse",
            metrics=[score]
        )

        callbacks = get_callbacks(f"{model_name}_fold{fold}")

        # Train
        model.fit(train_ds, validation_data=val_ds, epochs=50, verbose=0,callbacks=callbacks)

        # Load best weights and evaluate
        model.load_weights(f"{weights_dir}/{model_name}_fold{fold}.weights.h5")
        _, val_score = model.evaluate(val_ds, verbose=0)

        fold_score.append(val_score)  # percentage

    # Store CV stats
    mean_acc = np.mean(fold_score)
    std_acc = np.std(fold_score)

    results[cell_type] = {"fold_score": fold_score,
                          "mean": mean_acc, "std": std_acc}

    print(f"{model_name} Fold Score: {fold_score}")
    print(f"{model_name} Mean Score: {mean_acc:.4f}, Std: {std_acc:.4f}")

# Final summary
print("\n=== Final Model Results ===")
for cell_type, data in results.items():
    print(f"{base_model_name}_{cell_type}: "
          f"Mean={data['mean']:.4f}, Std={data['std']:.4f}")
