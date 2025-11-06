# main_fixed.py
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import numpy as np
import tensorflow as tf
import pandas as pd
from utils.preprocess import process_features
from sklearn.model_selection import KFold
from tensorflow.keras.layers import (
    Input, Dense, Reshape, Conv2D, Dropout, Lambda,
    RNN, SimpleRNNCell, LSTMCell, GRUCell, Flatten,
    Attention, MultiHeadAttention, Activation
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import AdamW
from utils import batch_generator, INPUT_SHAPE
from liquid_attention import LAN
from ncps.tf import LTCCell, CfCCell
from ncps.wirings import FullyConnected
from baseline_cells import CTRNNCell, ODELSTM, PhasedLSTM, GRUODE, ODEformer
from neuronal_attention_circuit import NAC

# Paths and hyperparameters
base_model_name = 'Udacity_Simulator'
data_dir = 'data'
weights_dir = 'model_weights'
nb_epoch = 10
batch_size = 40
learning_rate = 1.0e-4
keep_prob = 0.5

# Load CSV
data_df = pd.read_csv(os.path.join(data_dir, 'driving_log.csv'))
X = data_df[['center', 'left', 'right']].values
y = data_df['steering'].values

# Wiring for custom cells
wiring = FullyConnected(32)

# Model builder
def build_model(cell_type, input_shape=INPUT_SHAPE, num_classes=1):
    inp = Input(shape=input_shape)
    x = Lambda(lambda x: x / 127.5 - 1.0)(inp)
    x = Conv2D(24, (5, 5), activation='elu', strides=(2, 2))(x)
    x = Conv2D(36, (5, 5), activation='elu', strides=(2, 2))(x)
    x = Conv2D(48, (5, 5), activation='elu', strides=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='elu')(x)
    x = Conv2D(64, (3, 3), activation='elu')(x)
    x = Dropout(0.5)(x)
    x = Reshape((-1, x.shape[1]*x.shape[2]))(x)

    if cell_type == "RNNCell":
        x = RNN(SimpleRNNCell(64), return_sequences=False)(x)
    elif cell_type == "LSTMCell":
        x = RNN(LSTMCell(64), return_sequences=False)(x)
    elif cell_type == "GRUCell":
        x = RNN(GRUCell(64), return_sequences=False)(x)
    elif cell_type == "LTCCell":
        x = RNN(LTCCell(wiring), return_sequences=False)(x)
    elif cell_type == "CfCCell":
        x = RNN(CfCCell(64), return_sequences=False)(x)
    elif cell_type == "ODELSTM":
        x = RNN(ODELSTM(16), return_sequences=False)(x)
    elif cell_type == "PhasedLSTM":
        x = RNN(PhasedLSTM(64), return_sequences=False)(x)
    elif cell_type == "GRUODE":
        x = RNN(GRUODE(64), return_sequences=False)(x)
    elif cell_type == "CTRNNCell":
        x = RNN(CTRNNCell(64, num_unfolds=5, method='euler'), return_sequences=False)(x)
    elif cell_type == "Attention":
        x = Attention()([x, x])
        x = Flatten()(x)
    elif cell_type == "MultiHeadAttention":
        x = MultiHeadAttention(num_heads=16, key_dim=64)(x, x)
        x = Flatten()(x)
    elif cell_type == "odeformer":
        x = ODEformer(hidden_dim=64, num_heads=16, ff_dim=64)(x)
        x = Flatten()(x)
    elif cell_type == "LAN_Exact":
        x = LAN(d_model=64, num_heads=16, mode='exact', return_sequences=False)(x)
    elif cell_type == "LAN_Euler":
        x = LAN(d_model=64, num_heads=16, mode='euler', euler_steps=10, delta_t=0.1, return_sequences=False)(x)
    elif cell_type == "LAN_Steady":
        x = LAN(d_model=64, num_heads=16, mode='steady', return_sequences=False)(x)
    elif cell_type == "NAC":
        x = NAC(d_model=64, num_heads=16, sparsity=0.5, delta_t=1.0, return_sequences=False)(x)
        
    else:
        raise ValueError(f"Unknown cell type: {cell_type}")
    x = Activation('elu')(x)
    out = Dense(num_classes, activation='linear')(x)
    return Model(inp, out)

# Callbacks
def get_callbacks(model_name):
    return [
        ModelCheckpoint(
            f"{weights_dir}/{model_name}.weights.h5",
            monitor="val_loss",
            mode="auto",
            save_best_only=True,
            save_weights_only=True,
            overwrite=True,
            verbose=0
        )
    ]

# Model types
model_types = [
    # "RNNCell", "LSTMCell", "GRUCell", 'GRUODE', 'CTRNNCell', 'PhasedLSTM',
    # 'ODELSTM', "CfCCell", "LTCCell", 
    # "MultiHeadAttention", 'Attention',
    # "LAN_Exact", "LAN_Euler", "LAN_Steady", "odeformer",
    "NAC"
]


# K-fold CV
k_folds = 5
results = {}

for cell_type in model_types:
    model_name = f"{base_model_name}_{cell_type}"
    print(f"\nTraining {model_name} with {k_folds}-fold CV...")

    fold_mse = []
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
        print(f"  Fold {fold}/{k_folds}")

        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Build model
        model = build_model(cell_type, input_shape=INPUT_SHAPE)
        model.compile(
            optimizer=AdamW(learning_rate=0.001),
            loss="mse",
            metrics=["mae"]
        )

        callbacks = get_callbacks(f"{model_name}_fold{fold}")

        # Train
        model.fit(
            batch_generator(data_dir, X_train, y_train, batch_size, True),
            steps_per_epoch=len(X_train)//batch_size,
            epochs=10,
            validation_data=batch_generator(data_dir, X_val, y_val, batch_size, False),
            validation_steps=len(X_val)//batch_size,
            callbacks=callbacks,
            verbose=0,
        )

        # Load best weights and evaluate
        model.load_weights(f"{weights_dir}/{model_name}_fold{fold}.weights.h5")
        mse, _ = model.evaluate(
            batch_generator(data_dir, X_val, y_val, batch_size, False),
            steps=len(X_val)//batch_size,
            verbose=0
        )
        fold_mse.append(mse)

    # Store CV stats
    mean_acc = np.mean(fold_mse)
    std_acc = np.std(fold_mse)
    results[cell_type] = {"fold_mse": fold_mse, "mean": mean_acc, "std": std_acc}

    print(f"{model_name} Fold MSE: {fold_mse}")
    print(f"{model_name} Mean MSE: {mean_acc:.4f}, Std: {std_acc:.4f}")

# Final summary
print("\n=== Final Model Results ===")
for cell_type, data in results.items():
    print(f"{base_model_name}_{cell_type}: Mean={data['mean']:.4f}, Std={data['std']:.4f}")
