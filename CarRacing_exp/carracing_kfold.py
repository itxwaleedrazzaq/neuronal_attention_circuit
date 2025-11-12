import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import pickle
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import (
    Input, Dense, LSTMCell, GRUCell, RNN, SimpleRNNCell, MultiHeadAttention, Flatten,
      Attention,TimeDistributed,Conv2D,Dense,Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import StratifiedKFold

from ncps.tf import LTCCell, CfCCell
from ncps.wirings import FullyConnected
from baseline_cells import CTRNNCell, ODELSTM, PhasedLSTM, GRUODE, ODEformer,CTA
from neuronal_attention_circuit import NAC
from NAC_with_Pairwise import NAC_PW
from NAC_with_FC import NAC_FC
from NAC_with_PWFC import NAC_PWFC


base_model_name = 'CarRacing'

feature_dir = 'tf_features'
weights_dir = 'model_weights'
stat_dir = 'statistics'



X = []
y = []
pickle_in = open('CarRacing/data/data.pickle','rb')
data = pickle.load(pickle_in)

for obs,actions in data:
    X.append(obs)
    y.append(actions)

X_events = np.expand_dims(np.array(X),axis=1)
y = np.array(y).astype(dtype='uint8')

num_classes = len(np.unique(y))


# Wiring
wiring = FullyConnected(64)

# ---- Model builder ----
def build_model(cell_type,input_shape=(None,96,96,3), num_classes=5):
    inp = Input(shape=input_shape)
    x = TimeDistributed(Conv2D(10,(3,3),activation='relu',strides=(2,2)))(inp)
    x = TimeDistributed(Dropout(0.2))(x)
    x = TimeDistributed(Conv2D(20,(5,5),activation='relu',strides=(2,2)))(x)
    x = TimeDistributed(Dropout(0.2))(x)
    x = TimeDistributed(Conv2D(30,(5,5),activation='relu',strides=(2,2)))(x)
    x = TimeDistributed(Dropout(0.2))(x)
    x = TimeDistributed(Flatten())(x)

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
        x = RNN(ODELSTM(64), return_sequences=False)(x)
    elif cell_type == "PhasedLSTM":
        x = RNN(PhasedLSTM(64), return_sequences=False)(x)
    elif cell_type == "GRUODE":
        x = RNN(GRUODE(64), return_sequences=False)(x)
    elif cell_type == "CTRNNCell":
        x = RNN(CTRNNCell(64, num_unfolds=5, method='euler'), return_sequences=False)(x)
    elif cell_type == "Attention":
        x = Attention()([x, x])
        x = tf.keras.layers.Lambda(lambda t: tf.squeeze(t, axis=1))(x)
    elif cell_type == "MultiHeadAttention":
        x = MultiHeadAttention(num_heads=8, key_dim=64)(x,x)
        x = tf.keras.layers.Lambda(lambda t: tf.squeeze(t, axis=1))(x)
    elif cell_type == "odeformer":
        x = ODEformer(hidden_dim=64, num_heads=8, ff_dim=64)(x)
        x = tf.keras.layers.Lambda(lambda t: tf.squeeze(t, axis=1))(x)
    elif cell_type == "CTA":
        x = CTA(hidden_size=64)(x)
    elif cell_type == "NAC_Exact":
        x = NAC(d_model=64, num_heads=16, mode='exact',dt=1.0,sparsity=0.7, topk=10, return_sequences=False)(x)
    elif cell_type == "NAC_Euler":
        x = NAC(d_model=64, num_heads=16, mode='exact',dt=1.0,sparsity=0.7, topk=10, euler_steps=5, return_sequences=False)(x)
    elif cell_type == "NAC_Steady":
        x = NAC(d_model=64, num_heads=16, mode="steady",dt=1.0,sparsity=0.7, topk=10, return_sequences=False)(x)
    elif cell_type == "NAC_FC":
        x = NAC_FC(d_model=64, num_heads=16, mode="exact",dt=1.0, return_sequences=False)(x)
    elif cell_type == "NAC_PW":
        x = NAC_PW(d_model=64, num_heads=16, mode="exact",dt=1.0, return_sequences=False)(x)
    elif cell_type == "NAC_PWFC":
        x = NAC_PWFC(d_model=64, num_heads=16, mode="exact",dt=1.0, return_sequences=False)(x)
    else:
        raise ValueError(f"Unknown cell type: {cell_type}")

    x = Dense(64, activation="relu")(x)
    out = Dense(num_classes, activation="softmax")(x)
    return Model(inp, out)

# ---- Model types ----
model_types = [
    # "RNNCell", "LSTMCell", "GRUCell",
    # "GRUODE", "CTRNNCell", "PhasedLSTM",
    # "ODELSTM", "CfCCell", "LTCCell",
    # "MultiHeadAttention", "Attention", "odeformer", 
    "CTA"
    # "NAC_Exact", "NAC_Euler", "NAC_Steady",
    # "NAC_FC","NAC_PWFC","NAC_PW"
]

# ---- Callbacks ----
def get_callbacks(model_name):
    return [
        ReduceLROnPlateau(monitor='val_accuracy',
                            factor=0.90,
                            patience=2, 
                            min_lr=1e-10,
                            mode='max'),
        ModelCheckpoint(
            f"{weights_dir}/{model_name}.weights.h5",
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=0
        )
    ]

# ---- K-Fold CV ----
k_folds = 5
results = {}

skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

for cell_type in model_types:
    model_name = f"{base_model_name}_{cell_type}"
    print(f"\nTraining {model_name} with {k_folds}-fold CV...")

    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_events, y), 1):
        print(f"  Fold {fold}/{k_folds}")

        # Split data
        X_train, X_val = X_events[train_idx], X_events[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(10000).batch(32)
        val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(32)

        # Build model
        model = build_model(cell_type, input_shape=(None,96,96,3), num_classes=5)
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        callbacks = get_callbacks(f"{model_name}_fold{fold}")

        # Train
        model.fit(train_ds, validation_data=val_ds, epochs=70,callbacks=callbacks, verbose=0)

        # Load best weights and evaluate
        model.load_weights(f"{weights_dir}/{model_name}_fold{fold}.weights.h5")
        _, val_acc = model.evaluate(val_ds, verbose=0)

        fold_accuracies.append(val_acc * 100)

    # Store CV results
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)

    results[cell_type] = {"fold_accuracies": fold_accuracies,
                          "mean": mean_acc, "std": std_acc}

    print(f"{model_name} Fold Accuracies: {fold_accuracies}")
    print(f"{model_name} Mean Accuracy: {mean_acc:.2f}%, Std: {std_acc:.2f}%")

# ---- Final summary ----
print("\n=== Final Model Results ===")
for cell_type, data in results.items():
    print(f"{base_model_name}_{cell_type}: "
          f"Mean={data['mean']:.2f}%, Std={data['std']:.2f}%")
