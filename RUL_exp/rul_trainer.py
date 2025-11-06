import numpy as np
import pandas as pd
from utils.preprocess import process_features
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras._tf_keras.keras.layers import Input,Dense,Reshape,Conv1D
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.optimizers import AdamW
from neuronal_attention_circuit import NAC


MODEL_NAME = 'RUL_NAC'
FEATURE_DIR = 'tf_features_pronostia'
WEIGHTS_DIR = 'model_weights'
STATS_DIR = 'statistics'
bearings = ['Bearing1_4', 'Bearing1_5', 'Bearing1_6', 'Bearing1_7']


# Load and preprocess data
dfs = [pd.read_csv(f'{FEATURE_DIR}/{bearing}_features.csv') for bearing in bearings]

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
X = scaler.fit_transform(X)


#build LAN model
def build_model(input_shape=(16,)):
    inp = Input(shape=input_shape)
    x = Reshape((1,input_shape[0]))(inp)
    x = Conv1D(32,kernel_size=3, activation='relu', padding='same')(x)
    x = Conv1D(16,kernel_size=2, activation='relu', padding='same')(x)
    x = NAC(d_model=16,num_heads=4,delta_t=1.0,sparsity=0.5,activation='relu',return_sequences=False)(x)
    out = Dense(1, activation='linear')(x)
    return Model(inp, out)


model = build_model()
model.summary()
model.compile(AdamW(learning_rate=0.001),loss='mse', metrics=['mae'])


history = model.fit(X,y,epochs=20,validation_split=0.2)
model.evaluate(X,y)
model.save(f"{WEIGHTS_DIR}/{MODEL_NAME}.keras")
