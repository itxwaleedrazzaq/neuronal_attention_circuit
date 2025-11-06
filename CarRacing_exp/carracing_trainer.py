import pickle
import numpy as np
import tensorflow as tf
from keras._tf_keras.keras.layers import Input,Dense,Flatten,Dropout,TimeDistributed,Conv2D
from sklearn.model_selection import train_test_split
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.callbacks import ReduceLROnPlateau,ModelCheckpoint
from neuronal_attention_circuit import NAC

model_name = 'CarRacing_NAC'

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


X = np.expand_dims(np.array(X),axis=1)
y = np.array(y).astype(dtype='uint8')

num_classes = len(np.unique(y))

#split the data using sklearn
X_train, X_test_events, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32).shuffle(10000)
test_ds = tf.data.Dataset.from_tensor_slices((X_test_events, y_test)).batch(32)


def build_model(input_shape=(None,96,96,3)):
    inp = Input(shape=input_shape)
    x = TimeDistributed(Conv2D(10,(3,3),activation='relu',strides=2))(inp)
    x = TimeDistributed(Dropout(0.2))(x)
    x = TimeDistributed(Conv2D(20,(5,5),activation='relu',strides=2))(x)
    x = TimeDistributed(Dropout(0.2))(x)
    x = TimeDistributed(Conv2D(30,(5,5),activation='relu',strides=2))(x)
    x = TimeDistributed(Dropout(0.2))(x)
    x = TimeDistributed(Flatten())(x)
    x = NAC(d_model=64,num_heads=16,delta_t=1.5,sparsity=0.7, topk=10, return_sequences=False)(x)
    x = Dense(64, activation='relu')(x)
    out = Dense(num_classes, activation='softmax')(x)
    return Model(inp, out)



model = build_model()
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

rl = ReduceLROnPlateau(monitor='val_accuracy', factor=0.90, patience=2, min_lr=1e-10,mode='max')
checkpoint = ModelCheckpoint(f"{weights_dir}/{model_name}.weights.h5", monitor='val_accuracy',mode='max', save_best_only=True, save_weights_only=True)

model.fit(train_ds, validation_data=test_ds, epochs=100, callbacks=[rl,checkpoint])

# model.save(f"{weights_dir}/{model_name}.keras")

model.load_weights(f"{weights_dir}/{model_name}.weights.h5")
loss, acc = model.evaluate(test_ds)
print(f"Test Accuracy: {acc*100:.2f}%")
