import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from neuronal_attention_circuit import NAC

# Generate training data
def generate_data(n_samples=2000):
    x = np.random.uniform(-2 * np.pi, 2 * np.pi, size=(n_samples, 1))
    y = np.sin(x)
    return x.astype(np.float32), y.astype(np.float32)

x_train, y_train = generate_data(4000)
x_test, y_test = generate_data(1000)

x_plot = np.linspace(-2*np.pi, 2*np.pi, 500).astype(np.float32)
y_true = np.sin(x_plot)

# Store final predictions
predictions_final = {}

# Training settings
epochs = 150
batch_size = 64

inputs = tf.keras.Input(shape=(1, 1))
x = NAC(
    d_model=64,
    num_heads=16,
    delta_t=1.0,
    sparsity=0.5,
    activation='sigmoid',
    return_sequences=False
)(inputs)
outputs = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="mse",
    metrics=["mae"]
)

model.summary()

model.fit(
    x_train[:, None, :],
    y_train,
    validation_split=0.2,
    epochs=epochs,
    batch_size=batch_size,
    verbose=1,
)

    # Save final predictions in memory
y_pred = model.predict(x_plot[:, None, None], verbose=0)

# Plot final results
plt.figure(figsize=(8, 4))
plt.plot(x_plot, y_true, label="sin(x)", color="black")

colors = {"euler": "red", "exact": "blue", "steady": "green"}
plt.plot(x_plot, y_pred, label=f"NAC")

plt.legend()
plt.title("Final Predictions after Training")
plt.xlabel("x")
plt.ylabel("y")
plt.ylim(-1.5, 1.5)
plt.show()

