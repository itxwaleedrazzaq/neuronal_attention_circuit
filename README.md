# Neuronal Attention Circuit (NAC)
---
The repository contains the code of the Neuronal Attention Circuit developed at the Networked Intelligent Control (NIC) Lab at the University of Science and Technology of China (USTC).


## LAN Model Usage Example

```python
import tensorflow as tf
from neuronal_attention_circuit import NAC

# Create the model
inputs = tf.keras.Input(shape=(1, 1))
x = NAC(
    d_model=64,         # Dimensionality of the model (feature size per token)
    num_heads=16,       # Number of attention heads
    delta_t=1.0,        # Time-step for integration
    sparsity=0.5,       # Controls how sparse the attention or activations are
    topk=8,             # Limits attention to top-k elements (for efficiency/sparsity)
    tau_epsilon=1e-6,   # Small value for numerical stability
    use_bias=True,      # Whether to include bias terms in linear layers
    dropout=0.1,        # Dropout rate for regularization
    activation='sigmoid',   # Activation function used inside the layer
    return_sequences=False  # Whether to return the full sequence or just the last output
)(inputs)

outputs = tf.keras.layers.Dense(1)(attn)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)
```


## Experiments

### 1. Universal Approximation Verification

Demonstration and verification of LAN’s universal approximation capability.

![Universal Approximation Demo](plots/uat.gif)

---

### 2. Event-based MNIST

Training and evaluation on the event-based MNIST dataset.

* Code available in: `mnist_exp/`

```bash
python mnist_trainer.py
```

---

### 3. Person Activity Recognition (PAR)

Activity recognition experiment implementation.

* Code available in: `PAR_exp/`

```bash
python PAR_trainer.py
```

---

### 4. Lane Keeping for Autonomous Vehicles

**a) CarRacing (`CarRacing_exp/`)**

```bash
python drive.py
```
[Watch the demo on YouTube](https://youtu.be/PAclVXbzsms)


**b) Udacity Simulator (`Udacity_exp/`)**

1. Run the simulator in Autonomous mode.
2. Execute:

```bash
python drive.py model_weights/Udacity_LAN.keras
```
[Watch the demo on YouTube](https://youtu.be/tKfO55TwN0M)

---

### 5. Remaining Useful Life (RUL) Estimation

Dataset and code for RUL estimation experiments.

* Code and data available in: `RUL_exp/`

```bash
# Example: how to run
python rul_trainer.py
```


### 6. Seq2Seq Time-Series Imputation
Code for LAN imputation experiment.

* Code and data available in: `Impute_exp/`
```bash
# Example: how to run
python impute.py
```

### 7. Run Time Experiment
Code for LAN run time experiment.

* Code and data available in: `RunTime_exp/`
```bash
# Example: how to run
python run_time.py
```

### 8. Ablation Experiments
Code for LAN ablation experiments.

* Code and data available in: `ablation_exp/`

```bash
# Example: how to run
python ablation.py
```

---

## License

[Specify your license here, e.g., MIT License]
