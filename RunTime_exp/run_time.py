import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import pandas as pd
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, LSTMCell, GRUCell,
    RNN, SimpleRNNCell, MultiHeadAttention, Flatten, Attention
)
from tensorflow.keras.models import Model
from liquid_attention import LAN
from ncps.tf import LTCCell, CfCCell
from ncps.wirings import FullyConnected
from baseline_cells import CTRNNCell, ODELSTM, CTGRU, GRUD, PhasedLSTM, GRUODE, ODEformer
from neuronal_attention_circuit import NAC

# CONFIG
stat_dir = 'statistics'
sequence_length = 1024   # long sequence (>1000)
hidden_dim = 64
batch_size = 1
num_runs = 10
num_heads = 4
np.random.seed(42)

# Wiring for LTC and CfC
wiring = FullyConnected(hidden_dim)

# MODEL BUILDER
def build_model(cell_type, seq_len=sequence_length, hidden_dim=hidden_dim, num_heads=num_heads):
    inp = Input(shape=(seq_len, hidden_dim))

    if cell_type == "RNNCell":
        x = RNN(SimpleRNNCell(hidden_dim), return_sequences=False)(inp)
    elif cell_type == "LSTMCell":
        x = RNN(LSTMCell(hidden_dim), return_sequences=False)(inp)
    elif cell_type == "GRUCell":
        x = RNN(GRUCell(hidden_dim), return_sequences=False)(inp)
    elif cell_type == "LTCCell":
        x = RNN(LTCCell(wiring), return_sequences=False)(inp)
    elif cell_type == "CfCCell":
        x = RNN(CfCCell(hidden_dim), return_sequences=False)(inp)
    elif cell_type == "ODELSTM":
        x = RNN(ODELSTM(hidden_dim), return_sequences=False)(inp)
    elif cell_type == "CTGRU":
        x = RNN(CTGRU(hidden_dim), return_sequences=False)(inp)
    elif cell_type == "GRUD":
        x = RNN(GRUD(hidden_dim), return_sequences=False)(inp)
    elif cell_type == "PhasedLSTM":
        x = RNN(PhasedLSTM(hidden_dim), return_sequences=False)(inp)
    elif cell_type == "GRUODE":
        x = RNN(GRUODE(hidden_dim), return_sequences=False)(inp)
    elif cell_type == "CTRNNCell":
        x = RNN(CTRNNCell(hidden_dim, num_unfolds=5, method='euler'), return_sequences=False)(inp)
    elif cell_type == "Attention":
        x = Attention()([inp, inp])
        x = Flatten()(x)
    elif cell_type == "MultiHeadAttention":
        x = MultiHeadAttention(num_heads=num_heads, key_dim=hidden_dim)(inp, inp)
        x = Flatten()(x)
    elif cell_type == "LAN_Exact":
        x = LAN(d_model=hidden_dim, num_heads=num_heads, mode="exact", return_sequences=False)(inp)
    elif cell_type == "LAN_Euler":
        x = LAN(d_model=hidden_dim, num_heads=num_heads, mode="euler", euler_steps=20, return_sequences=False)(inp)
    elif cell_type == "LAN_Steady":
        x = LAN(d_model=hidden_dim, num_heads=num_heads, mode="steady", return_sequences=False)(inp)
    elif cell_type == "odeformer":
        x = ODEformer(hidden_dim=hidden_dim, num_heads=num_heads, ff_dim=hidden_dim)(inp)
        x = Flatten()(x)
    elif cell_type == "NAC":
        x = NAC(d_model=hidden_dim,num_heads=num_heads,delta_t=1.0,sparsity=0.5,return_sequences=False)(inp)
    else:
        raise ValueError(f"Unknown cell type: {cell_type}")

    return Model(inp, x)

# RUNTIME + MEMORY FUNCTION
def measure_runtime_and_memory(model, num_runs=10):
    dummy_input = np.random.randn(batch_size, sequence_length, hidden_dim).astype(np.float32)
    runtimes = []

    # warm-up
    _ = model(dummy_input)

    # Track memory (GPU or CPU)
    gpu_mem_usage = None
    if tf.config.list_physical_devices('GPU'):
        gpu = tf.config.list_physical_devices('GPU')[0]
        tf.config.experimental.reset_memory_stats(gpu)

    for _ in range(num_runs):
        start = time.time()
        _ = model(dummy_input)
        end = time.time()
        runtimes.append(end - start)

    # Compute runtime stats
    runtimes = np.array(runtimes)
    mean_rt = runtimes.mean()
    std_rt = runtimes.std()
    throughput = 1.0 / mean_rt

    # Memory usage (GPU)
    if tf.config.list_physical_devices('GPU'):
        mem_info = tf.config.experimental.get_memory_info('GPU:0')
        gpu_mem_usage = mem_info['peak'] / (1024 ** 2)  # MB
    else:
        import psutil
        process = psutil.Process(os.getpid())
        gpu_mem_usage = process.memory_info().rss / (1024 ** 2)  # MB (RAM)

    return mean_rt, std_rt, throughput, gpu_mem_usage

# MODEL TYPES
model_types = [
    "RNNCell", "LSTMCell", "GRUCell",
    "GRUODE", "CTRNNCell", "PhasedLSTM",
    "ODELSTM", "CfCCell", "LTCCell",
    "MultiHeadAttention", "Attention", "odeformer",
    "LAN_Exact", "LAN_Euler", "LAN_Steady",
    'NAC'
]

# MAIN LOOP
results = []

for cell_type in model_types:
    print(f"\nBenchmarking {cell_type}...")
    model = build_model(cell_type)
    mean_rt, std_rt, throughput, mem_usage = measure_runtime_and_memory(model, num_runs)

    results.append({
        "Model": cell_type,
        "Sequence Length (n)": sequence_length,
        "Hidden Dim (k)": hidden_dim,
        "Mean Runtime (s)": round(mean_rt, 4),
        "Std Dev (s)": round(std_rt, 4),
        "Throughput (seq/s)": round(throughput, 2),
        "Peak Memory (MB)": round(mem_usage, 2)
    })

# RESULTS TABLE
df = pd.DataFrame(results)
print("\n=== Runtime + Memory Benchmark Results ===")
print(df.to_string(index=False))
df.to_csv(f"{stat_dir}/runtime_memory_results.csv", index=False)
print("\nResults saved to runtime_memory_results.csv")
