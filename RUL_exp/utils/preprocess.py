import numpy as np

def correct_dims(img):
    return np.expand_dims(img,axis=0)

def correct_dims3(img):
    return np.expand_dims(img,axis=-1)

def correct_dims2(img):
    return np.expand_dims(np.expand_dims(np.array(img), axis=-1),axis=0)

def process_features(data):
    energy = np.array([i[0] for i in data])
    dominant_frequency = np.array([i[1] for i in data])
    entropy = np.array([i[2] for i in data])
    kurtosis = np.array([i[3] for i in data])
    skewness = np.array([i[4] for i in data])
    mean = np.array([i[5] for i in data])
    std = np.array([i[6] for i in data])
    log_normalized_energy = np.log10(energy + 1e-8)
    return np.column_stack((log_normalized_energy, dominant_frequency, entropy, kurtosis, skewness, mean, std))