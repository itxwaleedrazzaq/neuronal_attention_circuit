import pandas as pd
import numpy as np

# Original headers
columns = [
    "sequence_name",
    "tag_id",
    "timestamp",
    "date",
    "x",
    "y",
    "z",
    "activity"
]

# Read the dataset
df = pd.read_csv("data/data.txt", names=columns)

# Drop unnecessary columns
df = df.drop(columns=["sequence_name", "tag_id", "date", "timestamp"])

# Create time axis based on sample index * 0.211 seconds
sampling_interval = 0.211  # seconds
df["t"] = np.arange(len(df)) * sampling_interval

# Encode activities as numeric classes (consistent mapping)
activity_classes = [
    "walking", "falling", "lying down", "lying",
    "sitting down", "sitting", "standing up from lying",
    "on all fours", "sitting on the ground",
    "standing up from sitting", "standing up from sitting on the ground"
]
activity_map = {label: idx for idx, label in enumerate(df["activity"].unique())}
df["activity_id"] = df["activity"].map(activity_map)

# Reorder columns for clarity
df = df[["t", "x", "y", "z", "activity", "activity_id"]]

# Save to CSV
df.to_csv("PAR/data.csv", index=False)

print("Training dataset created as 'training_dataset.csv'")
print("Activity mapping:", activity_map)
