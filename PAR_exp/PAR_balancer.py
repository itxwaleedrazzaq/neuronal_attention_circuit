import pandas as pd
import numpy as np

# 1. columns
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

# 2. Read raw TXT dataset
df = pd.read_csv("PAR/data.txt", names=columns)

# 3. Extract person from sequence_name
df["person"] = df["sequence_name"].str[0]  # 'A', 'B', etc.

# 4. Drop unnecessary columns
df = df.drop(columns=["sequence_name", "tag_id", "date", "timestamp"])

# 5. Map activities to numeric IDs
activity_classes = [
    "walking", "falling", "lying down", "lying",
    "sitting down", "sitting", "standing up from lying",
    "on all fours", "sitting on the ground",
    "standing up from sitting", "standing up from sitting on the ground"
]
activity_map = {label: idx for idx, label in enumerate(activity_classes)}
df["activity_id"] = df["activity"].map(activity_map)

# 6. Separate per person, downsample to balance, calculate time, and save
sampling_interval = 0.211  # seconds per sample
persons = df["person"].unique()

for p in persons:
    df_person = df[df["person"] == p].copy()
    
    # Reset index and calculate time_seconds per person
    df_person = df_person.reset_index(drop=True)
    df_person["time_seconds"] = np.arange(len(df_person)) * sampling_interval
    
    # Find the minimum number of samples across activities
    min_size = df_person["activity_id"].value_counts().min()
    
    # Downsample each activity to min_size (keep original order)
    balanced_groups = []
    for activity_id, group in df_person.groupby("activity_id"):
        balanced_groups.append(group.iloc[:min_size])
    
    # Concatenate groups in original order
    df_balanced = pd.concat(balanced_groups, ignore_index=True)
    
    # Save CSV
    filename = f"PAR/balanced_person_{p}.csv"
    df_balanced.to_csv(filename, index=False)
    print(f"Saved balanced dataset for person {p} as '{filename}'")
