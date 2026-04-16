import pandas as pd
import os

print("1. Loading dirty dataset...")
current_dir = os.path.dirname(__file__)
csv_path = os.path.join(current_dir, 'gesture_dataset.csv')

# Load the CSV, forcing Pandas to ignore the mixed-type warning for now
df = pd.read_csv(csv_path, low_memory=False)

print(f"Original row count: {len(df)}")

# 2. Destroy the rogue header rows
# This keeps only the rows where the label column is NOT the word 'label'
df = df[df.iloc[:, 0] != 'label']

# 3. Force all the coordinate columns (columns 1 through 63) to be strictly numbers
for col in df.columns[1:]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 4. Drop any rows that got completely corrupted (NaNs)
df = df.dropna()

print(f"Cleaned row count: {len(df)}")

# 5. Save the perfectly clean data back over the old file
df.to_csv(csv_path, index=False)
print("Data scrubbed and saved! Ready for PyTorch.")