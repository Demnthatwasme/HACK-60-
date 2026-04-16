import pandas as pd
import os

print("1. Loading dataset...")
current_dir = os.path.dirname(__file__)
csv_path = os.path.join(current_dir, 'gesture_dataset.csv')
df = pd.read_csv(csv_path)

print("2. Normalizing coordinates relative to the wrist...")
# Store the absolute wrist coordinates for each row
wrist_x = df['x0'].copy()
wrist_y = df['y0'].copy()
wrist_z = df['z0'].copy()

# Subtract the wrist position from all 21 joints
for i in range(21):
    df[f'x{i}'] = df[f'x{i}'] - wrist_x
    df[f'y{i}'] = df[f'y{i}'] - wrist_y
    df[f'z{i}'] = df[f'z{i}'] - wrist_z

df.to_csv(csv_path, index=False)
print("Done! Dataset normalized perfectly.")