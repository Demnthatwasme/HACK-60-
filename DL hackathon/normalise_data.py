import pandas as pd
import os
import numpy as np

print("1. Loading dataset...")
current_dir = os.path.dirname(__file__)
csv_path = os.path.join(current_dir, 'gesture_dataset.csv')
df = pd.read_csv(csv_path)

print("2. Normalizing relative to wrist and applying Rigid Bone Scaling...")
# 1. Subtract the wrist position
wrist_x = df['x0'].copy()
wrist_y = df['y0'].copy()
wrist_z = df['z0'].copy()

for i in range(21):
    df[f'x{i}'] = df[f'x{i}'] - wrist_x
    df[f'y{i}'] = df[f'y{i}'] - wrist_y
    df[f'z{i}'] = df[f'z{i}'] - wrist_z

# 2. Rigid Bone Scale Invariance (Wrist to Middle Knuckle)
# Calculate the 3D Euclidean distance of joint 9 from the wrist
bone_lengths = np.sqrt(df['x9']**2 + df['y9']**2 + df['z9']**2)

# CLAMP: Prevent division by near-zero if the hand is pointing directly at the camera
bone_lengths = np.clip(bone_lengths, 0.05, None)

# Divide all coordinates by this fixed bone length
coord_cols = [col for col in df.columns if col != 'label']
for col in coord_cols:
    df[col] = df[col] / bone_lengths

df.to_csv(csv_path, index=False)
print("Done! Dataset perfectly normalized using Rigid Bone Scaling.")