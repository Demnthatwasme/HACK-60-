import pandas as pd
import os

print("--- Gesture Dataset Merger ---")

current_dir = os.path.dirname(__file__)
main_file = os.path.join(current_dir, 'gesture_dataset.csv')
new_file = os.path.join(current_dir, 'gesture_dataset_2.csv')

# 1. Load both datasets
print("Loading datasets...")
df_main = pd.read_csv(main_file)
df_new = pd.read_csv(new_file)

print(f"Main dataset: {len(df_main)} frames")
print(f"New dataset: {len(df_new)} frames")

# 2. Fix the label in the new dataset
print("\nChanging GRAB to DAB in the new dataset...")
# This specifically targets the 'label' column and replaces exact matches
df_new['label'] = df_new['label'].replace("GRAB","DAB")

# 3. Merge them together
print("\nMerging datasets...")
# ignore_index=True ensures the row numbers flow continuously 
merged_df = pd.concat([df_main, df_new], ignore_index=True)
print(f"Total merged frames: {len(merged_df)}")

# 4. Save over the main dataset
merged_df.to_csv(main_file, index=False)
print(f"\nDone! Overwrote '{main_file}' with the combined data.")