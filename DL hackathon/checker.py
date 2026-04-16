import pandas as pd

print("--- Dataset Health Check ---\n")

try:
    # Load the dataset
    df = pd.read_csv('gesture_dataset.csv')
    
    # 1. Total Frames
    print(f"Total Frames Captured: {len(df)}")
    
    # 2. Column Check
    col_count = len(df.columns)
    print(f"Total Columns: {col_count} (Expected: 64)")
    if col_count != 64:
        print("❌ ERROR: Your columns don't match the 1 label + 63 coordinates format.")
    else:
        print("✅ Column count is perfect.")
    
    # 3. Class Balance
    print("\n--- Class Distribution ---")
    dist = df['label'].value_counts()
    print(dist)
    
    # Check for heavy imbalance (if the smallest class is less than half the largest)
    if dist.min() < (dist.max() * 0.5):
        print("⚠️ WARNING: Your dataset is heavily imbalanced. Consider recording more of the lower classes.")
    else:
        print("✅ Dataset is well-balanced.")
        
    # 4. Missing Values
    missing_values = df.isnull().sum().sum()
    print(f"\n--- Missing Values: {missing_values} ---")
    if missing_values > 0:
        print("❌ ERROR: You have missing/blank data! You must drop these rows before training.")
        # Uncomment the next line to automatically fix it:
        # df.dropna(inplace=True); df.to_csv('gesture_dataset.csv', index=False)
    else:
        print("✅ Data is clean (No NaNs).")

except Exception as e:
    print(f"Could not read the CSV. Error: {e}")