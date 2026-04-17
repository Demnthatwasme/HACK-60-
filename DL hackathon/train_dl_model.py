import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

import os

print("1. Loading and preparing dataset...")
# Automatically find the CSV in the exact same folder as this python script
current_dir = os.path.dirname(__file__)
csv_path = os.path.join(current_dir, 'gesture_dataset.csv')

df = pd.read_csv(csv_path)
# Separate labels and features
X = df.drop('label', axis=1).values # Convert to numpy array
y_labels = df['label'].values

# Deep Learning needs numbers, not text labels. Encode "STOP", "FORWARD", etc., to 0, 1, 2...
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y_labels)

# Save the encoder classes so we know which number is which gesture later
np.save(os.path.join(current_dir, 'classes.npy'), encoder.classes_)

# Split the data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Convert to PyTorch Tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

# Create a PyTorch DataLoader
class GestureDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(GestureDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)

print("2. Building the Deep Learning Model...")
# Define a simple, fast Feedforward Neural Network (MLP)
class GestureNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(GestureNet, self).__init__()
        # 63 inputs (21 joints * 3 coordinates) -> Hidden Layers -> 5 Outputs
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2), # Prevents overfitting on your "small" dataset
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.network(x)

# Initialize model, loss function, and optimizer
num_classes = len(encoder.classes_)
model = GestureNet(input_size=63, num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("3. Training the Network with Augmentation & Early Stopping...")
epochs = 90 # We can train longer now because early stopping protects us
best_loss = float('inf')
best_model_state = None

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    
    for inputs, labels in train_loader:
        optimizer.zero_grad() 
        
        # --- DATA AUGMENTATION (JITTER) ---
        # Add tiny random noise to coordinates so the model learns the "idea" of the shape
        noise = torch.randn_like(inputs) * 0.015 
        noisy_inputs = inputs + noise
        
        outputs = model(noisy_inputs) 
        loss = criterion(outputs, labels) 
        loss.backward() 
        optimizer.step() 
        running_loss += loss.item()
    
    avg_loss = running_loss / len(train_loader)
    
    # --- EARLY STOPPING / BEST WEIGHT SAVING ---
    if avg_loss < best_loss:
        best_loss = avg_loss
        best_model_state = model.state_dict().copy() # Save a copy of the best weights
        
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f} (Best: {best_loss:.4f})")

print("5. Saving BEST Model...")
# Load the best weights back into the model before saving
model.load_state_dict(best_model_state)
torch.save(model.state_dict(), os.path.join(current_dir, 'gesture_dl_model.pth'))
print("Done! Smartest weights saved as 'gesture_dl_model.pth'")