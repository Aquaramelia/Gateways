import torch
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

from model_references import EmbeddingToParams, target_columns
from load_dataset import get_training_data, get_combined_df

X_train, X_test, y_train, y_test = get_training_data()
combined_df = get_combined_df()

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train) 
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)

# Create model
model = EmbeddingToParams()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.1)

# Check if the training can be done on the GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("✅ GPU detected! Training will use CUDA.")
else:
    device = torch.device('cpu')
    print("⚠️ GPU not detected. Training will use CPU.")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

# Training loop
num_epochs = 500
batch_size = 16

# Add early stopping
best_val_loss = float('inf')
patience = 30
patience_counter = 0

for epoch in range(num_epochs):
    model.train()
    
    # Shuffle training data
    perm = torch.randperm(X_train.size(0))
    X_train_shuffled = X_train[perm]
    y_train_shuffled = y_train[perm]
    
    total_loss = 0
    for i in range(0, len(X_train), batch_size):
        batch_X = X_train_shuffled[i:i+batch_size]
        batch_y = y_train_shuffled[i:i+batch_size]
        
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    # Validation
    if (epoch + 1) % 10 == 0: # Check every 10 epochs instead of 50
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test)
            val_loss = criterion(val_outputs, y_test)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {total_loss/len(X_train):.4f}, '
              f'Val Loss: {val_loss.item():.4f}')
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'Model/embedding_to_params_best.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# Save the model
torch.save(model.state_dict(), 'Model/embedding_to_params.pth')