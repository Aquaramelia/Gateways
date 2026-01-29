import torch
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

print("SIMPLIFIED TRAINING SCRIPT - FOCUSING ON PREDICTABLE PARAMETERS \n")

# TODO: Update dataset references

# Load XANEW filtered words
xanew = pd.read_csv('Data/xanew_filtered.csv')

# Load your custom ratings
custom = pd.read_csv('Data/semantic_parameters_dataset.csv')

# Load the word embeddings
word_embeddings = pd.read_csv('Data/word_embeddings.csv')

# Convert stringified lists back to numpy arrays
word_embeddings['embedding'] = word_embeddings['embedding'].apply(
    lambda x: np.fromstring(x.strip("[]"), sep=' ', dtype=np.float32)
)

# Merge datasets
combined_xanew_custom = pd.merge(xanew, custom, left_on='word', right_on='word', how='inner')
combined_df = pd.merge(combined_xanew_custom, word_embeddings, left_on='word', right_on='word', how='inner')

print(f"Combined dataset has {len(combined_df)} words")

# Based on our results, these work well:
predictable_params = [
    'arousal_norm',
    'valence_norm',       
    'dominance_norm',   
    'thickness',        
    'thickness_decay',  
    'upward_bias'      
]

# They showed bad R² scores and high MAE:
excluded_params = [
    'segment_length', 
    'angle',        
    'arch_curve_angle'
]

# Extract features (X) and targets (y)
X = np.stack(combined_df['embedding'].values)
y = combined_df[predictable_params].values

# Split into train/test
X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
    X, y, range(len(X)), test_size=0.2, random_state=42
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Normalize the data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train) 
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)

# Create SIMPLER model for fewer outputs
class SimplifiedModel(nn.Module):
    def __init__(self, input_size=384, output_size=6):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

model = SimplifiedModel(input_size=384, output_size=len(predictable_params))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

model.to(device)
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

# Training loop with early stopping
num_epochs = 500
batch_size = 16
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
        
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    # Validation every 10 epochs
    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test)
            val_loss = criterion(val_outputs, y_test)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {total_loss/len(X_train):.4f}, '
              f'Val Loss: {val_loss.item():.4f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'Model/simplified_model_best.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"\n Early stopping at epoch {epoch+1}")
            break

# Load best model
model.load_state_dict(torch.load('Model/simplified_model_best.pth'))

# Evaluate in original scale
print("\n FINAL EVALUATION (Original Scale)")

model.eval()
with torch.no_grad():
    val_outputs = model(X_test)
    val_outputs_original = scaler_y.inverse_transform(val_outputs.cpu().numpy())
    y_test_original = scaler_y.inverse_transform(y_test.cpu().numpy())
    
    mae_per_param = np.abs(val_outputs_original - y_test_original).mean(axis=0)
    
    print("\nMAE per parameter:")
    for i, col in enumerate(predictable_params):
        print(f'  {col:20s}: {mae_per_param[i]:.4f}')
    
    print(f"\nAverage MAE: {mae_per_param.mean():.4f}")

# Calculate R² score
from sklearn.metrics import r2_score
r2 = r2_score(y_test_original, val_outputs_original)
print(f"R² Score: {r2:.4f}")