"""
Ensemble Model Training - Train multiple models and average predictions
"""
import torch
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import joblib
from load_dataset import return_combined_df
from model_references import target_columns, SimplifiedModel

print("\n ENSEMBLE TRAINING \n")

combined_df = return_combined_df()

X = np.stack(combined_df['embedding'].values)
y = combined_df[target_columns].values

# Train multiple models with different random seeds
NUM_MODELS = 5
models = []
scalers_X = []
scalers_y = []
test_predictions = []

for model_idx in range(NUM_MODELS):
    print(f"\n Training Model {model_idx + 1}/{NUM_MODELS} (seed={42 + model_idx}) \n")
    
    # Use different random seed for each model
    random_seed = 42 + model_idx
    
    # Split with this seed
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, range(len(X)), test_size=0.2, random_state=random_seed
    )
    
    # Normalize
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train_scaled)
    y_train_t = torch.FloatTensor(y_train_scaled)
    X_test_t = torch.FloatTensor(X_test_scaled)
    y_test_t = torch.FloatTensor(y_test_scaled)
    
    # Create model
    model = SimplifiedModel(input_size=384, output_size=len(target_columns))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    X_train_t = X_train_t.to(device)
    y_train_t = y_train_t.to(device)
    X_test_t = X_test_t.to(device)
    y_test_t = y_test_t.to(device)
    
    # Training with early stopping
    best_val_loss = float('inf')
    patience = 30
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(500):
        model.train()
        
        # Shuffle
        perm = torch.randperm(X_train_t.size(0))
        X_shuffled = X_train_t[perm]
        y_shuffled = y_train_t[perm]
        
        total_loss = 0
        batch_size = 16
        for i in range(0, len(X_train_t), batch_size):
            batch_X = X_shuffled[i:i+batch_size]
            batch_y = y_shuffled[i:i+batch_size]
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation every 50 epochs
        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test_t)
                val_loss = criterion(val_outputs, y_test_t)
            
            print(f'  Epoch [{epoch+1}/500], Train Loss: {total_loss/len(X_train_t):.4f}, Val Loss: {val_loss.item():.4f}')
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        test_pred_scaled = model(X_test_t).cpu().numpy()
        test_pred = scaler_y.inverse_transform(test_pred_scaled)
    
    # Store for ensemble
    models.append(model)
    scalers_X.append(scaler_X)
    scalers_y.append(scaler_y)
    test_predictions.append(test_pred)
    
    # Individual model performance
    mae = np.abs(test_pred - y_test).mean(axis=0)
    print(f"\n  Model {model_idx + 1} MAE: {mae.mean():.4f}")

# Ensemble predictions (average of all models)
print("\n ENSEMBLE RESULTS (Average of all models) \n")

# For ensemble, use the first split for consistency
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Get predictions from all models
ensemble_predictions = []
for i in range(NUM_MODELS):
    X_test_scaled = scalers_X[i].transform(X_test)
    X_test_t = torch.FloatTensor(X_test_scaled).to(device)
    
    models[i].eval()
    with torch.no_grad():
        pred_scaled = models[i](X_test_t).cpu().numpy()
        pred = scalers_y[i].inverse_transform(pred_scaled)
    
    ensemble_predictions.append(pred)

# Average predictions
ensemble_pred = np.mean(ensemble_predictions, axis=0)

# Calculate ensemble MAE
ensemble_mae = np.abs(ensemble_pred - y_test).mean(axis=0)

print("\nEnsemble MAE per parameter:")
for i, param in enumerate(target_columns):
    print(f"  {param:20s}: {ensemble_mae[i]:.4f}")

print(f"\nAverage MAE: {ensemble_mae.mean():.4f}")

# Calculate R²
from sklearn.metrics import r2_score
r2 = r2_score(y_test, ensemble_pred)
print(f"R² Score: {r2:.4f}")

# Compare to best individual model
best_single_mae = min([np.abs(pred - y_test).mean() for pred in test_predictions])
print(f"\nBest single model MAE: {best_single_mae:.4f}")
print(f"Ensemble MAE: {ensemble_mae.mean():.4f}")
improvement = ((best_single_mae - ensemble_mae.mean()) / best_single_mae) * 100
print(f"Improvement: {improvement:.1f}%")

# Save ensemble models
import os
os.makedirs('Model/ensemble', exist_ok=True)

for i in range(NUM_MODELS):
    torch.save(models[i].state_dict(), f'Model/ensemble/model_{i}.pth')
    joblib.dump(scalers_X[i], f'Model/ensemble/scaler_X_{i}.pkl')
    joblib.dump(scalers_y[i], f'Model/ensemble/scaler_y_{i}.pkl')

# Save configuration
config = {
    'num_models': NUM_MODELS,
    'predictable_params': target_columns,
    'input_size': 384
}
joblib.dump(config, 'Model/ensemble/config.pkl')

print("\nEnsemble models saved to 'Model/ensemble/'")