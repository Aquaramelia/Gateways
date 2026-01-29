import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
import joblib
import os
from load_dataset import get_training_data, get_scalers, get_combined_df
from model_references import target_columns

X_train, X_test, y_train, y_test = get_training_data()
scaler_X, scaler_y = get_scalers()
combined_df = get_combined_df()

print("\n Training Random Forest Model")

# Create Random Forest model with MultiOutputRegressor
# Hyperparameters tuned for small datasets
rf_model = MultiOutputRegressor(
    RandomForestRegressor(
        n_estimators=200,        # Number of trees
        max_depth=8,             # Limit depth to prevent overfitting
        min_samples_split=5,     # Minimum samples to split a node
        min_samples_leaf=2,      # Minimum samples in leaf node
        max_features='sqrt',     # Number of features to consider at each split
        random_state=42,
        n_jobs=-1                # Use all CPU cores
    )
)

# Train the model
print("\n Training...")
rf_model.fit(X_train, y_train)
print("Training complete!")

# Make predictions on train and test sets
y_train_pred_scaled = rf_model.predict(X_train)
y_test_pred_scaled = rf_model.predict(X_test)

# Convert back to original scale
y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled)
y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled)

# Calculate metrics
from sklearn.metrics import r2_score

print("\n Training Set Performance \n")
train_mae_per_param = np.abs(y_train_pred - y_train).mean(axis=0)
for i, col in enumerate(target_columns):
    print(f'{col:20s}: MAE = {train_mae_per_param[i]:.4f}')

print("\n Test Set Performance (Validation) \n")
test_mae_per_param = np.abs(y_test_pred - y_test).mean(axis=0)
for i, col in enumerate(target_columns):
    print(f'{col:20s}: MAE = {test_mae_per_param[i]:.4f}')

# Calculate overall metrics
print("\n Overall Metrics \n")
print(f"Average Training MAE:   {train_mae_per_param.mean():.4f}")
print(f"Average Test MAE:       {test_mae_per_param.mean():.4f}")
print(f"Overfitting Ratio:      {test_mae_per_param.mean() / train_mae_per_param.mean():.2f}x")

# R² scores
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
print(f"\nTraining R² Score:      {train_r2:.4f}")
print(f"Test R² Score:          {test_r2:.4f}")

# Feature importance analysis (optional but interesting!)
print("\n Feature Importance (Top 10 embedding dimensions) \n")

# Get feature importances for each target
for i, col in enumerate(target_columns):
    importances = rf_model.estimators_[i].feature_importances_
    top_features = np.argsort(importances)[-10:][::-1]
    print(f"\n{col}:")
    for j, feat_idx in enumerate(top_features[:5]):  # Show top 5
        print(f"  Dimension {feat_idx}: {importances[feat_idx]:.4f}")

# Save the model and scalers
os.makedirs('Model', exist_ok=True)
joblib.dump(rf_model, 'Model/random_forest_model.pkl')
joblib.dump(scaler_X, 'Model/scaler_X.pkl')
joblib.dump(scaler_y, 'Model/scaler_y.pkl')

print("Model saved to 'Model/random_forest_model.pkl'")
print("Scalers saved to 'Model/scaler_X.pkl' and 'Model/scaler_y.pkl'")

# Save a comparison of predictions vs actual for test set

# Get the test set indices
_, test_indices = train_test_split(
    range(len(combined_df)), 
    test_size=0.2, 
    random_state=42
)

comparison_df = pd.DataFrame(y_test, columns=target_columns)
comparison_df['word'] = combined_df.iloc[test_indices]['word'].values
for i, col in enumerate(target_columns):
    comparison_df[f'{col}_predicted'] = y_test_pred[:, i]
    comparison_df[f'{col}_error'] = np.abs(y_test[:, i] - y_test_pred[:, i])

comparison_df.to_csv('Model/predictions_comparison.csv', index=False)
print("Predictions saved to 'Model/predictions_comparison.csv'")
