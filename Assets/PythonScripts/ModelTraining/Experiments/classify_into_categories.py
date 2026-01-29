"""
CATEGORY-BASED PARAMETER PREDICTION SYSTEM

Instead of predicting continuous parameters directly, we:
1. Classify each word into one of 35 semantic categories
2. Sample parameters from category-specific distributions

"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from load_dataset import return_combined_df
from category_definitions import CATEGORY_DEFINITIONS

# Build word-to-category mapping
WORD_TO_CATEGORY = {}
for category, info in CATEGORY_DEFINITIONS.items():
    for word in info['words']:
        WORD_TO_CATEGORY[word] = category

ALL_CATEGORIES = list(CATEGORY_DEFINITIONS.keys())
NUM_CATEGORIES = len(ALL_CATEGORIES)
CATEGORY_TO_IDX = {cat: i for i, cat in enumerate(ALL_CATEGORIES)}

print(f"Loaded {NUM_CATEGORIES} categories with {len(WORD_TO_CATEGORY)} words")

# Neural network classifier
class CategoryClassifier(nn.Module):
    """Classify word embeddings into semantic categories"""
    def __init__(self, input_size=384, num_categories=NUM_CATEGORIES):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_categories)
        )
    
    def forward(self, x):
        return self.network(x)


def sample_from_category(category_name, param_name=None, seed=None):
    """
    Sample parameter value(s) from category distribution
    
    Args:
        category_name: Name of semantic category
        param_name: Specific parameter (None = all parameters)
        seed: Random seed for reproducibility
    
    Returns:
        dict of parameter values or single value if param_name specified
    """
    if seed is not None:
        np.random.seed(seed)
    
    params_ranges = CATEGORY_DEFINITIONS[category_name]['params']
    
    if param_name:
        min_val, max_val = params_ranges[param_name]
        value = np.random.uniform(min_val, max_val)
        if param_name == 'detail_frequency':
            value = int(round(value))
        return value
    else:
        # Sample all parameters
        result = {}
        for name, (min_val, max_val) in params_ranges.items():
            value = np.random.uniform(min_val, max_val)
            if name == 'detail_frequency':
                value = int(round(value))
            result[name] = value
        return result


# Training process

print("\n CATEGORY-BASED PARAMETER PREDICTION \n")

# Load data
combined_df = return_combined_df()

# Filter to only words we have categories for
combined_df = combined_df[combined_df['word'].isin(WORD_TO_CATEGORY.keys())]
print(f"\nUsing {len(combined_df)} words with defined categories")

X = np.stack(combined_df['embedding'].values)
words = combined_df['word'].values

# Get category labels
y_categories = np.array([CATEGORY_TO_IDX[WORD_TO_CATEGORY[word]] for word in words])

print(f"\nCategory distribution:")
from collections import Counter
cat_counts = Counter(y_categories)
for cat_idx, count in sorted(cat_counts.items(), key=lambda x: -x[1])[:10]:
    cat_name = ALL_CATEGORIES[cat_idx]
    print(f"  {cat_name:30s}: {count} words")

# Split data
X_train, X_test, y_train, y_test, words_train, words_test = train_test_split(
    X, y_categories, words, test_size=0.2, random_state=42, stratify=y_categories
)

print(f"\nDataset: {len(X_train)} train, {len(X_test)} test")

# Train classifier
print("\nTraining category classifier...")

from sklearn.preprocessing import StandardScaler
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

model = CategoryClassifier(input_size=384, num_categories=NUM_CATEGORIES)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

X_train_t = torch.FloatTensor(X_train_scaled)
y_train_t = torch.LongTensor(y_train)
X_test_t = torch.FloatTensor(X_test_scaled)
y_test_t = torch.LongTensor(y_test)

best_val_acc = 0
best_state = None
patience = 20
patience_counter = 0

for epoch in range(300):
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
    
    # Validation
    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_t)
            val_loss = criterion(val_outputs, y_test_t)
            
            preds = torch.argmax(val_outputs, dim=1)
            accuracy = (preds == y_test_t).float().mean().item()
        
        avg_train_loss = total_loss / len(X_train_t)
        print(f'Epoch {epoch+1:3d}: Train Loss={avg_train_loss:.4f}, Val Loss={val_loss.item():.4f}, Val Acc={accuracy:.4f}')
        
        if accuracy > best_val_acc:
            best_val_acc = accuracy
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# Load best model
model.load_state_dict(best_state)

# Final evaluation
print("\n CLASSIFIER EVALUATION \n")

model.eval()
with torch.no_grad():
    test_outputs = model(X_test_t)
    test_preds = torch.argmax(test_outputs, dim=1).numpy()

test_accuracy = accuracy_score(y_test, test_preds)
print(f"\nTest Accuracy: {test_accuracy:.4f}")

# Per-category accuracy
print("\nPer-category results:")
for cat_idx in range(NUM_CATEGORIES):
    mask = y_test == cat_idx
    if mask.sum() > 0:
        cat_acc = accuracy_score(y_test[mask], test_preds[mask])
        cat_name = ALL_CATEGORIES[cat_idx]
        print(f"  {cat_name:30s}: {cat_acc:.3f} ({mask.sum()} samples)")

# Save model
import os
os.makedirs('Model/category_classifier', exist_ok=True)
torch.save(model.state_dict(), 'Model/category_classifier/model.pth')
joblib.dump(scaler_X, 'Model/category_classifier/scaler_X.pkl')
joblib.dump({
    'categories': ALL_CATEGORIES,
    'category_to_idx': CATEGORY_TO_IDX,
    'word_to_category': WORD_TO_CATEGORY
}, 'Model/category_classifier/config.pkl')

print("\n✓ Category classifier saved to 'Model/category_classifier/'")


# Parameter prediction evaluation
print("\n PARAMETER PREDICTION EVALUATION \n")

all_params = list(CATEGORY_DEFINITIONS[ALL_CATEGORIES[0]]['params'].keys())

# Get ground truth parameters
y_params_test = combined_df.iloc[X_test.shape[0]:][all_params].values

# Generate predictions by sampling from predicted categories
predicted_params = []
for i, word in enumerate(words_test):
    # Get predicted category
    pred_cat_idx = test_preds[i]
    pred_cat_name = ALL_CATEGORIES[pred_cat_idx]
    
    # Sample parameters from this category
    # Use word hash as seed for reproducibility
    seed = hash(word) % (2**32)
    params = sample_from_category(pred_cat_name, seed=seed)
    
    # Convert to array in correct order
    param_array = [params[p] for p in all_params]
    predicted_params.append(param_array)

predicted_params = np.array(predicted_params)

# Calculate MAE
mae_per_param = np.abs(predicted_params - y_params_test).mean(axis=0)

print("\nPer-parameter MAE:")
for i, param in enumerate(all_params):
    print(f"  {param:20s}: {mae_per_param[i]:.4f}")

print(f"\nAverage MAE: {mae_per_param.mean():.4f}")

from sklearn.metrics import r2_score
r2 = r2_score(y_params_test, predicted_params)
print(f"R² Score: {r2:.4f}")