import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from model_references import target_columns

# Define variables to be calculated in this file
combined_df = None
X_train, X_test, y_train, y_test, scaler_y, scaler_X = (None,) * 6

def return_combined_df():
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

    # Merge on 'word' column (inner join = only words in both datasets)
    combined_xanew_custom = pd.merge(xanew, custom, left_on='word', right_on='word', how='inner')

    # Merge with the word embeddings as well
    combined_with_embeddings = pd.merge(combined_xanew_custom, word_embeddings, left_on='word', right_on='word', how='inner')

    print(f"Combined dataset has {len(combined_with_embeddings)} words")
    print(combined_with_embeddings.head())

    return combined_with_embeddings

def get_combined_df():
    return combined_df;

def return_training_data():
    # Extract features (X) and targets (y)
    X = np.stack(combined_df['embedding'].values)  # Shape: (num_words, 384)
    y = combined_df[target_columns].values         # Shape: (num_words, 6)

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # Normalize data, to help reduce overfitting
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)
    y_train = scaler_y.fit_transform(y_train)
    y_test = scaler_y.transform(y_test)

    return X_train, X_test, y_train, y_test, scaler_X, scaler_y

def get_training_data():
    return X_train, X_test, y_train, y_test

def get_scalers():
    return scaler_X, scaler_y

combined_df = return_combined_df()
X_train, X_test, y_train, y_test, scaler_X, scaler_y = return_training_data()