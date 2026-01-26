import pandas as pd
from model_references import sentence_model

# Load XANEW filtered words
vocabulary = pd.read_csv('relevant_vocabulary.csv')

# Get embeddings for all words
words = vocabulary['word'].tolist()
embeddings = sentence_model.encode(words, show_progress_bar=True)

print(f"Embeddings shape: {embeddings.shape}")  # Should be (num_words, 384)

# Add embeddings to dataframe
vocabulary['embedding'] = list(embeddings)

# Keep only word + embedding
embeddings_df = vocabulary[["word", "embedding"]]

# Save to CSV
embeddings_df.to_csv("word_embeddings.csv", index=False)
