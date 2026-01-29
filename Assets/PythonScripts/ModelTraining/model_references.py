import torch.nn as nn

class EmbeddingToParams(nn.Module):
    def __init__(self, input_dim=384, output_dim=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)
    
class SimplifiedModel(nn.Module):
    def __init__(self, input_size=384, output_size=9):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

from sentence_transformers import SentenceTransformer
# Load sentence transformer
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Define which columns are the target parameters
# Map XANEW dimensions to Unity parameters
target_columns = [
    'arousal_norm',
    'valence_norm',
    'dominance_norm',
    'symmetry',
    'complexity',
    'ornament_density',
    'intensity',
    'depth',
    'thickness'
]