import torch.nn as nn

class EmbeddingToParams(nn.Module):
    def __init__(self, input_dim=384, output_dim=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

from sentence_transformers import SentenceTransformer
# Load sentence transformer
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')