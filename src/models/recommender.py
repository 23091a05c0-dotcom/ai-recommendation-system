import torch
import torch.nn as nn

class RecommenderNet(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32, hidden_dim=64, dropout=0.2):
        super(RecommenderNet, self).__init__()
        
        # Embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Neural Network Layers
        self.fc1 = nn.Linear(embedding_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def forward(self, user_ids, item_ids):
        # Look up embeddings
        user_embed = self.user_embedding(user_ids)
        item_embed = self.item_embedding(item_ids)
        
        # Concatenate user and item embeddings
        x = torch.cat([user_embed, item_embed], dim=1)
        
        # Pass through MLP
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Output probability
        return self.sigmoid(x).squeeze()
