import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.recommender import RecommenderNet
from src.config import Config

def generate_dummy_data(num_users=1000, num_items=500, num_samples=10000):
    """Generate random data for testing the training loop"""
    user_ids = torch.randint(0, num_users, (num_samples,))
    item_ids = torch.randint(0, num_items, (num_samples,))
    labels = torch.randint(0, 2, (num_samples,)).float()
    return user_ids, item_ids, labels

def load_real_data(data_path):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    user_ids = torch.tensor(df['user_id'].values, dtype=torch.long)
    item_ids = torch.tensor(df['item_id'].values, dtype=torch.long)
    # Using 'rating' or 'click'. For now, assuming binary or normalizing rating
    # If using regression, change model output to linear and loss to MSE
    # Here keeping it simple: treat rating > 3 as 1, else 0, or just use click if available
    if 'click' in df.columns:
        labels = torch.tensor(df['click'].values, dtype=torch.float)
    else:
        labels = torch.tensor((df['rating'] >= 3.5).astype(int).values, dtype=torch.float)
    
    num_users = df['user_id'].max() + 1
    num_items = df['item_id'].max() + 1
    return user_ids, item_ids, labels, num_users, num_items

def train_model():
    print(f"Starting training with config: LR={Config.LEARNING_RATE}, Batch={Config.BATCH_SIZE}")
    
    data_path = os.path.join(Config.DATA_DIR, "interactions.csv")
    
    if os.path.exists(data_path):
        user_ids, item_ids, labels, num_users, num_items = load_real_data(data_path)
        print(f"Loaded real data: {len(labels)} interactions, {num_users} users, {num_items} items")
    else:
        print("Real data not found, generating dummy data...")
        num_users = 1000
        num_items = 500
        user_ids, item_ids, labels = generate_dummy_data(num_users, num_items)
    
    # Prepare data
    dataset = TensorDataset(user_ids, item_ids, labels)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    
    # Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    model = RecommenderNet(
        num_users=int(num_users), 
        num_items=int(num_items),
        embedding_dim=Config.EMBEDDING_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        dropout=Config.DROPOUT
    ).to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    # Training Loop
    model.train()
    for epoch in range(Config.EPOCHS):
        total_loss = 0
        for batch_users, batch_items, batch_labels in dataloader:
            batch_users, batch_items, batch_labels = batch_users.to(device), batch_items.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_users, batch_items)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{Config.EPOCHS} - Loss: {avg_loss:.4f}")

    # Save Model
    if not os.path.exists(Config.MODEL_DIR):
        os.makedirs(Config.MODEL_DIR)
    
    model_path = os.path.join(Config.MODEL_DIR, "recommender.pth")
    torch.save(model.state_dict(), model_path)
    
    # Also save metadata for serving
    import json
    metadata = {"num_users": int(num_users), "num_items": int(num_items)}
    with open(os.path.join(Config.MODEL_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f)
        
    print(f"Model saved to {model_path}")
    print(f"Metadata saved to {os.path.join(Config.MODEL_DIR, 'metadata.json')}")

if __name__ == "__main__":
    train_model()
