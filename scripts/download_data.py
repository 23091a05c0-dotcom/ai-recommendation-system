import requests
import zipfile
import io
import pandas as pd
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Config

def download_movielens_small():
    url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    save_dir = Config.DATA_DIR
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    print(f"Downloading from {url}...")
    response = requests.get(url)
    
    print("Extracting...")
    z = zipfile.ZipFile(io.BytesIO(response.content))
    z.extractall(save_dir)
    
    # Process
    ratings_path = os.path.join(save_dir, "ml-latest-small", "ratings.csv")
    print(f"Processing {ratings_path}...")
    df = pd.read_csv(ratings_path)
    
    # Remap IDs to contiguous range 0...N-1
    user_mapping = {id: i for i, id in enumerate(df['userId'].unique())}
    item_mapping = {id: i for i, id in enumerate(df['movieId'].unique())}
    
    df['user_id'] = df['userId'].map(user_mapping)
    df['item_id'] = df['movieId'].map(item_mapping)
    df['click'] = 1 # Implicit feedback assumption for now
    
    final_path = os.path.join(save_dir, "interactions.csv")
    df[['user_id', 'item_id', 'rating', 'timestamp']].to_csv(final_path, index=False)
    
    # Save Item Mapping for Serving
    import json
    mapping_path = os.path.join(save_dir, "item_mapping.json")
    # Convert numpy ints to python ints for JSON serialization if needed, though enumerate yields ints
    # But keys in JSON must be strings? No, we want {internal_id: original_movieId} 
    # Actually, we want to look up "Title" by "internal_id".
    # So we need: internal_id -> original_movieId -> Title
    
    # Let's save {internal_id: original_movieId}
    # item_mapping is {original: internal} -> reverse it
    # ensure values are python ints (not numpy int64)
    reverse_item_mapping = {int(v): int(k) for k, v in item_mapping.items()}
    with open(mapping_path, "w") as f:
        json.dump(reverse_item_mapping, f)
        
    print(f"Saved processed data to {final_path}")
    print(f"Saved item mapping to {mapping_path}")
    print(f"Total Users: {len(user_mapping)}, Total Items: {len(item_mapping)}")

if __name__ == "__main__":
    download_movielens_small()
