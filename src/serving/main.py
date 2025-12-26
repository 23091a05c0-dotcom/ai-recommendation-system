from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from typing import List
import torch
import sys
import os
import json
from prometheus_client import Counter, Histogram, generate_latest

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.recommender import RecommenderNet
from src.config import Config

# Trigger reload (attempt 2)
app = FastAPI(title="RecSys Serving API", version="1.0")

# Metrics
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests')
REQUEST_DURATION = Histogram('api_request_duration_seconds', 'Request duration')

class PredictionRequest(BaseModel):
    user_id: int
    item_ids: List[int]

class PredictionResponse(BaseModel):
    user_id: int
    recommendations: List[dict]

# Global variables
model = None
device = None
item_mapping = {} # internal_id -> original_movieId
movie_titles = {} # original_movieId -> title

@app.middleware("http")
async def metrics_middleware(request, call_next):
    REQUEST_COUNT.inc()
    with REQUEST_DURATION.time():
        response = await call_next(request)
    return response

@app.on_event("startup")
def load_resources():
    global model, device, item_mapping, movie_titles
    print("Loading model and resources...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Init default sizes
    NUM_USERS = 1000  
    NUM_ITEMS = 500
    
    # Load Metadata
    metadata_path = os.path.join(Config.MODEL_DIR, "metadata.json")
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                NUM_USERS = metadata.get("num_users", NUM_USERS)
                NUM_ITEMS = metadata.get("num_items", NUM_ITEMS)
            print(f"Loaded metadata: {NUM_USERS} users, {NUM_ITEMS} items")
        except Exception as e:
            print(f"Error loading metadata: {e}")

    # Load Item Mapping (internal_id -> original_movieId)
    # The JSON keys are strings (e.g., "0": 1), need to convert keys to int
    mapping_path = os.path.join(Config.DATA_DIR, "item_mapping.json")
    if os.path.exists(mapping_path):
        try:
            with open(mapping_path, "r") as f:
                raw_mapping = json.load(f)
                item_mapping = {int(k): v for k, v in raw_mapping.items()}
            print(f"Loaded item mapping for {len(item_mapping)} items")
        except Exception as e:
            print(f"Error loading item mapping: {e}")
            
    # Load Movie Titles
    movies_path = os.path.join(Config.DATA_DIR, "ml-latest-small", "movies.csv")
    if os.path.exists(movies_path):
        try:
            import pandas as pd
            df_movies = pd.read_csv(movies_path)
            # Create dict: movieId -> title
            movie_titles = dict(zip(df_movies.movieId, df_movies.title))
            print(f"Loaded {len(movie_titles)} movie titles")
        except Exception as e:
            print(f"Error loading movie titles: {e}")

    # Initialize Model
    model = RecommenderNet(
        num_users=NUM_USERS,
        num_items=NUM_ITEMS,
        embedding_dim=Config.EMBEDDING_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        dropout=Config.DROPOUT
    ).to(device)
    
    model_path = os.path.join(Config.MODEL_DIR, "recommender.pth")
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            print(f"Model loaded from {model_path}")
        except RuntimeError as e:
            print(f"Error loading model weights (likely size mismatch): {e}")
            print("Re-initializing model with random weights for now.")
    else:
        print(f"Warning: Model not found at {model_path}. Using initialized weights.")
        model.eval()

# In-memory cache
prediction_cache = {}

@app.get("/")
def root():
    """Root endpoint with API documentation"""
    return {
        "message": "AI Recommendation System API",
        "version": "1.0",
        "status": "active",
        "endpoints": {
            "/health": "Health check endpoint",
            "/metrics": "Prometheus metrics",
            "/predict": "POST - Get predictions for specific items",
            "/recommend_top_k": "POST - Get top-K recommendations for a user"
        },
        "docs": "/docs",
        "redoc": "/redoc"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Check Cache
    # Create a simple key from user_id and sorted item_ids tuple
    cache_key = (request.user_id, tuple(sorted(request.item_ids)))
    if cache_key in prediction_cache:
        # metrics check: we could increment a cache_hit counter here
        return prediction_cache[cache_key]

    try:
        user_tensor = torch.tensor([request.user_id] * len(request.item_ids)).to(device)
        items_tensor = torch.tensor(request.item_ids).to(device)
        
        with torch.no_grad():
            predictions = model(user_tensor, items_tensor)
            
        # Convert to list
        scores = predictions.cpu().numpy().tolist()
        
        # Format results
        results = []
        for item_id, score in zip(request.item_ids, scores):
            # Resolve title
            original_id = item_mapping.get(item_id, None)
            title = "Unknown"
            if original_id is not None:
                title = movie_titles.get(original_id, f"Movie {original_id}")
            
            results.append({
                "item_id": item_id, 
                "original_id": original_id,
                "title": title,
                "score": score
            })
            
        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        
        response = PredictionResponse(user_id=request.user_id, recommendations=results)
        
        # Update Cache (Simple unbounded cache for demo - use LRU in prod)
        prediction_cache[cache_key] = response
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class TopKRequest(BaseModel):
    user_id: int
    k: int = 10

@app.post("/recommend_top_k", response_model=PredictionResponse)
async def recommend_top_k(request: TopKRequest):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
        
    try:
        # Create tensor for this user against ALL items
        # We need to know max_item_id. Ideally we get this from metadata or model.
        # Assuming model.item_embedding.num_embeddings is the count
        num_items = model.item_embedding.num_embeddings
        all_items = torch.arange(num_items).to(device)
        user_tensor = torch.tensor([request.user_id] * num_items).to(device)
        
        with torch.no_grad():
            predictions = model(user_tensor, all_items)
            
        # Get Top-K
        # torch.topk returns (values, indices)
        top_k_scores, top_k_indices = torch.topk(predictions, request.k)
        
        scores = top_k_scores.cpu().numpy().tolist()
        item_ids = top_k_indices.cpu().numpy().tolist()
        
        results = []
        for item_id, score in zip(item_ids, scores):
             # Resolve title
            original_id = item_mapping.get(item_id, None)
            title = "Unknown"
            if original_id is not None:
                title = movie_titles.get(original_id, f"Movie {original_id}")
                
            results.append({
                "item_id": item_id,
                "original_id": original_id,
                "title": title,
                "score": score
            })
            
        return PredictionResponse(user_id=request.user_id, recommendations=results)

    except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None, "cache_size": len(prediction_cache)}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")
