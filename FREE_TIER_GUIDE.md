# AI Recommendation Engine: Free Implementation Guide

## 🎯 Build Production-Grade Recommendation System with $0 Budget

---

## Free Technology Stack

| Component | Free Tool | Free Tier Limits |
|-----------|-----------|------------------|
| **Data Processing** | PySpark (Local/Colab) | Unlimited (local) |
| **Model Training** | PyTorch + Google Colab | 12 hrs/session, Free GPU |
| **Orchestration** | Apache Airflow (Local) | Unlimited (self-hosted) |
| **Database** | PostgreSQL/SQLite | Unlimited (local) |
| **Caching** | Redis (Local) | Unlimited (local) |
| **Model Serving** | FastAPI + Render/Railway | 750 hrs/month free |
| **Storage** | GitHub LFS / Hugging Face | 2GB free (GitHub) |
| **Monitoring** | Prometheus + Grafana Cloud | 10K metrics free |
| **Version Control** | Git + GitHub | Unlimited public repos |

---

## Architecture: Free Version

```
┌─────────────────────────────────────────────────────────┐
│              DATA COLLECTION (FREE)                      │
├─────────────────────────────────────────────────────────┤
│ CSV Files / SQLite Database / PostgreSQL (Local)        │
│ Sample Datasets: MovieLens, Amazon Reviews (Public)     │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│           DATA PROCESSING (FREE)                         │
├─────────────────────────────────────────────────────────┤
│ PySpark (Local) / Pandas / Dask                         │
│ Google Colab (Free GPU for processing)                  │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│            MODEL TRAINING (FREE)                         │
├─────────────────────────────────────────────────────────┤
│ Google Colab (Free GPU T4)                              │
│ PyTorch / TensorFlow                                     │
│ Kaggle Notebooks (30hrs/week GPU)                       │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│           DEPLOYMENT (FREE)                              │
├─────────────────────────────────────────────────────────┤
│ Render.com / Railway.app / Hugging Face Spaces          │
│ FastAPI + Uvicorn                                        │
└─────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Implementation

### Step 1: Setup Development Environment (FREE)

**Install Required Packages:**
```bash
# Create virtual environment
python -m venv rec_env
source rec_env/bin/activate  # On Windows: rec_env\Scripts\activate

# Install free libraries
pip install pandas numpy scikit-learn
pip install torch torchvision torchaudio  # CPU version (free)
pip install fastapi uvicorn
pip install sqlalchemy psycopg2-binary
pip install redis
pip install apache-airflow
pip install prometheus-client
```

---

### Step 2: Get Free Dataset

**Option 1: MovieLens Dataset (FREE)**
```python
# Download from: https://grouplens.org/datasets/movielens/
# Or use this code:

import pandas as pd
import requests
import zipfile
import io

# Download MovieLens 25M dataset (free)
url = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
response = requests.get(url)
z = zipfile.ZipFile(io.BytesIO(response.content))
z.extractall("data/")

# Load data
ratings = pd.read_csv('data/ml-25m/ratings.csv')
movies = pd.read_csv('data/ml-25m/movies.csv')

print(f"Total ratings: {len(ratings)}")  # ~25 million ratings
print(f"Total users: {ratings['userId'].nunique()}")
print(f"Total movies: {ratings['movieId'].nunique()}")
```

**Option 2: Generate Synthetic Data (FREE)**
```python
import numpy as np
import pandas as pd

# Generate synthetic user-item interactions
np.random.seed(42)

num_users = 10000
num_items = 5000
num_interactions = 100000

users = np.random.randint(0, num_users, num_interactions)
items = np.random.randint(0, num_items, num_interactions)
ratings = np.random.randint(1, 6, num_interactions)
timestamps = np.random.randint(1609459200, 1735171200, num_interactions)

df = pd.DataFrame({
    'user_id': users,
    'item_id': items,
    'rating': ratings,
    'timestamp': timestamps
})

df.to_csv('data/interactions.csv', index=False)
```

---

### Step 3: Data Processing with Free Tools

**Using Pandas (Small-Medium Data, FREE):**
```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('data/interactions.csv')

# Feature engineering
user_stats = df.groupby('user_id').agg({
    'rating': ['count', 'mean', 'std'],
    'item_id': 'nunique'
}).reset_index()

item_stats = df.groupby('item_id').agg({
    'rating': ['count', 'mean', 'std'],
    'user_id': 'nunique'
}).reset_index()

# Create user-item matrix
user_item_matrix = df.pivot_table(
    index='user_id', 
    columns='item_id', 
    values='rating'
).fillna(0)

# Train-test split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

print(f"Training samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")
```

**Using PySpark (Large Data, FREE - Local):**
```python
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS

# Initialize Spark (free, runs locally)
spark = SparkSession.builder \
    .appName("RecommendationEngine") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# Load data
df = spark.read.csv('data/interactions.csv', header=True, inferSchema=True)

# Split data
train, test = df.randomSplit([0.8, 0.2], seed=42)

# Train ALS model (Matrix Factorization)
als = ALS(
    userCol="user_id",
    itemCol="item_id", 
    ratingCol="rating",
    coldStartStrategy="drop",
    nonnegative=True
)

model = als.fit(train)
predictions = model.transform(test)

# Calculate RMSE
from pyspark.ml.evaluation import RegressionEvaluator
evaluator = RegressionEvaluator(
    metricName="rmse",
    labelCol="rating",
    predictionCol="prediction"
)
rmse = evaluator.evaluate(predictions)
print(f"RMSE: {rmse}")
```

---

### Step 4: Build Neural Network Model (FREE - Google Colab)

**Create this notebook in Google Colab (Free GPU):**

```python
# Run this in Google Colab for free GPU access
# Runtime > Change runtime type > T4 GPU (FREE)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# Mount Google Drive to save models
from google.colab import drive
drive.mount('/content/drive')

# Load data
df = pd.read_csv('/content/interactions.csv')

# Create PyTorch Dataset
class InteractionDataset(Dataset):
    def __init__(self, df):
        self.users = torch.LongTensor(df['user_id'].values)
        self.items = torch.LongTensor(df['item_id'].values)
        self.ratings = torch.FloatTensor(df['rating'].values)
    
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]

# Neural Collaborative Filtering Model
class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64):
        super(NCF, self).__init__()
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, user_ids, item_ids):
        user_embedded = self.user_embedding(user_ids)
        item_embedded = self.item_embedding(item_ids)
        concatenated = torch.cat([user_embedded, item_embedded], dim=-1)
        output = self.fc_layers(concatenated)
        return output.squeeze()

# Training setup
num_users = df['user_id'].max() + 1
num_items = df['item_id'].max() + 1

model = NCF(num_users, num_items, embedding_dim=64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

print(f"Training on: {device}")  # Should say 'cuda' in Colab with GPU

dataset = InteractionDataset(df)
train_loader = DataLoader(dataset, batch_size=1024, shuffle=True)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
for epoch in range(epochs):
    total_loss = 0
    model.train()
    
    for users, items, ratings in train_loader:
        users = users.to(device)
        items = items.to(device)
        ratings = ratings.to(device)
        
        optimizer.zero_grad()
        predictions = model(users, items)
        loss = criterion(predictions, ratings)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# Save model to Google Drive (FREE storage)
torch.save(model.state_dict(), '/content/drive/MyDrive/recommendation_model.pth')
print("Model saved to Google Drive!")
```

---

### Step 5: Build FastAPI Server (FREE)

**Create `app.py`:**
```python
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import numpy as np
import pickle
from typing import List

app = FastAPI(title="Free Recommendation API")

# Load model (from Google Drive or local)
class NCF(torch.nn.Module):
    # ... (same model definition as above)
    pass

# Initialize
model = NCF(num_users=10000, num_items=5000)
model.load_state_dict(torch.load('recommendation_model.pth', map_location='cpu'))
model.eval()

# In-memory cache (FREE alternative to Redis)
recommendation_cache = {}

class RecommendationRequest(BaseModel):
    user_id: int
    num_recommendations: int = 10

@app.get("/")
def read_root():
    return {"message": "Free Recommendation Engine API", "status": "active"}

@app.post("/recommend")
def get_recommendations(request: RecommendationRequest):
    user_id = request.user_id
    
    # Check cache
    cache_key = f"user_{user_id}"
    if cache_key in recommendation_cache:
        return {"user_id": user_id, "recommendations": recommendation_cache[cache_key]}
    
    # Generate predictions for all items
    user_tensor = torch.LongTensor([user_id] * 5000)
    item_tensor = torch.LongTensor(range(5000))
    
    with torch.no_grad():
        scores = model(user_tensor, item_tensor)
    
    # Get top-k recommendations
    top_k = torch.topk(scores, request.num_recommendations)
    recommended_items = top_k.indices.tolist()
    
    # Cache results
    recommendation_cache[cache_key] = recommended_items
    
    return {
        "user_id": user_id,
        "recommendations": recommended_items,
        "scores": top_k.values.tolist()
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": True}

# Run locally: uvicorn app:app --reload
```

---

### Step 6: Deploy for FREE

**Option 1: Render.com (FREE - 750 hours/month)**

1. Create `requirements.txt`:
```txt
fastapi==0.104.1
uvicorn==0.24.0
torch==2.1.0
numpy==1.24.3
pydantic==2.5.0
```

2. Create `render.yaml`:
```yaml
services:
  - type: web
    name: recommendation-api
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn app:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: PYTHON_VERSION
        value: 3.10.0
```

3. Push to GitHub and connect to Render.com
4. Deploy! (Takes 2-3 minutes)

**Option 2: Hugging Face Spaces (FREE - Unlimited)**

```python
# Create app.py for Gradio interface
import gradio as gr
import torch

def recommend(user_id, num_items):
    # ... recommendation logic
    return recommendations

iface = gr.Interface(
    fn=recommend,
    inputs=[
        gr.Number(label="User ID"),
        gr.Slider(1, 20, value=10, label="Number of Recommendations")
    ],
    outputs=gr.JSON(label="Recommendations"),
    title="Free AI Recommendation Engine"
)

iface.launch()
```

Upload to Hugging Face Spaces - completely FREE!

---

### Step 7: Free Orchestration with Airflow (Local)

**Install Airflow (FREE - runs locally):**
```bash
pip install apache-airflow

# Initialize database
airflow db init

# Create user
airflow users create \
    --username admin \
    --password admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com

# Start webserver
airflow webserver -p 8080

# Start scheduler (in another terminal)
airflow scheduler
```

**Create DAG (`dags/recommendation_pipeline.py`):**
```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import torch

def extract_data():
    """Extract new interaction data"""
    df = pd.read_csv('data/new_interactions.csv')
    df.to_csv('data/staging.csv', index=False)
    print(f"Extracted {len(df)} new interactions")

def train_model():
    """Retrain model with new data"""
    # ... training code
    print("Model retrained successfully")

def evaluate_model():
    """Evaluate model performance"""
    # ... evaluation code
    print("Model evaluation complete")

def deploy_model():
    """Deploy new model if better"""
    # ... deployment logic
    print("Model deployed")

default_args = {
    'owner': 'data-scientist',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'recommendation_training_pipeline',
    default_args=default_args,
    description='Free recommendation model training pipeline',
    schedule_interval='0 2 * * *',  # Daily at 2 AM
    catchup=False
)

t1 = PythonOperator(
    task_id='extract_data',
    python_callable=extract_data,
    dag=dag,
)

t2 = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

t3 = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    dag=dag,
)

t4 = PythonOperator(
    task_id='deploy_model',
    python_callable=deploy_model,
    dag=dag,
)

t1 >> t2 >> t3 >> t4
```

---

### Step 8: Free Monitoring

**Prometheus + Grafana Cloud (FREE):**

```python
# Add to app.py
from prometheus_client import Counter, Histogram, generate_latest
from fastapi import Response

# Metrics
request_count = Counter('api_requests_total', 'Total API requests')
request_duration = Histogram('api_request_duration_seconds', 'Request duration')

@app.middleware("http")
async def metrics_middleware(request, call_next):
    request_count.inc()
    with request_duration.time():
        response = await call_next(request)
    return response

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

**Sign up for Grafana Cloud (FREE tier):**
- 10K metrics
- 50GB logs
- 14-day retention

---

## Complete Free Setup Summary

### Total Cost: $0/month

| Service | Free Tier | What You Get |
|---------|-----------|--------------|
| **Google Colab** | Free T4 GPU | 12 hours/session |
| **Kaggle Notebooks** | Free GPU | 30 hours/week |
| **Render.com** | 750 hours | API hosting |
| **Hugging Face** | Unlimited | Model + app hosting |
| **GitHub** | Unlimited | Code + 2GB storage |
| **Grafana Cloud** | 10K metrics | Monitoring |
| **Local Airflow** | Unlimited | Orchestration |

### Performance Expectations (FREE Tier):
- ✅ Handle 100-1K requests/day
- ✅ Train models up to 100M parameters
- ✅ Store 10GB+ of data
- ✅ Full MLOps pipeline
- ✅ A/B testing capability
- ✅ Real-time monitoring

---

## Quick Start Commands

```bash
# 1. Clone starter template
git clone https://github.com/your-username/free-recommendation-engine
cd free-recommendation-engine

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset
python scripts/download_data.py

# 4. Train model (locally or upload to Colab)
python train.py

# 5. Run API locally
uvicorn app:app --reload

# 6. Test API
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 123, "num_recommendations": 10}'

# 7. Deploy to Render (free)
git push origin main  # Auto-deploys if connected to Render
```

---

## Scaling Tips (Still FREE)

1. **Use Colab Pro (if needed)**: $9.99/month for longer GPU access
2. **Cache aggressively**: Use Python dict or local Redis
3. **Batch predictions**: Pre-compute for active users
4. **Model compression**: Quantize models to reduce size
5. **CDN for models**: Use GitHub LFS or Hugging Face Hub

---

## Free Alternatives Comparison

| Need | Free Option | Paid Alternative |
|------|-------------|------------------|
| GPU Training | Google Colab | GCP Vertex AI ($$$) |
| API Hosting | Render/Railway | AWS Lambda ($$) |
| Data Processing | Local Spark | Databricks ($$$) |
| Monitoring | Grafana Cloud | Datadog ($$$) |
| Storage | GitHub/HF | AWS S3 ($$) |

**Savings: $500-2000/month by using free tools!** 
