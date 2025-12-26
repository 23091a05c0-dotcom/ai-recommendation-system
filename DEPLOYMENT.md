# Deployment Guide: AI Recommendation System

This guide walks you through deploying the AI recommendation system to Render.com's free tier.

## Prerequisites

- GitHub account
- Render.com account (free tier)
- Git installed locally

## Step 1: Initialize Git Repository

```bash
# Navigate to project directory
cd "d:\ai recomendation system"

# Initialize Git
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: AI Recommendation System"
```

## Step 2: Create GitHub Repository

1. Go to [GitHub](https://github.com) and log in
2. Click the "+" icon in the top right → "New repository"
3. Name it: `ai-recommendation-system`
4. Keep it **Public** (required for Render free tier)
5. **Do NOT** initialize with README, .gitignore, or license
6. Click "Create repository"

## Step 3: Push to GitHub

```bash
# Add GitHub remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/ai-recommendation-system.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## Step 4: Check Model File Size

```bash
# Check if model file is over 100MB
dir models\recommender.pth
```

**If the file is over 100MB:**
- Option A: Use Git LFS (see section below)
- Option B: Exclude from Git and retrain on deployment
- Option C: Use a smaller model

### Using Git LFS (if needed)

```bash
# Install Git LFS
git lfs install

# Track .pth files
git lfs track "*.pth"

# Add .gitattributes
git add .gitattributes

# Commit and push
git add models/recommender.pth
git commit -m "Add model with Git LFS"
git push
```

## Step 5: Deploy to Render.com

1. Go to [Render.com](https://render.com) and sign up/log in
2. Click "New +" → "Web Service"
3. Connect your GitHub account if not already connected
4. Select your `ai-recommendation-system` repository
5. Configure the service:
   - **Name**: `ai-recommendation-engine` (or your choice)
   - **Region**: Oregon (Free)
   - **Branch**: `main`
   - **Runtime**: Python 3
   - **Build Command**: `pip install --upgrade pip && pip install -r requirements.txt`
   - **Start Command**: `uvicorn src.serving.main:app --host 0.0.0.0 --port $PORT`
6. Click "Create Web Service"

Render will automatically use the `render.yaml` configuration file.

## Step 6: Monitor Deployment

1. Watch the deployment logs in Render dashboard
2. Wait for "Live" status (usually 3-5 minutes)
3. Note your service URL: `https://ai-recommendation-engine.onrender.com`

## Step 7: Test Deployed API

### Test Health Endpoint
```bash
curl https://ai-recommendation-engine.onrender.com/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "cache_size": 0
}
```

### Test Root Endpoint
Visit in browser: `https://ai-recommendation-engine.onrender.com/`

### Test Recommendations
```bash
curl -X POST https://ai-recommendation-engine.onrender.com/recommend_top_k \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "k": 5}'
```

### View API Documentation
Visit: `https://ai-recommendation-engine.onrender.com/docs`

## Step 8: Monitor with Prometheus

Access metrics at: `https://ai-recommendation-engine.onrender.com/metrics`

## Troubleshooting

### Issue: Service won't start

**Check logs in Render dashboard for:**
- Missing dependencies → Update `requirements.txt`
- Model file not found → Verify Git LFS or model path
- Port binding issues → Ensure using `$PORT` environment variable

### Issue: Cold starts (slow first request)

**This is normal for free tier:**
- Services sleep after 15 minutes of inactivity
- First request after sleep takes 30-60 seconds
- Subsequent requests are fast

**Solution**: Upgrade to paid tier or use a ping service

### Issue: Model file too large

**Solutions:**
1. Use Git LFS (see Step 4)
2. Download model during build:
   ```yaml
   buildCommand: |
     pip install -r requirements.txt
     python scripts/download_model.py
   ```
3. Use a smaller model or quantization

### Issue: Out of memory

**Free tier has 512MB RAM:**
- Reduce model size
- Use CPU-only PyTorch
- Optimize embedding dimensions

## Environment Variables

If needed, add environment variables in Render dashboard:
- `MODEL_PATH`: Custom model path
- `DATA_PATH`: Custom data path
- `LOG_LEVEL`: Logging level (INFO, DEBUG, etc.)

## Auto-Deploy

Render automatically deploys when you push to the `main` branch:

```bash
# Make changes
git add .
git commit -m "Update model"
git push

# Render will auto-deploy
```

## Monitoring & Logs

- **Logs**: View in Render dashboard → Your service → Logs
- **Metrics**: Access `/metrics` endpoint
- **Health**: Monitor `/health` endpoint

## Cost

**Free Tier Limits:**
- 750 hours/month (enough for 1 service running 24/7)
- Services sleep after 15 min inactivity
- 512MB RAM
- Shared CPU

**Total Cost: $0/month** ✅

## Next Steps

1. Set up Grafana Cloud for monitoring (free tier)
2. Implement CI/CD with GitHub Actions
3. Add A/B testing framework
4. Set up automated model retraining
5. Add caching layer (Redis free tier)

## Support

- Render Docs: https://render.com/docs
- FastAPI Docs: https://fastapi.tiangolo.com
- Project Issues: GitHub repository issues page
