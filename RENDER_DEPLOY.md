# 🚀 Render.com Deployment Instructions

Your code is now on GitHub! Follow these steps to deploy to Render.com:

## Step 1: Sign Up / Log In to Render.com

1. Go to **[Render.com](https://render.com)**
2. Click **"Get Started"** or **"Sign In"**
3. Sign in with your GitHub account (recommended)

## Step 2: Create New Web Service

1. Once logged in, click the **"New +"** button in the top right
2. Select **"Web Service"**

## Step 3: Connect Your Repository

1. You'll see a list of your GitHub repositories
2. Find **`ai-recommendation-system`**
3. Click **"Connect"** next to it

   > If you don't see your repository:
   > - Click "Configure account" 
   > - Grant Render access to your repositories
   > - Return and refresh the list

## Step 4: Configure the Service

Render will automatically detect your `render.yaml` file, but verify these settings:

### Basic Settings:
- **Name**: `ai-recommendation-engine` (or your choice)
- **Region**: `Oregon (US West)` ✅ Free tier available
- **Branch**: `main`
- **Runtime**: `Python 3`

### Build & Deploy (Auto-filled from render.yaml):
- **Build Command**: 
  ```bash
  pip install --upgrade pip && pip install -r requirements.txt
  ```
- **Start Command**: 
  ```bash
  uvicorn src.serving.main:app --host 0.0.0.0 --port $PORT --workers 1
  ```

### Plan:
- **Instance Type**: Select **"Free"** ✅ $0/month

### Advanced Settings (Optional):
- **Health Check Path**: `/health` (already configured in render.yaml)
- **Auto-Deploy**: `Yes` (already enabled - pushes to main will auto-deploy)

## Step 5: Deploy!

1. Review all settings
2. Click **"Create Web Service"** at the bottom
3. Render will start building and deploying your app

## Step 6: Monitor Deployment

You'll see the deployment logs in real-time:

```
==> Cloning from https://github.com/23091a05c0-dotcom/ai-recommendation-system...
==> Downloading cache...
==> Running build command: pip install --upgrade pip && pip install -r requirements.txt
==> Installing dependencies...
==> Build successful!
==> Starting service with: uvicorn src.serving.main:app --host 0.0.0.0 --port $PORT --workers 1
==> Loading model and resources...
==> Loaded metadata: 610 users, 9724 items
==> Model loaded successfully
==> Your service is live! 🎉
```

**Expected deployment time**: 3-5 minutes

## Step 7: Get Your Deployment URL

Once deployed, you'll see:
- **Status**: 🟢 **Live**
- **URL**: `https://ai-recommendation-engine.onrender.com` (or your chosen name)

Copy this URL!

## Step 8: Test Your Deployed API

### Test in Browser:
```
https://your-app-name.onrender.com/
https://your-app-name.onrender.com/health
https://your-app-name.onrender.com/docs
```

### Test with curl:
```bash
# Health check
curl https://your-app-name.onrender.com/health

# Get recommendations
curl -X POST https://your-app-name.onrender.com/recommend_top_k \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "k": 5}'
```

## Expected Response:

### Health Endpoint:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "cache_size": 0
}
```

### Recommendations Endpoint:
```json
{
  "user_id": 1,
  "recommendations": [
    {
      "item_id": 123,
      "original_id": 456,
      "title": "Toy Story (1995)",
      "score": 4.8
    },
    ...
  ]
}
```

## Important Notes

### ⚠️ Free Tier Limitations:
- **Cold Starts**: Service sleeps after 15 minutes of inactivity
- **First Request**: May take 30-60 seconds to wake up
- **Memory**: 512MB RAM limit
- **Hours**: 750 hours/month (enough for 24/7 operation)

### 🔄 Auto-Deploy:
Every time you push to the `main` branch on GitHub, Render will automatically redeploy:
```bash
git add .
git commit -m "Update model"
git push
# Render automatically deploys! 🚀
```

### 📊 Monitoring:
- **Logs**: View in Render dashboard → Your service → Logs
- **Metrics**: Access at `/metrics` endpoint
- **Health**: Monitor at `/health` endpoint

## Troubleshooting

### Issue: Build fails
**Check logs for:**
- Missing dependencies → Update `requirements.txt`
- Python version mismatch → Verify `PYTHON_VERSION` env var

### Issue: Service won't start
**Common causes:**
- Model file not found → Verify it's in the Git repository
- Port binding → Ensure using `$PORT` environment variable
- Out of memory → Model might be too large for free tier

### Issue: 404 errors
**Verify:**
- Service is "Live" in dashboard
- URL is correct
- Health check endpoint works first

## Next Steps After Deployment

1. ✅ Update README.md with your live URL
2. ✅ Test all API endpoints
3. ✅ Set up monitoring with Grafana Cloud (optional, free tier)
4. ✅ Share your API with others!

## Support

- **Render Docs**: https://render.com/docs
- **Render Community**: https://community.render.com
- **Your Deployment Guide**: See `DEPLOYMENT.md` in your repository

---

**🎉 Congratulations! Your AI Recommendation System is going live!**

Total Cost: **$0/month** using Render's free tier! 💰
