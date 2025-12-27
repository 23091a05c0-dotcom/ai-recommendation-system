# Terminal Deployment Monitoring Guide

## Current Status

✅ **Your code is already deploying!** 

When you pushed the fix to GitHub, Render automatically started a new deployment because `autoDeploy: true` is enabled in your `render.yaml`.

## Monitor Deployment from Terminal

While Render doesn't have a free-tier CLI, you can monitor your deployment using these methods:

### Option 1: Watch Deployment in Browser
Your deployment is live at:
```
https://dashboard.render.com/web/srv-d578nie3jp1c73aqqm60/deploys
```

### Option 2: Check Deployment Status via API (Once Live)

After deployment completes, test from terminal:

```bash
# Set your app URL (replace with your actual URL after deployment)
$APP_URL = "https://ai-recommendation-system.onrender.com"

# Health check
curl $APP_URL/health

# Root endpoint
curl $APP_URL/

# Get recommendations
curl -X POST "$APP_URL/recommend_top_k" `
  -H "Content-Type: application/json" `
  -d '{"user_id": 1, "k": 5}'
```

### Option 3: Install Render CLI (Optional)

For more control, install the Render CLI:

```bash
# Install via npm
npm install -g @render/cli

# Login
render login

# List services
render services list

# View logs
render logs --service srv-d578nie3jp1c73aqqm60

# Trigger manual deploy
render deploy --service srv-d578nie3jp1c73aqqm60
```

## What's Happening Now

Your deployment is processing through these stages:

1. ✅ **Cloning** - Fetching code from GitHub
2. ✅ **Installing Python 3.13.4** - Setting up environment
3. 🔄 **Installing dependencies** - Running `pip install -r requirements.txt`
4. ⏳ **Building** - Preparing your application
5. ⏳ **Starting** - Running `uvicorn src.serving.main:app`
6. ⏳ **Health check** - Verifying `/health` endpoint
7. ⏳ **Live** - Your API is deployed!

## Expected Timeline

- **Total deployment time**: 3-5 minutes
- **Build phase**: 2-3 minutes (installing PyTorch takes time)
- **Startup phase**: 30-60 seconds

## How to Know When It's Done

### Method 1: Check the Render Dashboard
Watch the logs in your browser - you'll see:
```
==> Your service is live 🎉
```

### Method 2: Poll the Health Endpoint

```bash
# Keep checking until it responds (Windows PowerShell)
while ($true) {
    try {
        $response = Invoke-WebRequest -Uri "https://ai-recommendation-system.onrender.com/health" -UseBasicParsing
        Write-Host "✅ DEPLOYED! Response: $($response.Content)"
        break
    } catch {
        Write-Host "⏳ Still deploying... (checking again in 10s)"
        Start-Sleep -Seconds 10
    }
}
```

### Method 3: Use curl with retry

```bash
# Check every 10 seconds until successful
for ($i=1; $i -le 30; $i++) {
    Write-Host "Attempt $i/30..."
    curl https://ai-recommendation-system.onrender.com/health
    if ($LASTEXITCODE -eq 0) { 
        Write-Host "✅ Deployment successful!"
        break 
    }
    Start-Sleep -Seconds 10
}
```

## After Deployment Succeeds

### Test All Endpoints

```bash
# Set your app URL
$APP_URL = "https://ai-recommendation-system.onrender.com"

# 1. Root endpoint
Write-Host "`n=== Testing Root Endpoint ==="
curl $APP_URL/

# 2. Health check
Write-Host "`n=== Testing Health Endpoint ==="
curl $APP_URL/health

# 3. Metrics
Write-Host "`n=== Testing Metrics Endpoint ==="
curl $APP_URL/metrics

# 4. Get recommendations
Write-Host "`n=== Testing Recommendations ==="
curl -X POST "$APP_URL/recommend_top_k" `
  -H "Content-Type: application/json" `
  -d '{"user_id": 1, "k": 5}'

# 5. Interactive docs (open in browser)
Start-Process "$APP_URL/docs"
```

### Save Your Deployment URL

```bash
# Add to your environment (PowerShell profile)
$env:AI_RECSYS_URL = "https://ai-recommendation-system.onrender.com"

# Or create a config file
@{
    API_URL = "https://ai-recommendation-system.onrender.com"
    DEPLOYED_AT = (Get-Date).ToString()
} | ConvertTo-Json | Out-File deployment-config.json
```

## Troubleshooting

### If deployment fails again:

1. **Check the logs** in Render dashboard
2. **Common issues**:
   - Dependency conflicts → Check `requirements.txt`
   - Model file too large → Use Git LFS
   - Memory limit exceeded → Reduce model size
   - Port binding → Ensure using `$PORT` variable

### View logs from terminal (with Render CLI):

```bash
render logs --service srv-d578nie3jp1c73aqqm60 --tail 100
```

## Quick Deploy Commands Summary

```bash
# Make changes locally
git add .
git commit -m "Your changes"
git push

# Render auto-deploys! Monitor with:
# 1. Browser: https://dashboard.render.com/web/srv-d578nie3jp1c73aqqm60
# 2. Terminal: Keep checking health endpoint
# 3. CLI: render logs --service srv-d578nie3jp1c73aqqm60
```

## Your Current Deployment

**Service ID**: `srv-d578nie3jp1c73aqqm60`
**Repository**: `https://github.com/23091a05c0-dotcom/ai-recommendation-system`
**Branch**: `main`
**Latest Commit**: Fixed PyTorch version for Python 3.13 compatibility

**Status**: 🔄 Deploying now...

Check your Render dashboard to see real-time progress!
