# Monitor Render Deployment
# This script polls your Render service until it's live

$APP_URL = "https://ai-recommendation-system.onrender.com"
$MAX_ATTEMPTS = 60  # 10 minutes (60 attempts * 10 seconds)

Write-Host "🚀 Monitoring Render Deployment..." -ForegroundColor Cyan
Write-Host "Service URL: $APP_URL" -ForegroundColor Yellow
Write-Host "This may take 3-5 minutes for first deployment...`n" -ForegroundColor Gray

for ($i = 1; $i -le $MAX_ATTEMPTS; $i++) {
    Write-Host "[$i/$MAX_ATTEMPTS] Checking deployment status..." -NoNewline
    
    try {
        $response = Invoke-WebRequest -Uri "$APP_URL/health" -UseBasicParsing -TimeoutSec 5
        
        if ($response.StatusCode -eq 200) {
            Write-Host " ✅ SUCCESS!" -ForegroundColor Green
            Write-Host "`n🎉 Your API is LIVE!" -ForegroundColor Green
            Write-Host "`nHealth Check Response:" -ForegroundColor Cyan
            Write-Host $response.Content
            
            Write-Host "`n📊 Testing other endpoints..." -ForegroundColor Cyan
            
            # Test root endpoint
            Write-Host "`n1. Root Endpoint (/):" -ForegroundColor Yellow
            $root = Invoke-WebRequest -Uri $APP_URL -UseBasicParsing
            Write-Host $root.Content
            
            # Test recommendations
            Write-Host "`n2. Recommendations Endpoint:" -ForegroundColor Yellow
            $body = '{"user_id": 1, "k": 5}'
            $rec = Invoke-WebRequest -Uri "$APP_URL/recommend_top_k" -Method POST -Body $body -ContentType "application/json" -UseBasicParsing
            Write-Host $rec.Content
            
            Write-Host "`n✅ All endpoints working!" -ForegroundColor Green
            Write-Host "`n🌐 Access your API:" -ForegroundColor Cyan
            Write-Host "   - API Docs: $APP_URL/docs" -ForegroundColor White
            Write-Host "   - Health: $APP_URL/health" -ForegroundColor White
            Write-Host "   - Metrics: $APP_URL/metrics" -ForegroundColor White
            
            # Open docs in browser
            Write-Host "`nOpening API documentation in browser..." -ForegroundColor Gray
            Start-Process "$APP_URL/docs"
            
            break
        }
    }
    catch {
        Write-Host " ⏳ Still deploying..." -ForegroundColor Yellow
        
        if ($i -eq $MAX_ATTEMPTS) {
            Write-Host "`n❌ Deployment timeout. Check Render dashboard for details:" -ForegroundColor Red
            Write-Host "   https://dashboard.render.com/web/srv-d578nie3jp1c73aqqm60" -ForegroundColor White
        }
        else {
            Write-Host "   Waiting 10 seconds before next check..." -ForegroundColor Gray
            Start-Sleep -Seconds 10
        }
    }
}
