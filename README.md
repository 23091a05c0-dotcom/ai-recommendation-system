# AI-Powered Recommendation Engine

[![Deploy Status](https://img.shields.io/badge/deploy-ready-brightgreen)](https://render.com)
[![Python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-EE4C2C)](https://pytorch.org/)

**Technologies:** Python, PyTorch, FastAPI, Prometheus, Render.com

## 🚀 Live Demo

**API Endpoint:** `https://[your-app].onrender.com` (after deployment)

**Interactive Docs:** `https://[your-app].onrender.com/docs`

## 📋 Project Overview

A production-ready deep learning recommendation system built with:
- **Neural Collaborative Filtering** using PyTorch
- **FastAPI** for high-performance serving
- **Prometheus** metrics for monitoring
- **Free-tier deployment** on Render.com
- **MovieLens dataset** for training and evaluation

### Key Features
- ✅ Real-time movie recommendations
- ✅ RESTful API with automatic documentation
- ✅ Prometheus metrics integration
- ✅ Health monitoring endpoints
- ✅ In-memory caching for performance
- ✅ Production-ready deployment configuration

## 🎯 Quick Start

### Test the Deployed API

```bash
# Health check
curl https://[your-app].onrender.com/health

# Get top-5 recommendations for user 1
curl -X POST https://[your-app].onrender.com/recommend_top_k \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "k": 5}'
```

### Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn src.serving.main:app --reload

# Visit http://localhost:8000/docs
```

## 📦 Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for complete deployment instructions.

**Quick Deploy to Render.com:**
1. Push code to GitHub
2. Connect repository to Render.com
3. Deploy automatically using `render.yaml`

## 🏗️ Architecture

```
User Request → FastAPI → Model Inference → Cached Response
                  ↓
            Prometheus Metrics
```

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information and documentation |
| `/health` | GET | Health check and status |
| `/metrics` | GET | Prometheus metrics |
| `/predict` | POST | Get predictions for specific items |
| `/recommend_top_k` | POST | Get top-K recommendations |
| `/docs` | GET | Interactive API documentation |

## 🔧 Technologies

- **ML Framework:** PyTorch 2.1
- **API Framework:** FastAPI 0.109
- **Monitoring:** Prometheus
- **Deployment:** Render.com (Free Tier)
- **Dataset:** MovieLens 25M

## 📈 Performance Metrics

- **Latency:** <100ms for cached requests
- **Throughput:** 100+ requests/second
- **Model Size:** ~10MB
- **Memory Usage:** <512MB (free tier compatible)

## 📚 Documentation

- [Deployment Guide](DEPLOYMENT.md) - Step-by-step deployment instructions
- [Free Tier Guide](FREE_TIER_GUIDE.md) - Building with $0 budget
- [API Docs](https://[your-app].onrender.com/docs) - Interactive API documentation

## 🛠️ Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Start development server
uvicorn src.serving.main:app --reload --port 8000
```

## 📄 License

MIT License - feel free to use for your projects!

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a PR.
