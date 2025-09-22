# 🤖 Real AI Deployment Guide - Connect 4 with Actual .pt Models

This guide explains how to deploy the full-stack Connect 4 game that uses your actual trained `.pt` models via a Python Flask API.

## 🏗️ Architecture Overview

```
┌─────────────────┐    HTTP API    ┌──────────────────┐
│   Web Frontend  │ ────────────► │   Flask API      │
│   (JavaScript)  │               │   (Python)       │
│                 │ ◄──────────── │                  │
│ • Game UI       │    JSON       │ • Load .pt models│
│ • User Input    │               │ • Ensemble Agent │
│ • Visualization │               │ • Move Prediction│
└─────────────────┘               └──────────────────┘
                                           │
                                           ▼
                                  ┌──────────────────┐
                                  │   Trained Models │
                                  │                  │
                                  │ • M1-CNN-750k.pt │
                                  │ • M1-CNN-700k.pt │
                                  │ • M1-CNN-650k.pt │
                                  │ • M1-CNN-600k.pt │
                                  │ • M1-CNN-550k.pt │
                                  └──────────────────┘
```

## 🚀 Quick Start

### 1. Backend Setup (Flask API)

```bash
# Navigate to the API directory
cd web_connect4_game/api

# Install Python dependencies
pip install -r requirements.txt

# Run the Flask API server
python app.py
```

### 2. Frontend Setup (Web Game)

```bash
# Navigate to the web game directory
cd web_connect4_game

# Serve the web files (choose one method):

# Option A: Python simple server
python -m http.server 8000

# Option B: Node.js serve (if you have Node.js)
npx serve .

# Option C: Any web server pointing to this directory
```

### 3. Access the Game

- **Web Game**: http://localhost:8000
- **API Health**: http://localhost:5000
- **API Info**: http://localhost:5000/api/ensemble/info

## 📋 Detailed Setup Instructions

### Prerequisites

1. **Python 3.8+** with PyTorch installed
2. **Your trained .pt model files** in the correct directories
3. **Web browser** with JavaScript enabled
4. **Network connectivity** between frontend and API

### Backend (Flask API) Setup

#### 1. Install Dependencies

```bash
cd web_connect4_game/api
pip install -r requirements.txt
```

**Required packages:**
- `Flask==2.3.3` - Web framework
- `Flask-CORS==4.0.0` - Cross-origin requests
- `numpy==1.24.3` - Numerical computing
- `torch==2.0.1` - PyTorch for model loading
- `gunicorn==21.2.0` - Production WSGI server

#### 2. Verify Model Files

Ensure your trained models are in the correct locations:

```
forza_quattro/
├── models_m1_cnn/
│   ├── m1_cnn_dqn_ep_750000.pt  ✅
│   ├── m1_cnn_dqn_ep_700000.pt  ✅
│   ├── m1_cnn_dqn_ep_650000.pt  ✅
│   ├── m1_cnn_dqn_ep_600000.pt  ✅
│   └── m1_cnn_dqn_ep_550000.pt  ✅
├── agents/
├── game/
└── ensemble_agent.py
```

#### 3. Test the API

```bash
# Start the API server
python app.py

# In another terminal, test the endpoints
curl http://localhost:5000/
curl http://localhost:5000/api/ensemble/info
```

**Expected output:**
```json
{
  "status": "healthy",
  "service": "Connect 4 Ensemble AI API",
  "ensemble_loaded": true,
  "model_info": {
    "ensemble_method": "q_value_averaging",
    "num_models": 5,
    "models": [...]
  }
}
```

### Frontend (Web Game) Setup

#### 1. Serve Web Files

The web game consists of static files that need to be served by a web server:

**Option A: Python Built-in Server**
```bash
cd web_connect4_game
python -m http.server 8000
```

**Option B: Node.js Serve**
```bash
cd web_connect4_game
npx serve . -p 8000
```

**Option C: Apache/Nginx**
Point your web server document root to the `web_connect4_game` directory.

#### 2. Configure API URL

If your API is running on a different host/port, update the API URL in the frontend:

```javascript
// In connect4-real-ai.js, line 10
const apiUrl = 'http://your-api-server:5000';  // Update this
```

## 🔧 Configuration Options

### Ensemble Configuration

#### Using Custom Models

Create a custom configuration file:

```json
{
  "name": "My-Custom-Ensemble",
  "method": "q_value_averaging",
  "models": [
    {
      "path": "models_m1_cnn/m1_cnn_dqn_ep_800000.pt",
      "weight": 0.4,
      "name": "M1-CNN-800k"
    },
    {
      "path": "models_m1_cnn/m1_cnn_dqn_ep_750000.pt",
      "weight": 0.3,
      "name": "M1-CNN-750k"
    },
    {
      "path": "models_m1_cnn/m1_cnn_dqn_ep_700000.pt",
      "weight": 0.3,
      "name": "M1-CNN-700k"
    }
  ]
}
```

Load it by modifying `app.py`:
```python
config_path = "path/to/your/custom_config.json"
ensemble_agent = load_ensemble_from_config(config_path)
```

#### Ensemble Methods

- **`q_value_averaging`** - Average Q-values across models (recommended)
- **`weighted_voting`** - Democratic vote on best action
- **`confidence_weighted`** - Weight by model confidence

### Environment Variables

Configure the API using environment variables:

```bash
export PORT=5000              # API port
export DEBUG=false            # Debug mode
export MODEL_CONFIG_PATH=/path/to/config.json
```

## 🐳 Docker Deployment

### Dockerfile for API

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY api/requirements.txt .
RUN pip install -r requirements.txt

# Copy the entire project
COPY . .

# Expose port
EXPOSE 5000

# Run the API
CMD ["python", "api/app.py"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./models_m1_cnn:/app/models_m1_cnn
    environment:
      - DEBUG=false
      
  frontend:
    image: nginx:alpine
    ports:
      - "8080:80"
    volumes:
      - ./web_connect4_game:/usr/share/nginx/html
    depends_on:
      - api
```

Deploy with:
```bash
docker-compose up -d
```

## 🌐 Production Deployment

### API Server (Backend)

#### Option 1: Gunicorn + Nginx

```bash
# Install gunicorn
pip install gunicorn

# Run with gunicorn
cd web_connect4_game/api
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

**Nginx configuration:**
```nginx
server {
    listen 80;
    server_name your-api-domain.com;
    
    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

#### Option 2: Cloud Platforms

**Heroku:**
```bash
# Create Procfile
echo "web: gunicorn -w 4 -b 0.0.0.0:\$PORT api.app:app" > Procfile

# Deploy
git add .
git commit -m "Deploy Connect 4 AI API"
git push heroku main
```

**AWS Lambda + API Gateway:**
Use `serverless` framework with `serverless-wsgi` plugin.

**Google Cloud Run:**
```bash
gcloud run deploy connect4-api \
  --source=. \
  --platform=managed \
  --region=us-central1 \
  --allow-unauthenticated
```

### Web Frontend

#### Option 1: GitHub Pages

```bash
# Copy web files to your GitHub Pages repo
cp -r web_connect4_game/* /path/to/your-github-pages-repo/connect4/

# Update API URL in connect4-real-ai.js
# Update to your production API URL

# Commit and push
cd /path/to/your-github-pages-repo
git add .
git commit -m "Add Connect 4 with Real AI"
git push origin main
```

#### Option 2: Netlify

```bash
# Drag and drop the web_connect4_game folder to Netlify
# Or use Netlify CLI
netlify deploy --prod --dir=web_connect4_game
```

#### Option 3: Vercel

```bash
cd web_connect4_game
npx vercel --prod
```

## 🔒 Security Considerations

### API Security

1. **CORS Configuration:**
```python
# In app.py, configure CORS for production
CORS(app, origins=["https://your-domain.com"])
```

2. **Rate Limiting:**
```python
from flask_limiter import Limiter
limiter = Limiter(app, key_func=lambda: request.remote_addr)

@app.route('/api/game/move', methods=['POST'])
@limiter.limit("10 per minute")  # Limit API calls
def get_ai_move():
    # ... existing code
```

3. **API Authentication (Optional):**
```python
# Add API key authentication for production use
API_KEY = os.environ.get('API_KEY')

@app.before_request
def check_api_key():
    if request.endpoint and 'api' in request.endpoint:
        if request.headers.get('X-API-Key') != API_KEY:
            return jsonify({'error': 'Invalid API key'}), 401
```

### Model Protection

1. **File Permissions:**
```bash
# Restrict access to model files
chmod 600 models_m1_cnn/*.pt
```

2. **Environment Isolation:**
```bash
# Use virtual environments
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

## 📊 Monitoring and Logging

### API Monitoring

```python
# Add logging to app.py
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)

# Log all API requests
@app.before_request
def log_request():
    app.logger.info(f'{request.method} {request.path} - {request.remote_addr}')
```

### Performance Metrics

```python
import time

@app.before_request
def before_request():
    g.start_time = time.time()

@app.after_request
def after_request(response):
    duration = time.time() - g.start_time
    app.logger.info(f'Request took {duration:.3f}s')
    return response
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Models Not Loading
```
❌ Failed to load M1-CNN-750k: No such file or directory
```

**Solution:**
- Verify model file paths in configuration
- Check file permissions
- Ensure PyTorch version compatibility

#### 2. API Connection Failed
```
⚠️ Real AI API not available, falling back to heuristic AI
```

**Solution:**
- Check if Flask API is running: `curl http://localhost:5000/`
- Verify CORS configuration
- Check browser console for network errors

#### 3. Memory Issues
```
❌ CUDA out of memory / RuntimeError: Unable to allocate memory
```

**Solution:**
- Use CPU-only inference: `torch.device('cpu')`
- Reduce batch size in model loading
- Limit concurrent requests

#### 4. Import Errors
```
❌ Failed to import required modules: No module named 'ensemble_agent'
```

**Solution:**
- Verify Python path configuration
- Install all dependencies: `pip install -r requirements.txt`
- Check that you're running from the correct directory

### Debug Mode

Enable debug logging:

```bash
# Start API in debug mode
export DEBUG=true
python app.py

# Check browser console for frontend errors
# Open browser dev tools and look for JavaScript errors
```

## 📈 Performance Optimization

### API Performance

1. **Model Caching:**
```python
# Models are loaded once at startup and cached
# Avoid reloading models for each request
```

2. **Async Processing:**
```python
from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=4)

# For high-traffic scenarios, consider async frameworks
# like FastAPI instead of Flask
```

3. **Request Batching:**
```python
# For multiple game instances, batch model inference
# to improve GPU utilization
```

### Frontend Performance

1. **API Caching:**
```javascript
// Cache model info to avoid repeated API calls
const modelInfoCache = localStorage.getItem('modelInfo');
```

2. **Request Debouncing:**
```javascript
// Prevent rapid successive API calls
const debounce = (func, wait) => {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
};
```

## 🎯 Next Steps

1. **Monitor Performance:** Track API response times and error rates
2. **Scale as Needed:** Add load balancing and multiple API instances
3. **Enhance UI:** Add real-time decision breakdowns and model insights
4. **A/B Testing:** Compare different ensemble configurations
5. **Analytics:** Track game statistics and player behavior

Your Connect 4 game now uses the actual trained neural networks that won your tournament! Players are facing the real deal. 🏆