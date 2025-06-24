# Railway Deployment Guide

## Quick Deploy to Railway

1. **Go to [Railway.app](https://railway.app)**
2. **Sign up/Login** with GitHub
3. **Click "New Project"**
4. **Select "Deploy from GitHub repo"**
5. **Choose `DIPESHGOEL27/option-pricing-models`**
6. **Railway will auto-detect and deploy**

## Configuration

Railway will automatically:

- Detect Python app
- Install requirements.txt
- Run with gunicorn
- Assign a public URL

## Environment Variables (if needed)

- `PORT` - Automatically set by Railway
- `PYTHON_VERSION` - Set to `3.12.7` (optional)

## Expected URL Structure

- Main app: `https://your-app-name.up.railway.app/`
- API status: `https://your-app-name.up.railway.app/api/status`
- Pricing API: `https://your-app-name.up.railway.app/api/price`

## Benefits over Vercel

- 512MB size limit (vs 250MB)
- Better for ML/data science apps
- Persistent storage options
- More generous compute limits

## Alternative: Render Deployment

1. **Go to [Render.com](https://render.com)**
2. **Connect GitHub account**
3. **Create new Web Service**
4. **Choose your repository**
5. **Use these settings:**
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn api.app:app --bind 0.0.0.0:$PORT`
   - Python Version: `3.12.7`
