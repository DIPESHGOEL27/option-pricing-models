# Google Cloud Run Deployment Guide

## Prerequisites
1. Install [Google Cloud CLI](https://cloud.google.com/sdk/docs/install)
2. Create a Google Cloud account (free $300 credits)

## Deployment Steps

### 1. Setup
```bash
# Login to Google Cloud
gcloud auth login

# Create a new project (optional)
gcloud projects create option-pricing-models --name="Option Pricing Models"

# Set the project
gcloud config set project option-pricing-models

# Enable required APIs
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
```

### 2. Deploy
```bash
# Build and deploy in one command
gcloud run deploy option-pricing-models \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1 \
  --timeout 300

# Or build Docker image separately
gcloud builds submit --tag gcr.io/option-pricing-models/option-pricing-app
gcloud run deploy --image gcr.io/option-pricing-models/option-pricing-app --platform managed
```

### 3. Benefits
- **2GB RAM** (vs 250MB Vercel limit)
- **Serverless scaling**
- **Free tier**: 2 million requests/month
- **Full ML library support**
- **Custom domains**
- **Environment variables**

### 4. Cost
- Free tier covers most usage
- Pay per request after free tier
- Much cheaper than dedicated servers

## Alternative: Cloud Run Jobs
For batch processing or model training:
```bash
gcloud run jobs create option-pricing-job \
  --image gcr.io/option-pricing-models/option-pricing-app \
  --region us-central1 \
  --memory 4Gi \
  --cpu 2
```
