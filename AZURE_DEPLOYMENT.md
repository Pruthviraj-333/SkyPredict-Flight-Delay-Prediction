# SkyPredict — Azure Deployment Guide

> Deploy the flight delay prediction app to Microsoft Azure for **free**.

---

## Architecture

```
┌─────────────────────────────┐     ┌──────────────────────────────┐
│   Azure Static Web Apps     │     │   Azure App Service (B1)     │
│   (Frontend — Free Tier)    │────▶│   (Backend — $200 credit)    │
│                             │     │                              │
│   Next.js 16 + React 19    │     │   FastAPI + ML Models        │
│   Firebase Auth (client)    │     │   Docker Container           │
└─────────────────────────────┘     └──────────────────────────────┘
                                              │
                                    ┌─────────┴─────────┐
                                    │                   │
                              Open-Meteo API      AviationStack API
                              (Free, no key)      (Free: 100 req/mo)
```

---

## Prerequisites

- [Azure Account](https://azure.microsoft.com/free/) (free — gives $200 credit for 30 days)
- [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli) installed
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed
- GitHub repository with this project

---

## Step 1: Azure Account Setup

1. Go to [azure.microsoft.com/free](https://azure.microsoft.com/free/)
2. Sign up with your Microsoft/GitHub account
3. You'll get **$200 free credit** for 30 days

```bash
# Login to Azure CLI
az login
```

---

## Step 2: Handle Model Files (Important!)

Your `.gitignore` excludes `models/*.pkl` and `data/**/*.csv`. You need these for deployment.

### Option A: Git LFS (Recommended for Teams)

```bash
# Install Git LFS
git lfs install

# Track model files
git lfs track "models/*.pkl"
git lfs track "data/**/*.csv"

# Add the tracking config
git add .gitattributes

# Add model files (LFS will handle them)
git add models/ data/
git commit -m "Add models via Git LFS"
git push
```

### Option B: Temporary Unignore for Docker Build

If you're building the Docker image **locally**, the files just need to exist on your machine. No git changes needed — Docker reads from disk.

---

## Step 3: Deploy Backend

### 3.1 Create Azure Resources

```bash
# Set variables
RESOURCE_GROUP="skypredict-rg"
LOCATION="centralindia"          # Closest to you
APP_NAME="skypredict-api"        # Must be globally unique
ACR_NAME="skypredictacr"        # Must be globally unique, alphanumeric only

# Create resource group
az group create --name $RESOURCE_GROUP --location $LOCATION

# Create Azure Container Registry (Basic tier — included in free credit)
az acr create --resource-group $RESOURCE_GROUP --name $ACR_NAME --sku Basic --admin-enabled true

# Get ACR credentials
az acr credential show --name $ACR_NAME
```

### 3.2 Build & Push Docker Image

```bash
# Login to ACR
az acr login --name $ACR_NAME

# Build the Docker image
docker build -t $ACR_NAME.azurecr.io/skypredict-backend:latest .

# Push to Azure Container Registry
docker push $ACR_NAME.azurecr.io/skypredict-backend:latest
```

### 3.3 Create App Service

```bash
# Create App Service Plan (B1 tier — uses your $200 credit)
az appservice plan create \
  --name skypredict-plan \
  --resource-group $RESOURCE_GROUP \
  --sku B1 \
  --is-linux

# Create Web App with Docker container
az webapp create \
  --resource-group $RESOURCE_GROUP \
  --plan skypredict-plan \
  --name $APP_NAME \
  --deployment-container-image-name $ACR_NAME.azurecr.io/skypredict-backend:latest

# Configure ACR credentials for the web app
ACR_PASSWORD=$(az acr credential show --name $ACR_NAME --query "passwords[0].value" -o tsv)
az webapp config container set \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --docker-registry-server-url https://$ACR_NAME.azurecr.io \
  --docker-registry-server-user $ACR_NAME \
  --docker-registry-server-password $ACR_PASSWORD
```

### 3.4 Set Environment Variables

```bash
az webapp config appsettings set \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --settings \
    AVIATIONSTACK_API_KEY="your-api-key-here" \
    WEBSITES_PORT="8000"
```

### 3.5 Test the Backend

```bash
# Wait ~2 minutes for startup, then:
curl https://$APP_NAME.azurewebsites.net/api/health
```

---

## Step 4: Deploy Frontend

### 4.1 Create Static Web App

```bash
az staticwebapp create \
  --name "skypredict-frontend" \
  --resource-group $RESOURCE_GROUP \
  --source "https://github.com/YOUR_USERNAME/YOUR_REPO" \
  --branch main \
  --app-location "/frontend" \
  --output-location ".next" \
  --login-with-github
```

### 4.2 Set Frontend Environment Variables

Go to **Azure Portal** → Static Web Apps → "skypredict-frontend" → **Configuration** → Add:

| Name | Value |
|------|-------|
| `NEXT_PUBLIC_API_URL` | `https://skypredict-api.azurewebsites.net` |
| `NEXT_PUBLIC_FIREBASE_API_KEY` | *(your Firebase config)* |
| `NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN` | *(your Firebase config)* |
| `NEXT_PUBLIC_FIREBASE_PROJECT_ID` | *(your Firebase config)* |
| `NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET` | *(your Firebase config)* |
| `NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID` | *(your Firebase config)* |
| `NEXT_PUBLIC_FIREBASE_APP_ID` | *(your Firebase config)* |

### 4.3 Update Backend CORS

After getting your Static Web App URL, update CORS in the backend:

```bash
az webapp cors add \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --allowed-origins "https://skypredict-frontend.azurestaticapps.net"
```

---

## Step 5: Set Up GitHub Actions (CI/CD)

### 5.1 Add GitHub Secrets

Go to your GitHub repo → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**:

| Secret Name | Value |
|-------------|-------|
| `AZURE_CREDENTIALS` | Output of `az ad sp create-for-rbac` (see below) |
| `AZURE_CONTAINER_REGISTRY` | `skypredictacr.azurecr.io` |
| `ACR_USERNAME` | From `az acr credential show` |
| `ACR_PASSWORD` | From `az acr credential show` |
| `AZURE_WEBAPP_NAME` | `skypredict-api` |
| `AVIATIONSTACK_API_KEY` | Your API key |
| `AZURE_STATIC_WEB_APPS_API_TOKEN` | From Azure Portal → Static Web App → Manage deployment token |
| `NEXT_PUBLIC_FIREBASE_*` | All 6 Firebase config values |
| `NEXT_PUBLIC_API_URL` | `https://skypredict-api.azurewebsites.net` |

### 5.2 Create Azure Service Principal

```bash
az ad sp create-for-rbac \
  --name "skypredict-github" \
  --role contributor \
  --scopes /subscriptions/YOUR_SUBSCRIPTION_ID/resourceGroups/skypredict-rg \
  --json-auth
```

Copy the entire JSON output as the `AZURE_CREDENTIALS` secret.

---

## Cost Tracker

| Resource | Tier | Monthly Cost |
|----------|------|-------------|
| App Service Plan | B1 (Linux) | ~₹1,000/mo (covered by $200 credit for 30 days) |
| Container Registry | Basic | ~₹400/mo (covered by credit) |
| Static Web Apps | Free | ₹0 |
| **Total** | | **₹0 for first 30 days** |

> **After 30 days:** If you want to continue for free, switch the backend to F1 tier (non-Docker, Python runtime) using `startup.sh` instead of Docker.

---

## Switching to Free F1 Tier (After Credits Expire)

```bash
# Downgrade to free tier (no Docker support)
az appservice plan update --name skypredict-plan --resource-group $RESOURCE_GROUP --sku F1

# Set startup command for Python runtime
az webapp config set \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --startup-file "startup.sh"
```

> **Note:** F1 tier doesn't support Docker containers. You'll deploy code directly using ZIP deploy, and the `startup.sh` file will be used instead.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Cold start takes 30s+ | Normal for free/B1 tier with 65MB models. First request after idle is slow. |
| Memory errors | Models expand in RAM. Monitor via Azure Portal → App Service → Metrics. |
| CORS errors | Ensure backend CORS includes your Static Web App URL. |
| Firebase auth fails | Check `Cross-Origin-Opener-Policy` header in `staticwebapp.config.json`. |
| Models not found | Ensure Docker build has access to `models/` directory. |
