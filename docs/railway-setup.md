# Railway Deployment Guide for SOUL Project

This guide will help you deploy your FastAPI SOUL Project webapp to Railway.

## Prerequisites

1. **GitHub Account**: Your code should be in a GitHub repository
2. **Railway Account**: Sign up at [railway.app](https://railway.app) using your GitHub account

## Step-by-Step Deployment

### 1. Connect Repository to Railway

1. Go to [railway.app](https://railway.app) and log in with GitHub
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Choose your SOUL_project repository
5. Railway will automatically detect it's a Python project

### 2. Environment Configuration

Railway should automatically:
- Detect Python and install dependencies from `requirements-railway.txt` (cross-platform compatible)
- Use the `Procfile` to start the application
- Set the PORT environment variable

**Important**: We use `requirements-railway.txt` instead of `requirements.txt` because:
- It removes platform-specific hashes that cause macOS → Linux compatibility issues
- It properly handles NVIDIA CUDA dependencies for Linux x86_64
- It allows pip to select the correct wheels for Railway's Linux environment

### 3. Domain Access

After deployment:
- Railway will provide a `.railway.app` domain
- You can add a custom domain in the project settings if desired
- The webapp will be accessible at `https://your-app-name.railway.app`

### 4. Monitoring

- Check the deployment logs in Railway dashboard
- Monitor application health and performance
- Set up alerts if needed

## Important Notes

- **Platform Compatibility**: This project was developed on macOS but deploys to Linux. We've created `requirements-railway.txt` without platform-specific hashes to ensure compatibility
- **CUDA Support**: PyTorch will automatically use CUDA if available on Railway, falling back to CPU
- **Model Files**: Ensure your AI model checkpoints are included in your repository or uploaded separately
- **Environment Variables**: Set any required environment variables in Railway's dashboard
- **Build Time**: Initial deployment may take 5-10 minutes due to large dependencies (PyTorch, Ray, etc.)
- **Memory Usage**: Your app uses AI models, so ensure adequate memory allocation (recommend 2GB+ RAM)

## Troubleshooting

If deployment fails:
1. Check Railway build logs for errors
2. Verify all dependencies are in `requirements-railway.txt`
3. Ensure `src/webapp.py` path is correct
4. Check that all import paths work correctly
5. **Platform Issues**: If you see hash mismatch errors, regenerate `requirements-railway.txt` with:
   ```bash
   uv export --format requirements-txt --no-hashes --output-file requirements-railway.txt
   ```

## Alternative Platforms

If Railway doesn't work well for your use case:
- **Render**: Similar to Railway, good for Python apps
- **Google Cloud Run**: More scalable, supports containers
- **AWS App Runner**: AWS equivalent
- **Heroku**: Classic platform (has resource limitations)

## Local Testing

Before deploying, test locally:
```bash
# Install dependencies
uv sync

# Run the webapp
uvicorn src.webapp:app --host 0.0.0.0 --port 8000

# Access at http://localhost:8000
``` 