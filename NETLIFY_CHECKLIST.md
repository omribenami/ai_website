# Netlify Deployment Checklist

## Configuration Files
- [x] netlify.toml in client directory
- [x] _redirects file in client/public directory

## Pre-Deployment Steps
1. Install dependencies:
   ```bash
   cd /path/to/ai_website/client
   npm install
   ```

2. Create production build:
   ```bash
   npm run build
   ```

3. Set environment variables:
   - REACT_APP_API_URL: URL of your backend API

## Deployment Options
1. **GitHub + Netlify UI**:
   - Push code to GitHub
   - Connect repository in Netlify UI
   - Configure build settings
   - Deploy

2. **Netlify CLI**:
   - Install Netlify CLI: `npm install netlify-cli -g`
   - Login: `netlify login`
   - Initialize: `netlify init`
   - Deploy: `netlify deploy --prod`

## Post-Deployment
- Configure custom domain (if needed)
- Verify all routes are working
- Test API connections
- Set up continuous deployment

## Important Notes
- Backend must be deployed separately
- CORS must be configured on backend to allow requests from Netlify domain
- Environment variables must be set in Netlify UI or via CLI
